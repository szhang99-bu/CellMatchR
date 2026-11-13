"""Prepare reviewed CellMatchR masks and fine-tune a selected model.

Training writes a new model folder. Source masks and project validation files
are read only. Heavy training libraries are imported only for their backend.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from .roi_editor import _find_mask, _find_raw
from .validation_state import get_change_state, get_validation_tsv_path

BACKENDS = ("microsam", "cellpose", "stardist")
PREPROCESS_MODES = ("local_bg_subtract", "local_bg_subtract_mild",
                    "raw_percentile", "tile_clahe_mild", "tile_clahe",
                    "brightfield_invert")
TRAINING_PRESETS = {
    "High recall": {
        "description": "Repeat samples with human-added missed cells. Compare permissive prediction settings on held-out images before use.",
        "epochs": 20, "learning_rate": 1e-5, "validation_percent": 25,
        "sampling_focus": "missed_cells",
    },
    "Default training": {
        "description": "Use every reviewed sample once with the standard learning rate and a separate validation set.",
        "epochs": 20, "learning_rate": 1e-5, "validation_percent": 20,
        "sampling_focus": "none",
    },
    "High precision": {
        "description": "Repeat samples with rejected false predictions. Check held-out recall so real cells are still found.",
        "epochs": 20, "learning_rate": 1e-5, "validation_percent": 25,
        "sampling_focus": "rejected_cells",
    },
    "Gentle follow-up": {
        "description": "Continue training a saved model with a smaller learning rate and fewer epochs to limit large changes.",
        "epochs": 8, "learning_rate": 3e-6, "validation_percent": 20,
        "sampling_focus": "none",
    },
    "Recall follow-up": {
        "description": "Continue a saved model gently while repeating reviewed samples with manually added cells.",
        "epochs": 8, "learning_rate": 3e-6, "validation_percent": 25,
        "sampling_focus": "missed_cells",
    },
    "Precision follow-up": {
        "description": "Continue a saved model gently while repeating samples with rejected predictions; check missed cells afterward.",
        "epochs": 8, "learning_rate": 3e-6, "validation_percent": 25,
        "sampling_focus": "rejected_cells",
    },
    "Short trial": {
        "description": "Run five epochs to check data and training behavior before committing to a longer run.",
        "epochs": 5, "learning_rate": 1e-5, "validation_percent": 20,
        "sampling_focus": "none",
    },
}


def model_library_root() -> Path:
    """Shared, visible model folder; environment override supports isolated tests."""
    override = os.environ.get("CELLMATCHR_MODEL_LIBRARY")
    return Path(override).expanduser() if override else Path.home() / "CellMatchR" / "Models"


def available_channels(seg_folder: str) -> list[int]:
    """Return channel IDs present in a segmentation run's prediction masks."""
    root = Path(seg_folder)
    found = set()
    for mask in root.glob("*/*_predict_original_ch*.tif"):
        match = re.search(r"_predict_original_ch(\d+)(?:_|\.)", mask.name)
        if match:
            found.add(int(match.group(1)))
    return sorted(found)


def list_trained_models(project: str, backend: str | None = None,
                        include_missing: bool = False) -> list[dict]:
    """Read shared models and older project-local records without modifying either."""
    roots = [(model_library_root(), "Shared library")]
    if project:
        legacy = Path(project) / "Trained_Models"
        if legacy != roots[0][0]:
            roots.append((legacy, "This project"))
    found = []
    for root, location in roots:
        if not root.is_dir():
            continue
        for record in sorted(root.glob("*/model.json"), reverse=True):
            try:
                data = json.loads(record.read_text())
                kind = data["backend"]
                model_path = Path(data["model_path"])
                if not model_path.exists() and data.get("model_relative_path"):
                    model_path = record.parent / data["model_relative_path"]
                if not model_path.exists() and record.parent.name in model_path.parts:
                    parts = model_path.parts
                    index = len(parts) - 1 - parts[::-1].index(record.parent.name)
                    model_path = record.parent.joinpath(*parts[index + 1:])
                available = model_path.exists()
                if kind not in BACKENDS or (backend and kind != backend) or (not available and not include_missing):
                    continue
                data["name"] = str(data.get("name") or record.parent.name)
                data["model_path"] = str(model_path)
                data["record_folder"] = record.parent.name
                data["record_dir"] = str(record.parent)
                data["location"] = location
                data["available"] = available
                found.append(data)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return found


def registered_microsam_foreground_threshold(model_path: str,
                                              project: str = "") -> float | None:
    """Return a tested proposal setting only for its registered Micro-SAM model."""
    if not model_path:
        return None
    selected = os.path.normcase(os.path.realpath(model_path))
    for record in list_trained_models(project, "microsam"):
        if os.path.normcase(os.path.realpath(record["model_path"])) != selected:
            continue
        suggestion = record.get("suggested_inference")
        if not isinstance(suggestion, dict):
            return None
        try:
            threshold = float(suggestion["foreground_threshold"])
        except (KeyError, TypeError, ValueError):
            return None
        return threshold if 0.0 <= threshold <= 1.0 else None
    return None


def delete_managed_model(record_dir: str, project: str) -> None:
    """Remove one registered model folder after the UI's two confirmations."""
    folder = Path(record_dir)
    roots = [model_library_root()]
    if project:
        roots.append(Path(project) / "Trained_Models")
    if (folder.is_symlink() or not (folder / "model.json").is_file() or
            not any(folder.resolve().parent == root.resolve() for root in roots)):
        raise ValueError("Choose a model folder managed by CellMatchR")
    shutil.rmtree(folder)


def validate_advanced(backend: str, advanced: dict | None) -> dict:
    """Validate optional overrides before creating a model output folder."""
    settings = dict(advanced or {})
    allowed = {"learning_rate", "batch_size", "patch_size", "validation_percent",
               "loss", "freeze_encoder", "sampling_focus"}
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown advanced training setting: {', '.join(sorted(unknown))}")
    if "learning_rate" in settings:
        value = float(settings["learning_rate"])
        if not 0 < value <= 1:
            raise ValueError("Learning rate must be greater than 0 and at most 1")
        settings["learning_rate"] = value
    for key, limit in (("batch_size", 64), ("patch_size", 2048)):
        if key in settings:
            value = int(settings[key])
            if not 1 <= value <= limit or (key == "patch_size" and (value < 64 or value % 16)):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be 1–{limit}"
                                 + (" and a multiple of 16 (at least 64)" if key == "patch_size" else ""))
            settings[key] = value
    if backend == "cellpose" and "patch_size" in settings and settings["patch_size"] != 256:
        raise ValueError("Cellpose-SAM requires a fixed 256-pixel training patch")
    if "validation_percent" in settings:
        value = int(settings["validation_percent"])
        if not 5 <= value <= 50:
            raise ValueError("Validation percent must be 5–50")
        settings["validation_percent"] = value
    if "loss" in settings:
        if backend != "stardist" or settings["loss"] not in ("mae", "mse"):
            raise ValueError("Only StarDist supports choosing MAE or MSE distance loss")
    if "freeze_encoder" in settings:
        if backend != "microsam" or not isinstance(settings["freeze_encoder"], bool):
            raise ValueError("Freeze image encoder is only available for Micro-SAM")
    if "sampling_focus" in settings and settings["sampling_focus"] not in (
            "none", "missed_cells", "rejected_cells"):
        raise ValueError("Sampling focus must be none, missed cells, or rejected cells")
    return settings


def _review_complete(sample: Path, channel: int) -> bool:
    validation = sample / "ROI Validation Results"
    if (validation / f"_validation_complete_ch{channel}").exists():
        return True
    if not (validation / "_validation_complete").exists():
        return False
    # A channel-neutral old marker cannot identify which of several masks it
    # reviewed. Use it only for a genuinely single-channel sample.
    return len(list(sample.glob("*_predict_original_ch*.tif"))) == 1


def _decisions(path: str) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        if not {"roi_id", "decision"}.issubset(fields):
            raise ValueError(f"Validation file lacks ROI_ID or Decision: {path}")
        result = {}
        for row in reader:
            try:
                number = float(row[fields["roi_id"]])
                if not number.is_integer() or number <= 0:
                    raise ValueError("ROI_ID must be a positive whole number")
                label_id = int(number)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ROI_ID in {path}") from exc
            decision = str(row[fields["decision"]] or "").strip().upper()
            if label_id in result:
                raise ValueError(f"Duplicate ROI_ID {label_id} in {path}")
            result[label_id] = decision
    return result


def collect_examples(seg_folder: str, channel: int) -> tuple[list[dict], list[str]]:
    """Find completely reviewed raw/mask pairs, with explicit YES decisions."""
    root = Path(seg_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Segmentation run not found: {root}")
    examples, skipped = [], []
    for sample in sorted(root.iterdir()):
        if not sample.is_dir() or sample.name.startswith((".", "DEBUG")):
            continue
        raw, mask = _find_raw(str(sample), channel), _find_mask(str(sample), channel)
        if not raw or not mask:
            skipped.append(f"{sample.name}: raw image or channel mask missing")
            continue
        state = get_change_state(str(sample), channel, mask)
        if state["status"] in ("pending", "unknown") or not _review_complete(sample, channel):
            skipped.append(f"{sample.name}: finish channel {channel} validation first")
            continue
        review = get_validation_tsv_path(str(sample), channel)
        if not review:
            skipped.append(f"{sample.name}: validation decisions missing")
            continue
        decisions = _decisions(review)
        labels = tifffile.imread(mask)
        present = {int(v) for v in np.unique(labels) if v > 0}
        if present - decisions.keys():
            skipped.append(f"{sample.name}: some current cells have no decision")
            continue
        if any(decisions[v] not in {"YES", "NO", "MAYBE", "ARTIFACT"}
               for v in present):
            skipped.append(f"{sample.name}: custom decision words need an explicit mapping")
            continue
        if any(decisions[v] == "MAYBE" for v in present):
            skipped.append(f"{sample.name}: resolve MAYBE cells as YES, NO, or ARTIFACT before training")
            continue
        accepted = {v for v in present if decisions[v] == "YES"}
        if not accepted:
            skipped.append(f"{sample.name}: no accepted cells to train on")
            continue
        examples.append({"sample": sample.name, "raw": raw, "mask": mask,
                         "accepted_ids": sorted(accepted),
                         "rejected_count": len(present - accepted),
                         "manual_added_count": _accepted_manual_additions(
                             sample, channel, accepted)})
    return examples, skipped


def _accepted_manual_additions(sample: Path, channel: int,
                               accepted: set[int]) -> int:
    """Count reviewed YES cells that the editor recorded as manual additions."""
    path = sample / f"_manual_additions_ch{channel}.tsv"
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return len({int(row["label_id"]) for row in csv.DictReader(
                source, delimiter="\t") if int(row["label_id"]) in accepted})
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def _focus_training_pairs(pairs: list[tuple[str, str]], examples: list[dict],
                          focus: str) -> list[tuple[str, str]]:
    """Repeat difficult training samples only; validation samples stay untouched."""
    if focus == "none":
        return pairs
    count_key = ("manual_added_count" if focus == "missed_cells"
                 else "rejected_count")
    weighted = []
    for pair, example in zip(pairs, examples):
        weighted.extend([pair] * (2 if example.get(count_key, 0) > 0 else 1))
    return weighted


def _prepare_pair(example: dict, destination: Path, index: int,
                  preprocess_mode: str) -> tuple[str, str]:
    from .segmentation import preprocess_tile_for_sam

    raw = tifffile.imread(example["raw"])
    mask = tifffile.imread(example["mask"])
    if raw.ndim != 2 or mask.ndim != 2 or raw.shape != mask.shape:
        raise ValueError(f"{example['sample']}: raw image and mask must be matching 2D arrays")
    processed, _rgb = preprocess_tile_for_sam(raw, mode=preprocess_mode)
    curated = np.where(np.isin(mask, example["accepted_ids"]), mask, 0)
    image_path = destination / f"image_{index:04d}.tif"
    label_path = destination / f"labels_{index:04d}.tif"
    tifffile.imwrite(image_path, processed.astype(np.uint8))
    tifffile.imwrite(label_path, curated.astype(np.uint32))
    return str(image_path), str(label_path)


def _train_cellpose(train_pairs, val_pairs, output: Path, epochs: int,
                    base_model: str, advanced: dict | None = None) -> str:
    from cellpose import models, train
    import torch

    def arrays(pairs):
        return ([tifffile.imread(p[0]) for p in pairs],
                [tifffile.imread(p[1]) for p in pairs])

    x_train, y_train = arrays(train_pairs)
    x_val, y_val = arrays(val_pairs)
    use_gpu = (torch.cuda.is_available() or
               (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    model = models.CellposeModel(gpu=use_gpu,
                                pretrained_model=base_model or "cpsam")
    settings = advanced or {}
    model_path, _loss, _val_loss = train.train_seg(
        model.net, train_data=x_train, train_labels=y_train,
        test_data=x_val, test_labels=y_val, n_epochs=epochs,
        learning_rate=settings.get("learning_rate", 1e-5),
        batch_size=settings.get("batch_size", 1),
        bsize=settings.get("patch_size", 256), min_train_masks=1,
        save_path=str(output), model_name="cellmatchr_cellpose",
    )
    return str(model_path)


def _train_microsam(train_pairs, val_pairs, output: Path, epochs: int,
                    base_model: str, advanced: dict | None = None) -> str:
    from micro_sam.training import default_sam_loader, train_sam
    from micro_sam.util import export_custom_sam_model
    from torch_em.data.sampler import MinInstanceSampler

    settings = advanced or {}
    def loader(pairs, is_train):
        return default_sam_loader(
            raw_paths=[p[0] for p in pairs], raw_key=None,
            label_paths=[p[1] for p in pairs], label_key=None,
            patch_shape=(settings.get("patch_size", 256),) * 2,
            with_segmentation_decoder=True,
            batch_size=settings.get("batch_size", 1), num_workers=0,
            n_samples=max(4, 4 * len(pairs)),
            is_train=is_train, min_size=5,
            sampler=MinInstanceSampler(2, min_size=5),
            max_sampling_attempts=1000,
        )

    checkpoint = base_model if base_model and os.path.isfile(base_model) else None
    model_type = "vit_b" if checkpoint else (base_model or "vit_b_lm")
    train_sam(
        name="cellmatchr_microsam", model_type=model_type,
        train_loader=loader(train_pairs, True),
        val_loader=loader(val_pairs, False),
        n_epochs=epochs, n_sub_iteration=2, n_objects_per_batch=4,
        checkpoint_path=checkpoint, with_segmentation_decoder=True,
        freeze=["image_encoder"] if settings.get("freeze_encoder", True) else [],
        lr=settings.get("learning_rate", 1e-5), save_root=str(output),
        verify_n_labels_in_loader=4,
    )
    best = output / "checkpoints" / "cellmatchr_microsam" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Micro-SAM training finished without a best checkpoint")
    exported = output / "cellmatchr_microsam.pth"
    export_custom_sam_model(best, model_type="vit_b", save_path=exported,
                            with_segmentation_decoder=True)
    return str(exported)


def _train_stardist(train_pairs, val_pairs, output: Path, epochs: int,
                    base_model: str, advanced: dict | None = None) -> str:
    try:
        from stardist.models import StarDist2D
    except ImportError as exc:
        raise RuntimeError("StarDist is optional. Install the StarDist extra before training.") from exc

    if base_model and Path(base_model).is_dir():
        source = Path(base_model)
        pretrained = StarDist2D(None, name=source.name, basedir=str(source.parent))
    else:
        pretrained = StarDist2D.from_pretrained(base_model or "2D_versatile_fluo")
    settings = advanced or {}
    config = copy.deepcopy(pretrained.config)
    if "learning_rate" in settings:
        config.train_learning_rate = settings["learning_rate"]
    if "batch_size" in settings:
        config.train_batch_size = settings["batch_size"]
    if "patch_size" in settings:
        config.train_patch_size = (settings["patch_size"],) * 2
    if "loss" in settings:
        config.train_dist_loss = settings["loss"]
    trained = StarDist2D(config, name="cellmatchr_stardist",
                         basedir=str(output))
    trained.keras_model.set_weights(pretrained.keras_model.get_weights())
    x_train = [tifffile.imread(p[0]).astype(np.float32) / 255 for p in train_pairs]
    y_train = [tifffile.imread(p[1]) for p in train_pairs]
    x_val = [tifffile.imread(p[0]).astype(np.float32) / 255 for p in val_pairs]
    y_val = [tifffile.imread(p[1]) for p in val_pairs]
    trained.train(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=epochs, steps_per_epoch=max(1, len(x_train) * 4))
    return str(output / "cellmatchr_stardist")


TRAINERS = {"microsam": _train_microsam, "cellpose": _train_cellpose,
            "stardist": _train_stardist}


def train_model(project: str, seg_run: str, channel: int, backend: str,
                epochs: int, base_model: str = "", preprocess_mode: str = "local_bg_subtract",
                model_name: str = "", advanced: dict | None = None) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Choose one of: {', '.join(BACKENDS)}")
    if epochs < 1 or epochs > 500:
        raise ValueError("Epochs must be between 1 and 500")
    if channel < 0:
        raise ValueError("Channel must be 0 or higher")
    if preprocess_mode not in PREPROCESS_MODES:
        raise ValueError(f"Choose a preprocessing mode: {', '.join(PREPROCESS_MODES)}")
    model_name = model_name.strip()
    if len(model_name) > 60 or any(ch in model_name for ch in "/\\\n\r\t"):
        raise ValueError("Model name must be at most 60 characters with no slashes or line breaks")
    advanced = validate_advanced(backend, advanced)
    if backend == "stardist" and (importlib.util.find_spec("stardist") is None or
                                   importlib.util.find_spec("tensorflow") is None):
        raise RuntimeError("StarDist training needs the optional StarDist and TensorFlow packages")
    if base_model and (os.path.sep in base_model or base_model.endswith((".pt", ".pth"))):
        model = Path(base_model).expanduser()
        if backend == "stardist" and not model.is_dir():
            raise FileNotFoundError(f"Starting StarDist model folder not found: {model}")
        if backend != "stardist" and not model.is_file():
            raise FileNotFoundError(f"Starting model file not found: {model}")
        base_model = str(model.resolve())
    if os.path.basename(seg_run) != seg_run or not seg_run.startswith("Segmentation"):
        raise ValueError("Choose a segmentation run inside the project")
    project_path = Path(project).resolve()
    seg_folder = project_path / seg_run
    examples, skipped = collect_examples(str(seg_folder), channel)
    for reason in skipped:
        print(f"[SKIP] {reason}", flush=True)
    if len(examples) < 2:
        raise ValueError("At least two fully validated samples with YES cells are needed for training")
    print(f"Reviewed targets: {sum(len(e['accepted_ids']) for e in examples)} YES; "
          f"{sum(e['rejected_count'] for e in examples)} NO/ARTIFACT across "
          f"{len(examples)} samples", flush=True)

    parent = None
    custom_base = bool(base_model and os.path.isabs(base_model))
    if custom_base:
        starting_path = os.path.normcase(os.path.realpath(base_model))
        parent = next((model for model in list_trained_models(str(project_path), backend)
                       if os.path.normcase(os.path.realpath(model["model_path"])) == starting_path),
                      None)
    prior_rounds = int(parent.get("training_rounds") or 1) if parent else 0
    prior_epochs = int(parent.get("recorded_epochs") or parent.get("epochs") or 0) if parent else 0
    history_complete = (not custom_base if parent is None
                        else bool(parent.get("history_complete", parent.get("epochs") is not None)))

    output_root = model_library_root()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")[:32]
    folder_stem = f"{backend}_ch{channel}"
    if name_slug:
        folder_stem += f"_{name_slug}"
    output = output_root / f"{folder_stem}_{timestamp}"
    suffix = 2
    while output.exists():
        output = output_root / f"{folder_stem}_{timestamp}_{suffix}"
        suffix += 1
    output.mkdir()
    stage = Path(tempfile.mkdtemp(prefix=".training_data_", dir=output))
    try:
        pairs = [_prepare_pair(example, stage, i, preprocess_mode)
                 for i, example in enumerate(examples)]
        # Keep whole samples apart so validation does not reuse training pixels.
        n_val = max(1, len(pairs) // 5) if "validation_percent" not in advanced else max(
            1, round(len(pairs) * advanced["validation_percent"] / 100))
        train_pairs, val_pairs = pairs[:-n_val], pairs[-n_val:]
        focus = advanced.get("sampling_focus", "none")
        train_pairs = _focus_training_pairs(train_pairs, examples[:-n_val], focus)
        print(f"Training {backend}: {len(examples) - n_val} training samples "
              f"({len(train_pairs)} presentations; focus={focus}); "
              f"validation: {len(val_pairs)}; channel {channel}", flush=True)
        path = TRAINERS[backend](train_pairs, val_pairs, output, epochs, base_model, advanced)
        result = {"backend": backend, "model_path": path, "channel": channel,
                  "model_relative_path": str(Path(path).relative_to(output)),
                  "name": model_name or f"{backend} channel {channel} {timestamp}",
                  "source_project": str(project_path),
                  "starting_model_name": parent["name"] if parent else base_model,
                  "parent_model_path": parent["model_path"] if parent else "",
                  "training_rounds": prior_rounds + 1,
                  "recorded_epochs": prior_epochs + epochs,
                  "history_complete": history_complete,
                  "segmentation_run": seg_run, "preprocess": preprocess_mode,
                  "epochs": epochs, "advanced": advanced,
                  "training_samples": [e["sample"] for e in examples[:-n_val]],
                  "validation_samples": [e["sample"] for e in examples[-n_val:]],
                  "skipped": skipped, "base_model": base_model,
                  "completed_at": datetime.now().isoformat(timespec="seconds")}
        (output / "model.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[COMPLETE] Trained model: {path}", flush=True)
        return result
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a CellMatchR segmentation model")
    parser.add_argument("--project", required=True)
    parser.add_argument("--seg-run", required=True)
    parser.add_argument("--channel", type=int, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--preprocess", default="local_bg_subtract")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--validation-percent", type=int)
    parser.add_argument("--loss", choices=("mae", "mse"))
    parser.add_argument("--freeze-encoder", choices=("yes", "no"))
    parser.add_argument("--sampling-focus", choices=("none", "missed_cells", "rejected_cells"))
    args = parser.parse_args()
    advanced = {key: getattr(args, key) for key in
                ("learning_rate", "batch_size", "patch_size", "validation_percent",
                 "loss", "sampling_focus")
                if getattr(args, key) is not None}
    if args.freeze_encoder is not None:
        advanced["freeze_encoder"] = args.freeze_encoder == "yes"
    train_model(args.project, args.seg_run, args.channel, args.backend,
                args.epochs, args.base_model, args.preprocess, args.model_name, advanced)


if __name__ == "__main__":
    main()
