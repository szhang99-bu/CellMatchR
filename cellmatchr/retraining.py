"""Prepare reviewed CellMatchR masks and fine-tune a selected model.

Training writes a new model folder. Source masks and project validation files
are read only. Heavy training libraries are imported only for their backend.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
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
                    "raw_percentile", "tile_clahe_mild", "tile_clahe")


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
        accepted = {v for v in present if decisions[v] == "YES"}
        if not accepted:
            skipped.append(f"{sample.name}: no accepted cells to train on")
            continue
        examples.append({"sample": sample.name, "raw": raw, "mask": mask,
                         "accepted_ids": sorted(accepted),
                         "rejected_count": len(present - accepted)})
    return examples, skipped


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
                    base_model: str) -> str:
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
    model_path, _loss, _val_loss = train.train_seg(
        model.net, train_data=x_train, train_labels=y_train,
        test_data=x_val, test_labels=y_val, n_epochs=epochs,
        learning_rate=1e-5, batch_size=1, min_train_masks=1,
        save_path=str(output), model_name="cellmatchr_cellpose",
    )
    return str(model_path)


def _train_microsam(train_pairs, val_pairs, output: Path, epochs: int,
                    base_model: str) -> str:
    from micro_sam.training import default_sam_loader, train_sam
    from micro_sam.util import export_custom_sam_model
    from torch_em.data.sampler import MinInstanceSampler

    def loader(pairs, is_train):
        return default_sam_loader(
            raw_paths=[p[0] for p in pairs], raw_key=None,
            label_paths=[p[1] for p in pairs], label_key=None,
            patch_shape=(256, 256), with_segmentation_decoder=True,
            batch_size=1, num_workers=0, n_samples=max(4, 4 * len(pairs)),
            is_train=is_train, min_size=5,
            sampler=MinInstanceSampler(1, min_size=5, exclude_ids=[0]),
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
        freeze=["image_encoder"], save_root=str(output),
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
                    base_model: str) -> str:
    try:
        from stardist.models import StarDist2D
    except ImportError as exc:
        raise RuntimeError("StarDist is optional. Install the StarDist extra before training.") from exc

    if base_model and Path(base_model).is_dir():
        source = Path(base_model)
        pretrained = StarDist2D(None, name=source.name, basedir=str(source.parent))
    else:
        pretrained = StarDist2D.from_pretrained(base_model or "2D_versatile_fluo")
    trained = StarDist2D(pretrained.config, name="cellmatchr_stardist",
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
                epochs: int, base_model: str = "", preprocess_mode: str = "local_bg_subtract") -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Choose one of: {', '.join(BACKENDS)}")
    if epochs < 1 or epochs > 500:
        raise ValueError("Epochs must be between 1 and 500")
    if channel < 0:
        raise ValueError("Channel must be 0 or higher")
    if preprocess_mode not in PREPROCESS_MODES:
        raise ValueError(f"Choose a preprocessing mode: {', '.join(PREPROCESS_MODES)}")
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

    output_root = project_path / "Trained_Models"
    output_root.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output_root / f"{backend}_ch{channel}_{timestamp}"
    suffix = 2
    while output.exists():
        output = output_root / f"{backend}_ch{channel}_{timestamp}_{suffix}"
        suffix += 1
    output.mkdir()
    stage = Path(tempfile.mkdtemp(prefix=".training_data_", dir=output))
    try:
        pairs = [_prepare_pair(example, stage, i, preprocess_mode)
                 for i, example in enumerate(examples)]
        # Keep whole samples apart so validation does not reuse training pixels.
        n_val = max(1, len(pairs) // 5)
        train_pairs, val_pairs = pairs[:-n_val], pairs[-n_val:]
        print(f"Training {backend}: {len(train_pairs)} samples; "
              f"validation: {len(val_pairs)}; channel {channel}", flush=True)
        path = TRAINERS[backend](train_pairs, val_pairs, output, epochs, base_model)
        result = {"backend": backend, "model_path": path, "channel": channel,
                  "segmentation_run": seg_run, "preprocess": preprocess_mode,
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
    args = parser.parse_args()
    train_model(args.project, args.seg_run, args.channel, args.backend,
                args.epochs, args.base_model, args.preprocess)


if __name__ == "__main__":
    main()
