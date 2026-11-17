"""Prepare reviewed CellMatchR masks and fine-tune a selected model.

Training writes a new model folder. Source masks and project validation files
are read only. Heavy training libraries are imported only for their backend.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from .file_paths import filesystem_path, read_tiff
from .roi_editor import _find_mask, _find_raw, _prediction_baseline_path
from .validation_state import get_change_state, get_validation_tsv_path

BACKENDS = ("microsam", "cellpose", "stardist")
DEFAULT_PATCH_TARGET_INCORRECT = 8
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
        "description": "Repeat samples with ARTIFACT predictions. Check held-out recall so real cells are still found.",
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
        "description": "Continue a saved model gently while repeating samples with ARTIFACT predictions; check missed cells afterward.",
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


def _logged_ids(path: Path, field: str) -> set[int]:
    """Read optional editor provenance without rejecting older projects."""
    if not path.is_file():
        return set()
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return {int(row[field]) for row in csv.DictReader(source, delimiter="\t")}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def example_id(project: str, seg_run: str, channel: int, sample: str) -> str:
    """Stable identifier for a reviewed image across projects and platforms."""
    identity = json.dumps([os.path.normcase(os.path.realpath(project)),
                           seg_run, int(channel), sample], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def microsam_architecture(model_name: str) -> str:
    """Recover the SAM backbone for a built-in or previously saved model."""
    match = re.match(r"^(vit_[blh])(?:_|$)", str(model_name))
    if match:
        return match.group(1)
    model_file = Path(model_name).expanduser()
    if model_file.is_file():
        record_file = model_file.parent / "model.json"
        try:
            with record_file.open(encoding="utf-8") as source:
                record = json.load(source)
            if record.get("backend") == "microsam":
                value = (record.get("microsam_architecture") or
                         record.get("base_model") or
                         record.get("starting_model_name", ""))
                match = re.match(r"^(vit_[blh])(?:_|$)", str(value))
                if match:
                    return match.group(1)
        except (OSError, ValueError, TypeError):
            pass
    return "vit_b"  # Older custom checkpoints were always loaded as vit_b.


def project_runs(project_file: str) -> list[dict]:
    """List valid segmentation runs in a project file's actual folder."""
    path = Path(project_file).expanduser().resolve()
    if path.name != "cellmatchr_project.json" or not path.is_file():
        raise ValueError("Choose a project's cellmatchr_project.json file")
    with path.open(encoding="utf-8") as source:
        settings = json.load(source)
    if not isinstance(settings, dict):
        raise ValueError(f"Project settings are invalid: {path}")
    runs = []
    for folder in path.parent.iterdir():
        if folder.is_dir() and not folder.is_symlink() and folder.name.startswith("Segmentation"):
            channels = available_channels(str(folder))
            if channels:
                runs.append({"name": folder.name, "channels": channels})
    return sorted(runs, key=lambda item: item["name"])


def collect_source_examples(sources: list[dict], *, with_review_status: bool = False):
    """Read reviewed images from one or more projects, never altering sources."""
    examples, skipped, unfinished, seen = [], [], [], set()
    for source in sources:
        project = Path(source["project"]).expanduser().resolve()
        run = str(source["seg_run"])
        channel = int(source["channel"])
        if (not project.is_dir() or os.path.basename(run) != run or
                not run.startswith("Segmentation") or channel < 0):
            raise ValueError(f"Invalid training source: {project} / {run} / channel {channel}")
        folder = project / run
        if not folder.is_dir() or folder.is_symlink():
            raise FileNotFoundError(f"Segmentation run not found: {folder}")
        found, omitted, pending = collect_examples(
            str(folder), channel, with_review_status=True)
        title = str(source.get("name") or project.name)
        skipped.extend(f"{title} / {run} / ch{channel}: {reason}" for reason in omitted)
        unfinished.extend(f"{title} / {run} / ch{channel}: {reason}" for reason in pending)
        for item in found:
            key = example_id(str(project), run, channel, item["sample"])
            if key in seen:
                continue
            seen.add(key)
            item.update({"key": key, "project": str(project),
                         "project_name": title, "seg_run": run, "channel": channel})
            examples.append(item)
    return (examples, skipped, unfinished) if with_review_status else (examples, skipped)


def collect_examples(seg_folder: str, channel: int, *, with_review_status: bool = False):
    """Find fully reviewed pairs; only ARTIFACT is a false prediction."""
    root = Path(seg_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Segmentation run not found: {root}")
    examples, skipped, unfinished = [], [], []
    for sample in sorted(root.iterdir()):
        if not sample.is_dir() or sample.name.startswith((".", "DEBUG")):
            continue
        raw, mask = _find_raw(str(sample), channel), _find_mask(str(sample), channel)
        if not raw or not mask:
            skipped.append(f"{sample.name}: raw image or channel mask missing")
            continue
        state = get_change_state(str(sample), channel, mask)
        if state["status"] in ("pending", "unknown") or not _review_complete(sample, channel):
            reason = f"{sample.name}: finish channel {channel} validation first"
            skipped.append(reason)
            unfinished.append(reason)
            continue
        review = get_validation_tsv_path(str(sample), channel)
        if not review:
            reason = f"{sample.name}: validation decisions missing"
            skipped.append(reason)
            unfinished.append(reason)
            continue
        decisions = _decisions(review)
        labels = read_tiff(mask)
        present = {int(v) for v in np.unique(labels) if v > 0}
        if present - decisions.keys():
            reason = f"{sample.name}: some current cells have no decision"
            skipped.append(reason)
            unfinished.append(reason)
            continue
        if any(not decisions[v] for v in present):
            reason = f"{sample.name}: some current cells have an empty label"
            skipped.append(reason)
            unfinished.append(reason)
            continue
        if any(decisions[v] == "MAYBE" for v in present):
            reason = f"{sample.name}: resolve MAYBE cells before training"
            skipped.append(reason)
            unfinished.append(reason)
            continue
        accepted = {v for v in present if decisions[v] != "ARTIFACT"}
        artifacts = present - accepted
        if not accepted:
            skipped.append(f"{sample.name}: no labeled cells to train on")
            continue
        additions = _logged_ids(sample / f"_manual_additions_ch{channel}.tsv", "label_id")
        deletions = _logged_ids(sample / f"_manual_deletions_ch{channel}.tsv", "label_id")
        erased_ids = deletions - present
        baseline_path = _prediction_baseline_path(mask, channel)
        if os.path.isfile(baseline_path):
            baseline = read_tiff(baseline_path)
            if baseline.shape != labels.shape:
                raise ValueError(f"{sample.name}: prediction baseline shape differs from edited mask")
            original_count = len({int(value) for value in np.unique(baseline)
                                  if value > 0})
        else:
            # Older edited projects may have no pre-edit mask to consult.
            original_count = len(present - additions) + len(erased_ids)
        accepted_additions = additions & accepted
        label_ids = {word: sorted(v for v in accepted if decisions[v] == word)
                     for word in sorted({decisions[v] for v in accepted})}
        examples.append({"sample": sample.name, "raw": raw, "mask": mask,
                         "accepted_ids": sorted(accepted),
                         "label_ids": label_ids,
                         "manual_added_ids": sorted(accepted_additions),
                         "erased_prediction_ids": sorted(erased_ids),
                         "rejected_ids": sorted(artifacts),
                         "rejected_count": len(artifacts),
                         "manual_added_count": len(accepted_additions),
                         "original_count": original_count,
                         "incorrect_count": len(artifacts) + len(erased_ids),
                         "final_count": len(accepted),
                         "deletion_count": len(erased_ids),
                         "prediction_baseline": baseline_path if os.path.isfile(baseline_path) else ""})
    return (examples, skipped, unfinished) if with_review_status else (examples, skipped)


def select_training_labels(examples: list[dict], labels: list[str] | None
                           ) -> tuple[list[dict], list[str]]:
    """Choose cell targets without changing source masks or artifact decisions."""
    chosen = {str(word).strip().upper() for word in (labels or [])}
    if "" in chosen or chosen & {"ARTIFACT", "MAYBE"}:
        raise ValueError("Choose cell labels other than ARTIFACT or MAYBE")
    if not chosen:
        return examples, []
    filtered, skipped = [], []
    for example in examples:
        targets = sorted({int(value) for word, ids in example["label_ids"].items()
                          if word in chosen for value in ids})
        if not targets:
            skipped.append(f"{example['sample']}: no cells with selected labels")
            continue
        selected_additions = sorted(set(example.get("manual_added_ids", [])) &
                                    set(targets))
        item = dict(example, accepted_ids=targets, final_count=len(targets),
                    manual_added_ids=selected_additions,
                    manual_added_count=len(selected_additions))
        filtered.append(item)
    return filtered, skipped


def _accepted_manual_additions(sample: Path, channel: int,
                               accepted: set[int]) -> int:
    """Count target cells that the editor recorded as manual additions."""
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
    weighted = []
    for pair, example in zip(pairs, examples):
        count = (example.get("manual_added_count", 0) if focus == "missed_cells"
                 else example.get("incorrect_count", example.get("rejected_count", 0)))
        weighted.extend([pair] * (2 if count > 0 else 1))
    return weighted


def _patch_validation_split(example: dict, target_incorrect: int,
                            rng: np.random.Generator
                            ) -> tuple[list[dict], dict, dict]:
    """Reserve a spatial patch; training strips never contain its pixels."""
    labels = read_tiff(example["mask"])
    if labels.ndim != 2:
        raise ValueError(f"{example['sample']}: patch validation needs a 2D mask")
    rejected = [int(value) for value in example.get("rejected_ids", [])
                if np.any(labels == value)]
    if not rejected:
        raise ValueError("Patch validation needs at least one reviewed ARTIFACT "
                         "object in the selected image")
    centers = {}
    for value in rejected:
        y, x = np.where(labels == value)
        centers[value] = (float(y.mean()), float(x.mean()))
    anchor = rejected[int(rng.integers(len(rejected)))]
    ay, ax = centers[anchor]
    chosen = sorted(rejected, key=lambda value:
                    (centers[value][0] - ay) ** 2 + (centers[value][1] - ax) ** 2
                    )[:max(1, target_incorrect)]
    accepted_centers = {}
    for value in example["accepted_ids"]:
        y, x = np.where(labels == value)
        if y.size:
            accepted_centers[int(value)] = (float(y.mean()), float(x.mean()))
    if not accepted_centers:
        raise ValueError("The patch candidate has no accepted cells")
    closest_yes = min(accepted_centers, key=lambda value:
                      (accepted_centers[value][0] - ay) ** 2 +
                      (accepted_centers[value][1] - ax) ** 2)
    selected = np.isin(labels, chosen + [closest_yes])
    ys, xs = np.where(selected)
    height, width = labels.shape
    patch_h = min(height, max(256, int(ys.max() - ys.min() + 65)))
    patch_w = min(width, max(256, int(xs.max() - xs.min() + 65)))
    y0 = max(0, min(height - patch_h, int((ys.min() + ys.max()) / 2 - patch_h / 2)))
    x0 = max(0, min(width - patch_w, int((xs.min() + xs.max()) / 2 - patch_w / 2)))
    y1, x1 = y0 + patch_h, x0 + patch_w
    heldout = dict(example, crop=(y0, y1, x0, x1))
    heldout_ids = set(np.unique(labels[y0:y1, x0:x1]).tolist()) - {0}
    strips = ((0, y0, 0, width), (y1, height, 0, width),
              (y0, y1, 0, x0), (y0, y1, x1, width))
    training = []
    for box in strips:
        top, bottom, left, right = box
        if bottom - top < 64 or right - left < 64:
            continue
        cropped = labels[top:bottom, left:right]
        trainable_ids = [value for value in example["accepted_ids"]
                         if value not in heldout_ids and np.any(cropped == value)]
        if not trainable_ids:
            continue
        training.append(dict(example, crop=box, accepted_ids=trainable_ids,
                             sample=f"{example['sample']} [outside holdout {len(training)+1}]"))
    contained = set(np.unique(labels[y0:y1, x0:x1]).tolist()) & set(rejected)
    return training, heldout, {"crop": [y0, y1, x0, x1],
                               "incorrect_objects_in_patch": len(contained)}


def plan_examples(examples: list[dict], validation_mode: str,
                  validation_key: str = "",
                  patch_target_incorrect: int = DEFAULT_PATCH_TARGET_INCORRECT,
                  validation_percent: int = 20
                  ) -> tuple[list[dict], list[dict], dict]:
    """Create an explicit split, including the disclosed in-sample option."""
    if not examples:
        raise ValueError("No reviewed images are selected for training")
    if validation_mode == "automatic":
        if len(examples) < 2:
            raise ValueError("At least two reviewed images are needed")
        count = max(1, round(len(examples) * validation_percent / 100))
        return examples[:-count], examples[-count:], {"mode": "automatic"}
    if validation_mode == "whole_image":
        if len(examples) < 2:
            raise ValueError("Whole-image holdout needs at least two selected images")
        matches = [item for item in examples if item["key"] == validation_key]
        if len(matches) != 1:
            raise ValueError("Click one image in the library to reserve it for validation")
        return [item for item in examples if item["key"] != validation_key], matches, {
            "mode": "whole_image", "heldout_key": validation_key}
    if validation_mode == "training_image":
        # Deliberately reuse a training image. This is not independent validation.
        choice = next((item for item in examples if item["key"] == validation_key), examples[0])
        return examples, [choice], {"mode": "training_image", "heldout_key": choice["key"],
                                     "independent": False}
    if validation_mode == "patch":
        if not 1 <= patch_target_incorrect <= 50:
            raise ValueError("Incorrect predictions per patch must be 1–50")
        candidates = ([item for item in examples if item["key"] == validation_key]
                      if validation_key else examples)
        candidates = [item for item in candidates if item.get("rejected_ids")]
        if not candidates:
            raise ValueError("Patch validation needs an image with reviewed ARTIFACT objects")
        random_seed = secrets.randbits(64)
        rng = np.random.default_rng(random_seed)
        choice = candidates[int(rng.integers(len(candidates)))]
        strips, heldout, details = _patch_validation_split(
            choice, patch_target_incorrect, rng)
        training = [item for item in examples if item["key"] != choice["key"]] + strips
        if not training:
            raise ValueError("The validation patch leaves no trainable cells outside it; "
                             "choose another image or use the all-data option")
        return training, [heldout], {"mode": "patch", "heldout_key": choice["key"],
                                      "independent": True, "random_seed": random_seed,
                                      **details}
    raise ValueError("Choose a validation approach before starting training")


def _prepare_pair(example: dict, destination: Path, index: int,
                  preprocess_mode: str) -> tuple[str, str]:
    from .segmentation import preprocess_tile_for_sam

    raw = read_tiff(example["raw"])
    mask = read_tiff(example["mask"])
    if raw.ndim != 2 or mask.ndim != 2 or raw.shape != mask.shape:
        raise ValueError(f"{example['sample']}: raw image and mask must be matching 2D arrays")
    if example.get("crop"):
        y0, y1, x0, x1 = example["crop"]
        raw, mask = raw[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        if not raw.size:
            raise ValueError(f"{example['sample']}: empty training crop")
    processed, _rgb = preprocess_tile_for_sam(raw, mode=preprocess_mode)
    curated = np.where(np.isin(mask, example["accepted_ids"]), mask, 0)
    image_path = destination / f"image_{index:04d}.tif"
    label_path = destination / f"labels_{index:04d}.tif"
    tifffile.imwrite(filesystem_path(image_path), processed.astype(np.uint8))
    tifffile.imwrite(filesystem_path(label_path), curated.astype(np.uint32))
    return str(image_path), str(label_path)


def _remove_staging_data(stage: Path) -> None:
    """Release dataset readers before deleting training TIFFs on Windows."""
    for attempt in range(8):
        gc.collect()
        try:
            shutil.rmtree(stage)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if attempt == 7:
                print(f"[WARNING] Could not remove temporary training data {stage}: {exc}",
                      flush=True)
            else:
                time.sleep(0.25)


def _train_cellpose(train_pairs, val_pairs, output: Path, epochs: int,
                    base_model: str, advanced: dict | None = None) -> str:
    from cellpose import models, train
    import torch

    def arrays(pairs):
        return ([read_tiff(p[0]) for p in pairs], [read_tiff(p[1]) for p in pairs])

    x_train, y_train = arrays(train_pairs)
    x_val, y_val = arrays(val_pairs)
    use_gpu = (torch.cuda.is_available() or
               (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    model = models.CellposeModel(gpu=use_gpu,
                                pretrained_model=base_model or "cpsam")
    settings = advanced or {}
    model_path, loss, val_loss = train.train_seg(
        model.net, train_data=x_train, train_labels=y_train,
        test_data=x_val, test_labels=y_val, n_epochs=epochs,
        learning_rate=settings.get("learning_rate", 1e-5),
        batch_size=settings.get("batch_size", 1),
        bsize=settings.get("patch_size", 256), min_train_masks=1,
        save_path=str(output), model_name="cellmatchr_cellpose",
    )
    from .training_report import save_history
    save_history(output, {"train_loss": loss, "val_loss": val_loss})
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
    model_type = (microsam_architecture(base_model) if checkpoint else
                  (base_model or "vit_b_lm"))
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
    export_custom_sam_model(best, model_type=microsam_architecture(model_type),
                            save_path=exported,
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
    x_train = [read_tiff(p[0]).astype(np.float32) / 255 for p in train_pairs]
    y_train = [read_tiff(p[1]) for p in train_pairs]
    x_val = [read_tiff(p[0]).astype(np.float32) / 255 for p in val_pairs]
    y_val = [read_tiff(p[1]) for p in val_pairs]
    history = trained.train(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=epochs, steps_per_epoch=max(1, len(x_train) * 4))
    if isinstance(getattr(history, "history", None), dict):
        from .training_report import save_history
        save_history(output, history.history)
    return str(output / "cellmatchr_stardist")


TRAINERS = {"microsam": _train_microsam, "cellpose": _train_cellpose,
            "stardist": _train_stardist}


def train_model(project: str, seg_run: str, channel: int, backend: str,
                epochs: int, base_model: str = "", preprocess_mode: str = "local_bg_subtract",
                model_name: str = "", advanced: dict | None = None,
                sources: list[dict] | None = None,
                excluded_keys: set[str] | None = None,
                validation_mode: str = "automatic", validation_key: str = "",
                patch_target_incorrect: int = DEFAULT_PATCH_TARGET_INCORRECT,
                generate_report: bool = False,
                allow_completed_only: bool = False,
                selected_labels: list[str] | None = None) -> dict:
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
    if sources is None:
        examples, skipped, unfinished = collect_examples(
            str(seg_folder), channel, with_review_status=True)
        for item in examples:
            item.update({"key": example_id(str(project_path), seg_run, channel,
                                            item["sample"]),
                         "project": str(project_path),
                         "project_name": project_path.name,
                         "seg_run": seg_run, "channel": channel})
    else:
        examples, skipped, unfinished = collect_source_examples(
            sources, with_review_status=True)
    if unfinished and not allow_completed_only:
        raise ValueError(f"{len(unfinished)} image(s) still need verification or "
                         "a resolved cell label. Confirm training on completed images only.")
    for reason in skipped:
        print(f"[SKIP] {reason}", flush=True)
    excluded = set(excluded_keys or ())
    examples = [item for item in examples if item["key"] not in excluded]
    examples, label_skips = select_training_labels(examples, selected_labels)
    skipped.extend(label_skips)
    for reason in label_skips:
        print(f"[SKIP] {reason}", flush=True)
    if not examples:
        raise ValueError("No verified images contain training targets for the chosen labels")
    train_examples, val_examples, split = plan_examples(
        examples, validation_mode, validation_key, patch_target_incorrect,
        advanced.get("validation_percent", 20))
    print(f"Reviewed targets: {sum(len(e['accepted_ids']) for e in examples)} labeled cells; "
          f"{sum(e['incorrect_count'] for e in examples)} incorrect predictions "
          "(ARTIFACT or erased) across "
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
        train_pairs = [_prepare_pair(example, stage, i, preprocess_mode)
                       for i, example in enumerate(train_examples)]
        val_pairs = [_prepare_pair(example, stage, i + len(train_examples), preprocess_mode)
                     for i, example in enumerate(val_examples)]
        focus = advanced.get("sampling_focus", "none")
        train_pairs = _focus_training_pairs(train_pairs, train_examples, focus)
        print(f"Training {backend}: {len(train_examples)} training samples "
              f"({len(train_pairs)} presentations; focus={focus}); "
              f"validation: {len(val_pairs)} ({split['mode']}); channel {channel}",
              flush=True)
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
                  "training_samples": [e["sample"] for e in train_examples],
                  "validation_samples": [e["sample"] for e in val_examples],
                  "training_images": [{"project": e["project"],
                                       "seg_run": e["seg_run"],
                                       "channel": e["channel"],
                                       "sample": e["sample"],
                                       "crop": e.get("crop"),
                                       "prediction_baseline": e.get("prediction_baseline", ""),
                                       "original_proposals": e["original_count"],
                                       "manually_added_ids": e["manual_added_ids"],
                                       "erased_prediction_ids": e["erased_prediction_ids"],
                                       "incorrect_predictions": e["incorrect_count"]}
                                      for e in train_examples],
                  "training_sources": sources or [{"project": str(project_path),
                                                   "seg_run": seg_run, "channel": channel}],
                  "selected_cell_labels": sorted({str(word).strip().upper()
                                                  for word in (selected_labels or [])}),
                  "excluded_keys": sorted(excluded), "validation_plan": split,
                  "skipped": skipped, "unfinished_review": unfinished,
                  "allow_completed_only": allow_completed_only,
                  "base_model": base_model,
                  "microsam_architecture": (microsam_architecture(base_model)
                                            if backend == "microsam" else None),
                  "completed_at": datetime.now().isoformat(timespec="seconds")}
        if generate_report:
            try:
                from .training_report import build_report
                report_path = build_report(project_path, output, result,
                                           val_examples, base_model, preprocess_mode)
                result["report_path"] = str(report_path)
                print(f"[REPORT] {report_path}", flush=True)
            except Exception as exc:
                report_folder = project_path / "Training_Reports" / output.name
                report_folder.mkdir(parents=True, exist_ok=True)
                report_path = report_folder / "report.json"
                result["report_path"] = str(report_path)
                result["report_error"] = f"{type(exc).__name__}: {exc}"
                report_path.write_text(json.dumps({
                    "status": "report_error", "model_path": path,
                    "message": result["report_error"], "validation_plan": split,
                    "training_sources": result["training_sources"],
                    "selected_cell_labels": result["selected_cell_labels"],
                    "unfinished_review": unfinished,
                    "allow_completed_only": allow_completed_only}, indent=2) + "\n")
                print(f"[REPORT ERROR] {result['report_error']}; {report_path}", flush=True)
        (output / "model.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[COMPLETE] Trained model: {path}", flush=True)
        return result
    finally:
        _remove_staging_data(stage)


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
    parser.add_argument("--sources-json", help="JSON list of project/run/channel sources")
    parser.add_argument("--exclude-key", action="append", default=[])
    parser.add_argument("--allow-completed-only", action="store_true",
                        help="Train only on fully verified images when some are unfinished")
    parser.add_argument("--cell-label", action="append", default=[],
                        help="Train selected cell labels; repeat for multiple labels")
    parser.add_argument("--validation-mode", choices=("automatic", "whole_image",
                                                       "patch", "training_image"),
                        default="automatic")
    parser.add_argument("--validation-key", default="")
    parser.add_argument("--patch-target-incorrect", type=int,
                        default=DEFAULT_PATCH_TARGET_INCORRECT)
    args = parser.parse_args()
    advanced = {key: getattr(args, key) for key in
                ("learning_rate", "batch_size", "patch_size", "validation_percent",
                 "loss", "sampling_focus")
                if getattr(args, key) is not None}
    if args.freeze_encoder is not None:
        advanced["freeze_encoder"] = args.freeze_encoder == "yes"
    sources = json.loads(args.sources_json) if args.sources_json else None
    if sources is not None and not isinstance(sources, list):
        parser.error("--sources-json must contain a list")
    train_model(args.project, args.seg_run, args.channel, args.backend,
                args.epochs, args.base_model, args.preprocess, args.model_name, advanced,
                sources, set(args.exclude_key), args.validation_mode,
                args.validation_key, args.patch_target_incorrect,
                generate_report=True,
                allow_completed_only=args.allow_completed_only,
                selected_labels=args.cell_label)


if __name__ == "__main__":
    main()
