"""Portable training curves and before/after validation for saved models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .file_paths import read_tiff


def save_history(output: Path, values: dict) -> None:
    """Keep numeric histories independent of backend-specific log formats."""
    clean = {}
    for key, series in values.items():
        try:
            clean[str(key)] = [float(value) for value in series]
        except (TypeError, ValueError):
            continue
    (output / "learning_history.json").write_text(json.dumps(clean, indent=2) + "\n")


def read_history(output: Path, backend: str) -> dict:
    path = output / "learning_history.json"
    if path.is_file():
        return json.loads(path.read_text())
    if backend != "microsam":
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    history = {}
    for event_file in sorted(output.glob("logs/**/events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            history.setdefault(tag, []).extend(
                (int(point.step), float(point.value)) for point in accumulator.Scalars(tag))
    return {key: [value for _step, value in sorted(set(series))]
            for key, series in history.items()}


def learning_curve(history: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=145)
    colors = ("#4c78a8", "#e45756", "#54a24b", "#b279a2")
    shown = 0
    for key in ("train/loss", "validation/loss", "train_loss", "val_loss",
                "loss", "test_loss"):
        series = history.get(key)
        if not series:
            continue
        ax.plot(range(1, len(series) + 1), series, label=key.replace("/", " "),
                color=colors[shown % len(colors)], linewidth=1.7)
        shown += 1
    if not shown:
        ax.text(.5, .5, "No loss history was emitted by this backend",
                transform=ax.transAxes, ha="center", va="center")
    else:
        ax.legend(frameon=False)
    ax.set(xlabel="Training step / epoch", ylabel="Loss", title="Training history")
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def _predict_mask(raw: np.ndarray, backend: str, model_name: str,
                  preprocess: str) -> np.ndarray:
    """Run the same backend preprocessing and tiled predictor as segmentation."""
    from . import segmentation as seg
    import torch

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, "mps") and
              torch.backends.mps.is_available() else "cpu")
    predictor = segmenter = cellpose_model = stardist_model = None
    kwargs = {}
    if backend == "cellpose":
        from cellpose import models
        cellpose_model = models.CellposeModel(gpu=device != "cpu",
                                             pretrained_model=model_name or "cpsam")
    elif backend == "stardist":
        from stardist.models import StarDist2D
        location = Path(model_name) if model_name else None
        stardist_model = (StarDist2D(None, name=location.name,
                                    basedir=str(location.parent))
                          if location is not None and location.is_dir() else
                          StarDist2D.from_pretrained(model_name or "2D_versatile_fluo"))
    elif backend == "microsam":
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        from .retraining import microsam_architecture
        location = Path(model_name)
        custom = location.is_file()
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=(microsam_architecture(model_name) if custom else
                        model_name or "vit_b_lm"),
            device=device, checkpoint=str(location) if custom else None)
        kwargs["min_size"] = 0
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    override = {"PREPROCESS_MODE": preprocess, "MIN_OBJECT_SIZE": 0,
                "MAX_OBJECT_SIZE": 0, "MAX_MAJOR_MINOR_RATIO": 0,
                "RAW_MEAN_MIN": 0, "RAW_MEAN_MAX": 0,
                "SKIP_EMPTY_TILES": False, "TEST_RUN": False,
                "CELLPOSE_VARIANT": "cpsam", "CELLPOSE_DIAMETER": 0,
                "CELLPOSE_FLOW_THRESHOLD": .4,
                "CELLPOSE_CELLPROB_THRESHOLD": 0}
    previous = {key: getattr(seg, key) for key in override}
    try:
        for key, value in override.items():
            setattr(seg, key, value)
        tile = (min(1024, raw.shape[0]), min(1024, raw.shape[1]))
        halo = tuple(min(128, max(0, value // 4)) for value in tile)
        return seg.segment_by_tiling(
            predictor, segmenter, raw,
            tile, halo, 2, cellpose_model=cellpose_model,
            stardist_model=stardist_model, **kwargs)
    finally:
        for key, value in previous.items():
            setattr(seg, key, value)


def object_scores(predicted: np.ndarray, truth: np.ndarray,
                  threshold: float = .5) -> dict:
    """Greedy one-to-one object matching at an explicit IoU threshold."""
    if predicted.shape != truth.shape:
        raise ValueError("Validation mask dimensions differ")
    pred_ids, pred_count = np.unique(predicted[predicted > 0], return_counts=True)
    true_ids, true_count = np.unique(truth[truth > 0], return_counts=True)
    pred_area = dict(zip(pred_ids.tolist(), pred_count.tolist()))
    true_area = dict(zip(true_ids.tolist(), true_count.tolist()))
    both = (predicted > 0) & (truth > 0)
    pairs, intersections = np.unique(
        np.stack((predicted[both], truth[both]), axis=1), axis=0,
        return_counts=True) if np.any(both) else (np.empty((0, 2)), [])
    candidates = []
    for (pred_id, true_id), intersection in zip(pairs, intersections):
        iou = intersection / (pred_area[int(pred_id)] + true_area[int(true_id)] - intersection)
        if iou >= threshold:
            candidates.append((iou, int(pred_id), int(true_id)))
    used_pred, used_true = set(), set()
    for _iou, pred_id, true_id in sorted(candidates, reverse=True):
        if pred_id not in used_pred and true_id not in used_true:
            used_pred.add(pred_id)
            used_true.add(true_id)
    tp = len(used_pred)
    fp = len(pred_area) - tp
    fn = len(true_area) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positive": tp, "false_positive": fp, "missed_cell": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0, "iou_threshold": threshold}


def validation_overlay(raw: np.ndarray, mask: np.ndarray,
                       destination: Path) -> None:
    from PIL import Image
    from scipy.ndimage import binary_erosion

    stride = max(1, int(np.ceil(max(raw.shape) / 1100)))
    image = np.asarray(raw[::stride, ::stride], dtype=np.float32)
    lo, hi = np.percentile(image, (1, 99.7))
    gray = np.uint8(np.clip((image - lo) / max(hi - lo, 1), 0, 1) * 255)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    labels = mask[::stride, ::stride]
    edges = (labels > 0) & ~binary_erosion(labels > 0)
    rgb[edges] = (255, 222, 68)
    Image.fromarray(rgb).save(destination)


def build_report(project: Path, output: Path, result: dict,
                 validation_examples: list[dict], base_model: str,
                 preprocess: str) -> Path:
    """Write a project-local report and return its JSON path."""
    folder = project / "Training_Reports" / output.name
    folder.mkdir(parents=True, exist_ok=False)
    history = read_history(output, result["backend"])
    learning_curve(history, folder / "learning_curve.png")
    records = []
    for index, example in enumerate(validation_examples):
        raw = read_tiff(example["raw"])
        labels = read_tiff(example["mask"])
        if example.get("crop"):
            y0, y1, x0, x1 = example["crop"]
            raw, labels = raw[y0:y1, x0:x1], labels[y0:y1, x0:x1]
        truth = np.where(np.isin(labels, example["accepted_ids"]), labels, 0)
        original = _predict_mask(raw, result["backend"], base_model, preprocess)
        trained = _predict_mask(raw, result["backend"], result["model_path"], preprocess)
        before = object_scores(original, truth)
        after = object_scores(trained, truth)
        validation_overlay(raw, truth, folder / f"image_{index+1}_reviewed.png")
        validation_overlay(raw, original, folder / f"image_{index+1}_before.png")
        validation_overlay(raw, trained, folder / f"image_{index+1}_after.png")
        records.append({"project": example["project"], "sample": example["sample"],
                        "crop": example.get("crop"), "before": before, "after": after})
    report = {"model": result["name"], "backend": result["backend"],
              "model_path": result["model_path"], "validation_plan":
              result["validation_plan"], "history": history,
              "validation": records, "training_sources": result["training_sources"],
              "training_samples": result["training_samples"],
              "excluded_keys": result["excluded_keys"], "epochs": result["epochs"],
              "advanced": result["advanced"], "preprocess": preprocess}
    path = folder / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
