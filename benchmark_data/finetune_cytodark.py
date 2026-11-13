"""Small Cellpose-SAM Nissl experiment with source-image-held-out evaluation.

The CytoDArk0 image/mask pairs are local and excluded from Git. Run `train`
before `evaluate`; never use the test split to select training settings.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SAMPLES = Path(__file__).resolve().parent / "CytoDArk0_subset"
SPLITS = {"train": {1, 54, 55, 66}, "validation": {28}, "test": {58, 62}}


def pairs(split: str):
    for image_path in sorted(SAMPLES.glob("ID*.png")):
        match = re.match(r"ID(\d+)_", image_path.name)
        if match and int(match.group(1)) in SPLITS[split]:
            mask_path = image_path.with_suffix(".tiff")
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            yield image_path, mask_path


def normalized_image(path: Path) -> np.ndarray:
    """Use the same grayscale and percentile conversion as the baseline test."""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    low, high = np.percentile(gray, [0.5, 99.8])
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return ((np.clip(gray, low, high) - low) / (high - low) * 255).astype(np.uint8)


def stage_pairs(split: str, folder: Path) -> list[tuple[str, str]]:
    folder.mkdir(parents=True, exist_ok=True)
    staged = []
    for image_path, mask_path in pairs(split):
        image_out = folder / f"{image_path.stem}_image.tif"
        mask_out = folder / f"{image_path.stem}_mask.tif"
        tifffile.imwrite(image_out, normalized_image(image_path))
        tifffile.imwrite(mask_out, tifffile.imread(mask_path))
        staged.append((str(image_out), str(mask_out)))
    if not staged:
        raise RuntimeError(f"No images in {split} split")
    return staged


def score(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """One-to-one greedy cell matching at IoU >= 0.5, as in the baseline."""
    true_ids, true_flat = np.unique(truth, return_inverse=True)
    pred_ids, pred_flat = np.unique(prediction, return_inverse=True)
    n_true, n_pred = len(true_ids), len(pred_ids)
    overlap = np.bincount(true_flat * n_pred + pred_flat,
                          minlength=n_true * n_pred).reshape(n_true, n_pred)
    true_area = overlap.sum(axis=1)
    pred_area = overlap.sum(axis=0)
    true_select, pred_select = true_ids != 0, pred_ids != 0
    intersection = overlap[np.ix_(true_select, pred_select)]
    union = true_area[true_select, None] + pred_area[None, pred_select] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                    where=union > 0)
    candidates = sorted(np.argwhere(iou >= 0.5),
                        key=lambda pair: iou[tuple(pair)], reverse=True)
    used_true, used_pred = set(), set()
    for true_idx, pred_idx in candidates:
        if int(true_idx) not in used_true and int(pred_idx) not in used_pred:
            used_true.add(int(true_idx))
            used_pred.add(int(pred_idx))
    tp = len(used_true)
    return {"gt": int(true_select.sum()), "pred": int(pred_select.sum()),
            "tp": tp, "fp": int(pred_select.sum()) - tp,
            "fn": int(true_select.sum()) - tp}


def summarize(rows: list[dict]) -> dict:
    total = {key: sum(row[key] for row in rows)
             for key in ("gt", "pred", "tp", "fp", "fn")}
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    total["precision"] = tp / (tp + fp) if tp + fp else 0.0
    total["recall"] = tp / (tp + fn) if tp + fn else 0.0
    total["f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return total


def baseline(split: str) -> dict:
    recorded = json.loads((SAMPLES / "scores.json").read_text())["cellpose"]["samples"]
    names = {image.stem for image, _ in pairs(split)}
    return summarize([row for row in recorded if row["sample"] in names])


def train(output: Path, epochs: int, starting_model: str) -> Path:
    import torch
    from cellmatchr.retraining import _train_cellpose

    np.random.seed(24)
    torch.manual_seed(24)
    train_pairs = stage_pairs("train", output / "staged" / "train")
    val_pairs = stage_pairs("validation", output / "staged" / "validation")
    print(f"Training on {len(train_pairs)} patches from source IDs {sorted(SPLITS['train'])}; "
          f"validating on {len(val_pairs)} patches from ID 28", flush=True)
    print(f"Using GPU: {torch.cuda.is_available() or torch.backends.mps.is_available()}",
          flush=True)
    (output / "trained").mkdir(parents=True, exist_ok=True)
    path = Path(_train_cellpose(train_pairs, val_pairs, output / "trained",
                               epochs, starting_model,
                               {"learning_rate": 1e-5, "batch_size": 1,
                                "patch_size": 256}))
    record = {"checkpoint": str(path), "starting_model": starting_model,
              "epochs": epochs, "learning_rate": 1e-5,
              "train_source_ids": sorted(SPLITS["train"]),
              "validation_source_ids": sorted(SPLITS["validation"]),
              "held_out_test_source_ids": sorted(SPLITS["test"])}
    (output / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved checkpoint: {path}", flush=True)
    return path


def evaluate(checkpoint: Path, split: str, output: Path) -> dict:
    import torch
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
                          pretrained_model=str(checkpoint))
    rows = []
    for image_path, mask_path in pairs(split):
        image = normalized_image(image_path)
        prediction = np.asarray(model.eval(image, diameter=None, flow_threshold=0.4,
                                           cellprob_threshold=0.0)[0])
        row = score(tifffile.imread(mask_path), prediction)
        row["sample"] = image_path.stem
        rows.append(row)
        print(f"{image_path.stem}: {row['tp']} matched / {row['gt']} labeled",
              flush=True)
    result = {"split": split, "checkpoint": str(checkpoint),
              "baseline": baseline(split), "fine_tuned": summarize(rows),
              "samples": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{split} F1: baseline {result['baseline']['f1']:.4f}; "
          f"fine-tuned {result['fine_tuned']['f1']:.4f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--starting-model", default="cpsam")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.action == "train":
        train(args.output, args.epochs, args.starting_model)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluate")
        evaluate(args.checkpoint, args.split, args.output)


if __name__ == "__main__":
    main()
