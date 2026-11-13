"""Gently fine-tune Micro-SAM light microscopy on inverted Nissl crops.

All four crops from one source remain in the same split. ID68 is excluded from
training and validation. Original vit_b_lm weights are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from finetune_cytodark import score, summarize
from render_macaque_comparison import matched_ids
from train_full_cytodark import crop_source
from train_microsam_nissl import partition

OUTPUT = Path.home() / "CellMatchR/Experiments/Nissl_MicroSAM_LM_20261113"
TRAINING_NAME = "nissl_microsam_lm_gentle"
LR = 3e-6
EPOCHS = 6


def inverted_rgb(gray: np.ndarray) -> np.ndarray:
    """Use the same per-crop input as the original LM model comparison."""
    return np.repeat((255 - gray)[:, :, None], 3, axis=2)


def stage_pairs(items: list[dict], folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    pairs = []
    for item in items:
        for name, gray, labels in crop_source(item):
            raw_path = folder / f"{name}_image.png"
            label_path = folder / f"{name}_labels.tif"
            Image.fromarray(inverted_rgb(gray)).save(raw_path)
            tifffile.imwrite(label_path, labels.astype(np.uint32))
            pairs.append((str(raw_path), str(label_path)))
    return pairs


def make_loader(pairs, *, training: bool):
    from micro_sam.training import default_sam_loader
    from torch_em.data.sampler import MinInstanceSampler

    return default_sam_loader(
        raw_paths=[raw for raw, _ in pairs], raw_key=None,
        label_paths=[label for _, label in pairs], label_key=None,
        patch_shape=(256, 256), with_segmentation_decoder=True,
        batch_size=1, num_workers=2 if training else 0,
        n_samples=len(pairs) if training else max(8, len(pairs)),
        is_train=training, min_size=5,
        sampler=MinInstanceSampler(2, min_size=5),
        max_sampling_attempts=1000,
    )


def train(output: Path):
    import torch
    from micro_sam.training import train_sam
    from micro_sam.util import export_custom_sam_model

    if not torch.cuda.is_available():
        raise RuntimeError("Run this experiment on the lab GPU")
    splits = partition()
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "starting_model": "vit_b_lm", "preprocess": "brightfield_invert",
        "input": "per-crop 0.5–99.8 percentile grayscale, inverted, repeated RGB",
        "training_sources": [item["stem"] for item in splits["train"]],
        "validation_sources": [item["stem"] for item in splits["validation"]],
        "test_sources": [item["stem"] for item in splits["test"]],
        "epochs": EPOCHS, "learning_rate": LR, "batch_size": 1,
        "patch_size": 256, "freeze_encoder": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output / "split.json").write_text(json.dumps(record, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="staged_lm_", dir=output) as temp:
        root = Path(temp)
        training = stage_pairs(splits["train"], root / "train")
        validation = stage_pairs(splits["validation"], root / "validation")
        print(f"LM training: {len(training)} crops from {len(splits['train'])} "
              f"sources; validation {len(validation)} crops from "
              f"{len(splits['validation'])} sources; test source ID68 excluded",
              flush=True)
        train_loader = make_loader(training, training=True)
        val_loader = make_loader(validation, training=False)
        sample_raw, sample_labels = next(iter(train_loader))
        print(f"Loader shapes: {tuple(sample_raw.shape)}, "
              f"{tuple(sample_labels.shape)}", flush=True)
        if sample_raw.shape[1:] != (3, 256, 256):
            raise ValueError("Unexpected LM training input shape")
        train_sam(
            name=TRAINING_NAME, model_type="vit_b_lm",
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=EPOCHS, early_stopping=3, n_sub_iteration=2,
            n_objects_per_batch=4, with_segmentation_decoder=True,
            freeze=["image_encoder"], lr=LR, save_root=str(output),
            verify_n_labels_in_loader=4, save_every_kth_epoch=1,
        )
    best = output / "checkpoints" / TRAINING_NAME / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    exported = output / "nissl_microsam_lm_best_loss.pth"
    export_custom_sam_model(best, model_type="vit_b", save_path=exported,
                            with_segmentation_decoder=True)
    print(f"Exported LM candidate: {exported}", flush=True)


def edge_misses(truth: np.ndarray, prediction: np.ndarray, band: int = 24):
    from skimage.measure import regionprops
    matched, _ = matched_ids(truth, prediction)
    result = {"edge": 0, "interior": 0}
    height, width = truth.shape
    for cell in regionprops(truth):
        if cell.label in matched:
            continue
        top, left, bottom, right = cell.bbox
        edge = (top < band or left < band or bottom > height - band
                or right > width - band)
        result["edge" if edge else "interior"] += 1
    return result


def evaluate(output: Path, *, baseline: bool, source_kind: str):
    import torch
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    splits = partition()
    if source_kind == "test":
        items = splits["test"]
    else:
        items = [item for item in splits["validation"]
                 if item["stem"] == "ID69_Visual_Cortex_Macaque"]
    checkpoint = output / "nissl_microsam_lm_best_loss.pth"
    if not baseline and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_lm" if baseline else "vit_b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint=None if baseline else str(checkpoint))
    rows = []
    for item in items:
        for name, gray, truth in crop_source(item):
            image = inverted_rgb(gray)
            predictor.set_image(image)
            embedding = {"features": predictor.get_image_embedding(),
                         "original_size": predictor.original_size,
                         "input_size": predictor.input_size}
            segmenter.initialize(image=image, image_embeddings=embedding)
            prediction = np.asarray(segmenter.generate(
                min_size=0, output_mode=None), dtype=np.uint32)
            row = score(truth, prediction)
            row["name"] = name
            row["misses_by_location"] = edge_misses(truth, prediction)
            rows.append(row)
            folder = output / ("baseline_masks" if baseline else "best_loss_masks") / source_kind
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{name}.npz", mask=prediction)
            print(f"{name}: TP={row['tp']} FN={row['fn']} FP={row['fp']} "
                  f"missed_edge={row['misses_by_location']['edge']} "
                  f"missed_interior={row['misses_by_location']['interior']}",
                  flush=True)
    total = summarize(rows)
    total["missed_edge"] = sum(row["misses_by_location"]["edge"] for row in rows)
    total["missed_interior"] = sum(row["misses_by_location"]["interior"] for row in rows)
    path = output / f"{'baseline' if baseline else 'best_loss'}_{source_kind}_scores.json"
    path.write_text(json.dumps({"overall": total, "patches": rows}, indent=2) + "\n")
    print("OVERALL", total, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate_baseline",
                                            "evaluate_best_loss"))
    parser.add_argument("--source", choices=("validation", "test"),
                        default="test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "train":
        train(args.output)
    else:
        evaluate(args.output, baseline=args.action == "evaluate_baseline",
                 source_kind=args.source)


if __name__ == "__main__":
    main()
