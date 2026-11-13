"""Select light-microscopy Micro-SAM settings on ID69, then evaluate ID68.

The original and fine-tuned models both use the same inverted, normalized
grayscale crops. Selection favors found expert cells, with a validation-only
cap of twice the original model's extra proposals.
"""

from __future__ import annotations

import gc
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.util import export_custom_sam_model

from finetune_cytodark import score, summarize
from train_full_cytodark import crop_source
from train_microsam_lm_nissl import (OUTPUT, TRAINING_NAME, edge_misses,
                                      inverted_rgb)
from train_microsam_nissl import partition

FOREGROUND_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
VALIDATION_SOURCE = "ID69_Visual_Cortex_Macaque"
TEST_SOURCE = "ID68_Visual_Cortex_Macaque"


def source_crops(stem: str):
    item = next(item for item in partition()["validation"] + partition()["test"]
                if item["stem"] == stem)
    return [(name, inverted_rgb(gray), labels)
            for name, gray, labels in crop_source(item)]


def make_model(path: Path | None):
    return get_predictor_and_segmenter(
        model_type="vit_b" if path else "vit_b_lm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint=str(path) if path else None)


def f3(result: dict) -> float:
    p, r = result["precision"], result["recall"]
    return 10 * p * r / (9 * p + r) if 9 * p + r else 0.0


def evaluate_grid(predictor, segmenter, items, thresholds, *, save: Path | None = None):
    rows = {threshold: [] for threshold in thresholds}
    for name, image, truth in items:
        predictor.set_image(image)
        embedding = {"features": predictor.get_image_embedding(),
                     "original_size": predictor.original_size,
                     "input_size": predictor.input_size}
        segmenter.initialize(image=image, image_embeddings=embedding)
        for threshold in thresholds:
            prediction = np.asarray(segmenter.generate(
                foreground_threshold=threshold, min_size=0,
                output_mode=None), dtype=np.uint32)
            row = score(truth, prediction)
            row["name"] = name
            row["misses_by_location"] = edge_misses(truth, prediction)
            rows[threshold].append(row)
            if save is not None:
                target = save / f"foreground_{threshold:g}"
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / f"{name}.npz", mask=prediction)
    result = {}
    for threshold, group in rows.items():
        total = summarize(group)
        total["f3"] = f3(total)
        total["missed_edge"] = sum(r["misses_by_location"]["edge"] for r in group)
        total["missed_interior"] = sum(r["misses_by_location"]["interior"] for r in group)
        result[str(threshold)] = {"overall": total, "patches": group}
    return result


def pick(candidates: list[dict], extra_cap: int) -> dict:
    eligible = [c for c in candidates if c["validation"]["fp"] <= extra_cap]
    if not eligible:
        raise RuntimeError("No candidate met the validation extra-mask cap")
    return max(eligible, key=lambda c: (
        c["validation"]["tp"],
        -c["validation"]["missed_interior"],
        c["validation"]["f3"],
        -c["validation"]["fp"],
    ))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = source_crops(VALIDATION_SOURCE)
    test = source_crops(TEST_SOURCE)
    original_predictor, original_segmenter = make_model(None)
    original_grid = evaluate_grid(original_predictor, original_segmenter,
                                  validation, FOREGROUND_THRESHOLDS)
    original_candidates = [
        {"checkpoint": "original vit_b_lm", "foreground_threshold": float(threshold),
         "validation": result["overall"]}
        for threshold, result in original_grid.items()
    ]
    original_default = next(c for c in original_candidates
                            if c["foreground_threshold"] == 0.5)
    extra_cap = 2 * original_default["validation"]["fp"]
    best_original = pick(original_candidates, extra_cap)
    print("Original default validation:", original_default, flush=True)
    print("Best original validation:", best_original, flush=True)
    del original_predictor, original_segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoints = sorted((OUTPUT / "checkpoints" / TRAINING_NAME).glob("epoch-*.pt"),
                         key=lambda p: int(re.search(r"epoch-(\d+)", p.name).group(1)))
    if not checkpoints:
        raise FileNotFoundError("No saved LM training epochs")
    trained_candidates = []
    with tempfile.TemporaryDirectory(prefix="lm_selection_", dir=OUTPUT) as temp:
        exported = Path(temp) / "candidate.pth"
        for checkpoint in checkpoints:
            export_custom_sam_model(checkpoint, model_type="vit_b",
                                    save_path=exported,
                                    with_segmentation_decoder=True)
            predictor, segmenter = make_model(exported)
            grid = evaluate_grid(predictor, segmenter, validation,
                                 FOREGROUND_THRESHOLDS)
            for threshold, result in grid.items():
                trained_candidates.append({
                    "checkpoint": checkpoint.name,
                    "foreground_threshold": float(threshold),
                    "validation": result["overall"],
                })
            print(checkpoint.name,
                  [(row["foreground_threshold"], row["validation"]["tp"],
                    row["validation"]["fn"], row["validation"]["fp"])
                   for row in trained_candidates if row["checkpoint"] == checkpoint.name],
                  flush=True)
            del predictor, segmenter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    best_trained = pick(trained_candidates, extra_cap)
    selected_path = OUTPUT / "nissl_microsam_lm_recall_selected.pth"
    export_custom_sam_model(
        OUTPUT / "checkpoints" / TRAINING_NAME / best_trained["checkpoint"],
        model_type="vit_b", save_path=selected_path,
        with_segmentation_decoder=True)
    predictor, segmenter = make_model(selected_path)
    trained_test = evaluate_grid(
        predictor, segmenter, test, (best_trained["foreground_threshold"],),
        save=OUTPUT / "recall_selected_masks")
    del predictor, segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    original_predictor, original_segmenter = make_model(None)
    original_test = evaluate_grid(
        original_predictor, original_segmenter, test,
        (best_original["foreground_threshold"],),
        save=OUTPUT / "original_recall_masks")
    record = {
        "training_start": "original vit_b_lm",
        "preprocess": "brightfield_invert",
        "validation_source": VALIDATION_SOURCE,
        "test_source": TEST_SOURCE,
        "selection_rule": "maximize found cells with extras <= 2x original default on ID69; tie by interior misses, F3, extras",
        "validation_extra_cap": extra_cap,
        "original_candidates": original_candidates,
        "trained_candidates": trained_candidates,
        "best_original": best_original,
        "best_trained": best_trained,
        "selected_model_path": str(selected_path),
        "original_test": original_test,
        "trained_test": trained_test,
    }
    (OUTPUT / "recall_selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print("SELECTED ORIGINAL", best_original, flush=True)
    print("SELECTED TRAINED", best_trained, flush=True)
    print("TEST ORIGINAL", next(iter(original_test.values()))["overall"], flush=True)
    print("TEST TRAINED", next(iter(trained_test.values()))["overall"], flush=True)


if __name__ == "__main__":
    main()
