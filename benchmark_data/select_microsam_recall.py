"""Select a Micro-SAM checkpoint and proposal settings on validation data.

ID69 is used for selection. ID68 is evaluated only after settings are fixed.
This is an exploratory follow-up to the original default-setting test.
"""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.util import export_custom_sam_model

from finetune_cytodark import score, summarize
from train_full_cytodark import DATA

OUTPUT = Path.home() / "CellMatchR/Experiments/Nissl_MicroSAM_20261113"
CHECKPOINTS = OUTPUT / "checkpoints/nissl_microsam_histopathology"
VALIDATION = "ID69_Visual_Cortex_Macaque"
TEST = "ID68_Visual_Cortex_Macaque"
CHECKPOINT_NAMES = ("epoch-2.pt", "epoch-4.pt", "epoch-6.pt", "best.pt",
                    "epoch-8.pt")
SETTINGS = [
    {"center_distance_threshold": center,
     "boundary_distance_threshold": boundary,
     "foreground_threshold": foreground,
     "min_size": 0}
    for center in (0.35, 0.5)
    for boundary in (0.35, 0.5)
    for foreground in (0.2, 0.3, 0.4, 0.5)
]


def crops(stem: str):
    rgb = np.asarray(Image.open(DATA / "image" / f"{stem}.png").convert("RGB"))
    truth = tifffile.imread(DATA / "label" / f"{stem}.tiff")
    return [(f"{stem}_{r}_{c}",
             rgb[r*512:(r+1)*512, c*512:(c+1)*512],
             truth[r*512:(r+1)*512, c*512:(c+1)*512])
            for r in range(2) for c in range(2)]


def f3(row: dict) -> float:
    precision, recall = row["precision"], row["recall"]
    denominator = 9 * precision + recall
    return 10 * precision * recall / denominator if denominator else 0.0


def evaluate(predictor, segmenter, dataset, settings, *, save_dir: Path | None = None):
    rows = []
    for name, image, truth in dataset:
        predictor.set_image(image)
        embedding = {"features": predictor.get_image_embedding(),
                     "original_size": predictor.original_size,
                     "input_size": predictor.input_size}
        segmenter.initialize(image=image, image_embeddings=embedding)
        prediction = segmenter.generate(output_mode=None, **settings)
        row = score(truth, prediction)
        row["name"] = name
        rows.append(row)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / f"{name}.npz", mask=prediction)
    total = summarize(rows)
    total["f3"] = f3(total)
    return {"overall": total, "patches": rows}


def make_model(checkpoint: Path):
    return get_predictor_and_segmenter(
        model_type="vit_b", device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint=str(checkpoint))


def main():
    validation = crops(VALIDATION)
    test = crops(TEST)
    default = SETTINGS[-1]
    checkpoint_results = []
    with tempfile.TemporaryDirectory(prefix="micro_sam_checkpoint_", dir=OUTPUT) as temp:
        transient = Path(temp) / "candidate.pth"
        for name in CHECKPOINT_NAMES:
            source = CHECKPOINTS / name
            export_custom_sam_model(source, model_type="vit_b",
                                    save_path=transient,
                                    with_segmentation_decoder=True)
            predictor, segmenter = make_model(transient)
            result = evaluate(predictor, segmenter, validation, default)
            checkpoint_results.append({"checkpoint": name,
                                       "validation": result["overall"]})
            print(name, result["overall"], flush=True)
            del predictor, segmenter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    chosen = max(checkpoint_results,
                 key=lambda row: (row["validation"]["f3"],
                                  row["validation"]["recall"],
                                  -row["validation"]["fp"]))["checkpoint"]
    selected = OUTPUT / "nissl_microsam_recall_selected.pth"
    if chosen == "best.pt":
        selected = OUTPUT / "nissl_microsam_histopathology.pth"
    else:
        export_custom_sam_model(CHECKPOINTS / chosen, model_type="vit_b",
                                save_path=selected,
                                with_segmentation_decoder=True)
    predictor, segmenter = make_model(selected)
    setting_results = []
    for settings in SETTINGS:
        result = evaluate(predictor, segmenter, validation, settings)
        setting_results.append({"settings": settings,
                                "validation": result["overall"]})
        print(settings, result["overall"], flush=True)
    winner = max(setting_results,
                 key=lambda row: (row["validation"]["f3"],
                                  row["validation"]["recall"],
                                  -row["validation"]["fp"]))
    heldout = evaluate(predictor, segmenter, test, winner["settings"],
                       save_dir=OUTPUT / "recall_selected_masks")
    record = {"checkpoint": chosen, "model_path": str(selected),
              "selection_source": VALIDATION, "test_source": TEST,
              "selection_metric": "F3",
              "checkpoint_results": checkpoint_results,
              "setting_results": setting_results,
              "chosen_settings": winner["settings"],
              "chosen_validation": winner["validation"],
              "heldout": heldout}
    (OUTPUT / "recall_selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print("SELECTED", chosen, winner["settings"], flush=True)
    print("HELDOUT", heldout["overall"], flush=True)


if __name__ == "__main__":
    main()
