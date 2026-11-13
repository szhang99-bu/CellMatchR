"""Render original versus gently fine-tuned light-microscopy Micro-SAM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from render_microsam_heldout import NAME, render
from train_full_cytodark import DATA
from train_microsam_lm_nissl import OUTPUT


def load_crop(name: str):
    stem, row, column = name.rsplit("_", 2)
    top, left = int(row) * 512, int(column) * 512
    image = np.asarray(Image.open(DATA / "image" / f"{stem}.png").convert("RGB"))
    truth = tifffile.imread(DATA / "label" / f"{stem}.tiff")
    crop = np.s_[top:top + 512, left:left + 512]
    return image[crop], truth[crop]


def main():
    record = json.loads((OUTPUT / "recall_selection.json").read_text())
    threshold = record["best_trained"]["foreground_threshold"]
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    for name in (NAME, "ID68_Visual_Cortex_Macaque_0_1"):
        image, truth = load_crop(name)
        original = np.load(OUTPUT / "baseline_masks/test" /
                           f"{name}.npz")["mask"]
        best_loss = np.load(OUTPUT / "best_loss_masks/test" /
                            f"{name}.npz")["mask"]
        selected = np.load(OUTPUT / "recall_selected_masks" /
                           f"foreground_{threshold:g}" / f"{name}.npz")["mask"]
        figure_name = ("microsam_lm_before_after_heldout" if name == NAME
                       else f"microsam_lm_before_after_{name}")
        render(image, truth, {
            "Expert annotation": None,
            "Original Micro-SAM LM": original,
            "Fine-tuned · loss selected": best_loss,
            "Fine-tuned · recall selected": selected,
        }, OUTPUT / "figures", figure_name,
            "Fine-tuned model started from original LM weights. Source ID68 was excluded from training and validation; proposal settings used ID69.")
        print(OUTPUT / "figures" / f"{figure_name}.png")


if __name__ == "__main__":
    main()
