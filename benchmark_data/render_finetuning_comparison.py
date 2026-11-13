"""Render two held-out gold-standard comparisons as a PNG and PDF."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image
from skimage.segmentation import find_boundaries

from cellpose.models import CellposeModel
from benchmark_data.finetune_cytodark import normalized_image, score, summarize, SAMPLES

EXAMPLES = ("ID58_Hippocampus_Mouse_4", "ID62_Visual_Cortex_Chimpanzee_3")
COLORS = {"human": (220, 173, 0), "baseline": (213, 94, 0),
          "fine": (0, 220, 90)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = {name: np.asarray(Image.open(SAMPLES / f"{name}.png").convert("RGB"))
              for name in EXAMPLES}
    truth = {name: tifffile.imread(SAMPLES / f"{name}.tiff") for name in EXAMPLES}
    outputs = {"human": truth}
    for kind, source in (("baseline", "cpsam"), ("fine", str(args.checkpoint))):
        model = CellposeModel(gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
                              pretrained_model=source)
        outputs[kind] = {}
        for name in EXAMPLES:
            outputs[kind][name] = np.asarray(model.eval(
                normalized_image(SAMPLES / f"{name}.png"), diameter=None,
                flow_threshold=0.4, cellprob_threshold=0.0)[0])
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="white")
    labels = {"human": "Human reviewed", "baseline": "Original Cellpose-SAM",
              "fine": "Nissl fine tuned"}
    for row, name in enumerate(EXAMPLES):
        for column, kind in enumerate(("human", "baseline", "fine")):
            output = images[name].copy()
            edge = find_boundaries(outputs[kind][name], mode="inner") & (outputs[kind][name] > 0)
            output[edge] = COLORS[kind]
            axis = axes[row, column]
            axis.imshow(output, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            title = labels[kind]
            if kind != "human":
                result = summarize([score(truth[name], outputs[kind][name])])
                title += f"  ·  F1 {100 * result['f1']:.1f}%"
            axis.set_title(title, fontsize=14, color="black", pad=10)
        axes[row, 0].set_ylabel(
            "Mouse hippocampus" if row == 0 else "Chimpanzee visual cortex",
            fontsize=14, color="black", labelpad=14)
    fig.tight_layout(pad=2.0, w_pad=1.2, h_pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix),
                    **({"dpi": 220} if suffix == ".png" else {}),
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
