"""Compare original and fine-tuned Micro-SAM against human macaque masks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from compare_macaque_methods import DEFAULT_OUTPUT
from finetune_cytodark import score
from render_macaque_comparison import load_crop, matched_ids

NAME = "ID68_Visual_Cortex_Macaque_0_0"
EXPERIMENT = Path.home() / "CellMatchR/Experiments/Nissl_MicroSAM_20261113"


def _draw(axis, image: np.ndarray, truth: np.ndarray, mask: np.ndarray | None,
          title: str) -> None:
    axis.imshow(image, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if mask is None:
        boundaries = find_boundaries(truth, mode="inner") & (truth > 0)
        rgba = np.zeros((*truth.shape, 4), dtype=float)
        rgba[boundaries] = (1, 0.82, 0, 1)
        axis.imshow(rgba, interpolation="nearest")
        axis.set_title(f"{title}\n{len(np.unique(truth)) - 1} expert cells", fontsize=12,
                       fontweight="semibold", pad=7)
        return
    true_matches, pred_matches = matched_ids(truth, mask)
    boundaries = find_boundaries(mask, mode="inner") & (mask > 0)
    matched = np.isin(mask, list(pred_matches))
    rgba = np.zeros((*truth.shape, 4), dtype=float)
    rgba[boundaries & matched] = (0, 0.75, 0.42, 1)
    rgba[boundaries & ~matched] = (1, 0.55, 0, 1)
    axis.imshow(rgba, interpolation="nearest")
    for cell in regionprops(truth):
        if cell.label not in true_matches:
            y, x = cell.centroid
            axis.add_patch(Circle((x, y), 9.5, fill=False,
                                  edgecolor="#ed3359", linewidth=1.7))
    value = score(truth, mask)
    axis.set_title(f"{title}\nFound {value['tp']}  ·  missed {value['fn']}  ·  extra {value['fp']}",
                   fontsize=12, fontweight="semibold", pad=7)


def render(image, truth, masks, output, name, footnote):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(17.6, 5.7), dpi=190,
                             facecolor="white")
    for axis, (title, mask) in zip(axes, masks.items()):
        _draw(axis, image, truth, mask, title)
    fig.suptitle("Macaque visual cortex · held-out source ID68",
                 fontsize=17, fontweight="semibold", y=0.99)
    fig.legend(handles=[Patch(color="#00bf6b", label="Matched prediction"),
                        Patch(color="#ff8c00", label="Extra proposal"),
                        Patch(facecolor="none", edgecolor="#ed3359",
                              label="Missed expert cell")],
               loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.014, footnote, ha="center", fontsize=8.5,
             color="#525b65")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.13,
                        wspace=0.06)
    fig.savefig(output / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    output = EXPERIMENT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    image, truth = load_crop()
    original = np.load(EXPERIMENT / "baseline_masks/min_0" /
                       f"{NAME}.npz")["mask"]
    fine_tuned = np.load(EXPERIMENT / "heldout_masks/min_0" /
                         f"{NAME}.npz")["mask"]
    selected = np.load(EXPERIMENT / "recall_selected_masks" /
                       f"{NAME}.npz")["mask"]
    cellpose = np.load(DEFAULT_OUTPUT / "recall_candidate_held_out" /
                       f"{NAME}.npz")["mask"]
    render(image, truth, {
        "Expert annotation": None,
        "Original Micro-SAM": original,
        "Fine-tuned Micro-SAM": fine_tuned,
        "Cellpose recall candidate": cellpose,
    }, output, "microsam_before_after_heldout",
        "Both fine-tuned models held out this source field. Micro-SAM uses original RGB stain; Cellpose uses native gray.")
    render(image, truth, {
        "Expert annotation": None,
        "Original Micro-SAM": original,
        "Fine-tuned · loss selected": fine_tuned,
        "Fine-tuned · recall selected": selected,
    }, output, "microsam_recall_selection_heldout",
        "Checkpoint and proposal settings selected on source ID69; source ID68 is the test field. The original still has best recall.")
    for label, filename in (("Original Micro-SAM", "baseline_scores.json"),
                            ("Fine-tuned Micro-SAM", "heldout_scores.json")):
        data = json.loads((EXPERIMENT / filename).read_text())
        print(label, data["0"]["overall"])


if __name__ == "__main__":
    main()
