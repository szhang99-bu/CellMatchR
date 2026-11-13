"""Render human versus model masks on the same macaque visual-cortex crop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from matplotlib.patches import Circle, Patch
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from compare_macaque_methods import DEFAULT_OUTPUT
from finetune_cytodark import score
from train_full_cytodark import DATA

NAME = "ID68_Visual_Cortex_Macaque_0_0"
FOCUSED = (
    ("human", "Expert annotation"),
    ("cpsam", "Original Cellpose-SAM"),
    ("nissl_general_v1", "Nissl v1 · default*"),
    ("nissl_general_v1_high_recall", "Nissl v1 · recall setting*"),
    ("recall_candidate_held_out", "Recall candidate · held out"),
    ("watershed_tuned", "Watershed · tuned"),
)
ALTERNATIVES = (
    ("human", "Expert annotation"),
    ("cpsam", "Cellpose-SAM"),
    ("nissl_general_v1_high_recall", "Nissl v1 · recall*"),
    ("recall_candidate_held_out", "New candidate · held out"),
    ("cyto2_native", "Cellpose 2 · native"),
    ("cyto2_inverted", "Cellpose 2 · inverted"),
    ("rat_nissl_native", "Rat Nissl v2"),
    ("micro_sam_lm", "Micro-SAM · light microscopy"),
    ("micro_sam_histopathology", "Micro-SAM · histopathology"),
    ("stardist_fluo", "StarDist · fluorescence"),
    ("stardist_he", "StarDist · H&E"),
    ("watershed_tuned", "Watershed"),
)


def matched_ids(truth: np.ndarray, prediction: np.ndarray):
    true_ids, true_flat = np.unique(truth, return_inverse=True)
    pred_ids, pred_flat = np.unique(prediction, return_inverse=True)
    overlap = np.bincount(true_flat * len(pred_ids) + pred_flat,
                          minlength=len(true_ids) * len(pred_ids))
    overlap = overlap.reshape(len(true_ids), len(pred_ids))
    true_area = overlap.sum(axis=1)
    pred_area = overlap.sum(axis=0)
    true_indices = np.flatnonzero(true_ids != 0)
    pred_indices = np.flatnonzero(pred_ids != 0)
    inter = overlap[np.ix_(true_indices, pred_indices)]
    union = true_area[true_indices, None] + pred_area[None, pred_indices] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter, dtype=float),
                    where=union > 0)
    candidates = sorted(np.argwhere(iou >= 0.5),
                        key=lambda pair: iou[tuple(pair)], reverse=True)
    used_true, used_pred = set(), set()
    for ti, pi in candidates:
        tid, pid = int(true_ids[true_indices[ti]]), int(pred_ids[pred_indices[pi]])
        if tid not in used_true and pid not in used_pred:
            used_true.add(tid)
            used_pred.add(pid)
    return used_true, used_pred


def load_crop():
    stem = "ID68_Visual_Cortex_Macaque"
    image = np.asarray(Image.open(DATA / "image" / f"{stem}.png").convert("RGB"))[:512, :512]
    truth = tifffile.imread(DATA / "label" / f"{stem}.tiff")[:512, :512]
    return image, truth


def draw(axis, image: np.ndarray, truth: np.ndarray, method: str, label: str,
         root: Path) -> None:
    axis.imshow(image, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if method == "human":
        boundary = find_boundaries(truth, mode="inner") & (truth > 0)
        overlay = np.zeros((*truth.shape, 4), dtype=np.float32)
        overlay[boundary] = (1.0, 0.82, 0.0, 1.0)
        axis.imshow(overlay, interpolation="nearest")
        axis.set_title(f"{label}\n{len(np.unique(truth))-1} outlined cells",
                       fontsize=12, fontweight="semibold", pad=8)
        return
    path = root / method / f"{NAME}.npz"
    if not path.is_file():
        axis.set_title(label + "\nPending mask", fontsize=12, pad=8)
        return
    prediction = np.load(path)["mask"]
    true_match, pred_match = matched_ids(truth, prediction)
    boundary = find_boundaries(prediction, mode="inner") & (prediction > 0)
    is_matched = np.isin(prediction, list(pred_match))
    overlay = np.zeros((*truth.shape, 4), dtype=np.float32)
    overlay[boundary & is_matched] = (0.0, 0.75, 0.42, 1.0)
    overlay[boundary & ~is_matched] = (1.0, 0.55, 0.0, 1.0)
    axis.imshow(overlay, interpolation="nearest")
    for region in regionprops(truth):
        if region.label not in true_match:
            y, x = region.centroid
            axis.add_patch(Circle((x, y), radius=9.5, fill=False,
                                  edgecolor="#ed3359", linewidth=1.7))
    result = score(truth, prediction)
    axis.set_title(f"{label}\nTP {result['tp']}  ·  missed {result['fn']}  ·  extra {result['fp']}",
                   fontsize=12, fontweight="semibold", pad=8)


def render(panels, columns: int, name: str, root: Path, output: Path):
    image, truth = load_crop()
    rows = math.ceil(len(panels) / columns)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(rows, columns, figsize=(4.1*columns, 4.6*rows),
                             dpi=190, facecolor="white")
    for axis, (method, label) in zip(np.ravel(axes), panels):
        draw(axis, image, truth, method, label, root)
    for axis in np.ravel(axes)[len(panels):]:
        axis.axis("off")
    fig.suptitle("Macaque visual cortex · same 512 × 512 Nissl field",
                 fontsize=17, fontweight="semibold", y=0.992)
    fig.legend(handles=[Patch(color="#00bf6b", label="Matched predicted outline"),
                        Patch(color="#ff8c00", label="Extra / unmatched outline"),
                        Patch(facecolor="none", edgecolor="#ed3359",
                              label="Missed expert cell")],
               loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.024))
    fig.text(0.5, 0.014, "* Nissl v1 used these crops for training; the new candidate held out their source fields.",
             ha="center", fontsize=8.5, color="#525b65")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.865, bottom=0.09,
                        hspace=0.28, wspace=0.07)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_OUTPUT.parent / "figures")
    args = parser.parse_args()
    render(FOCUSED, 3, "macaque_before_after", args.root, args.output)
    render(ALTERNATIVES, 4, "macaque_all_models", args.root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
