"""Plot missed expert cells against extra mask proposals on macaque Nissl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from compare_macaque_methods import DEFAULT_OUTPUT

NAMES = {
    "cpsam": "Cellpose-SAM",
    "cpsam_high_recall": "Cellpose-SAM · recall",
    "nissl_general_v1": "Nissl v1 · default*",
    "nissl_general_v1_high_recall": "Nissl v1 · recall*",
    "recall_candidate_held_out": "New candidate · held out",
    "cyto2_native": "Cellpose 2 · native",
    "cyto2_inverted": "Cellpose 2 · inverted",
    "rat_nissl_native": "Rat Nissl v2",
    "micro_sam_lm": "Micro-SAM · LM",
    "micro_sam_histopathology": "Micro-SAM · histopathology",
    "stardist_fluo": "StarDist · fluorescence",
    "stardist_he": "StarDist · H&E",
    "watershed_tuned": "Watershed · tuned",
}
FOCUS = {"cpsam", "nissl_general_v1_high_recall",
         "recall_candidate_held_out", "micro_sam_histopathology",
         "watershed_tuned"}
COLORS = {
    "cpsam": "#336ca2",
    "nissl_general_v1_high_recall": "#d9592d",
    "recall_candidate_held_out": "#0b8b6b",
    "micro_sam_histopathology": "#8850a5",
    "watershed_tuned": "#8b6b21",
}
LABEL_OFFSETS = {
    "cpsam": (7, -13),
    "cpsam_high_recall": (7, 18),
    "nissl_general_v1": (7, -12),
    "nissl_general_v1_high_recall": (7, 5),
    "recall_candidate_held_out": (-70, -37),
    "cyto2_native": (7, -13),
    "cyto2_inverted": (7, 4),
    "rat_nissl_native": (7, -13),
    "micro_sam_lm": (-97, 6),
    "micro_sam_histopathology": (7, -15),
    "stardist_fluo": (7, 5),
    "stardist_he": (7, 5),
    "watershed_tuned": (-91, 5),
}


def main():
    rows = []
    for method, name in NAMES.items():
        file = DEFAULT_OUTPUT / method / "scores.json"
        if not file.is_file():
            continue
        data = json.loads(file.read_text())
        scores = (data["selected"]["all"] if method == "watershed_tuned"
                  else data["overall"])
        rows.append((method, name, scores["fn"], scores["fp"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.linewidth": 1})
    fig, ax = plt.subplots(figsize=(9.4, 6.7), dpi=190, facecolor="white")
    ax.set_facecolor("white")
    for method, name, fn, fp in rows:
        color = COLORS.get(method, "#9ca3aa")
        size = 105 if method in FOCUS else 52
        ax.scatter(fn, fp, s=size, color=color, edgecolor="white",
                   linewidth=0.9, zorder=3 if method in FOCUS else 2)
        ax.annotate(name, (fn, fp), xytext=LABEL_OFFSETS.get(method, (7, 5)),
                    textcoords="offset points",
                    fontsize=9, color="#1d2430" if method in FOCUS else "#67717b")
    ax.set_xlabel("Missed expert cells (false negatives)  ·  fewer is better", labelpad=10)
    ax.set_ylabel("Extra / unmatched masks  ·  fewer is better", labelpad=10)
    ax.set_title("Macaque visual cortex · 843 expert cells across 16 crops",
                 loc="left", fontsize=14, fontweight="semibold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#1d2430", direction="out", length=4)
    ax.spines["left"].set_color("#1d2430")
    ax.spines["bottom"].set_color("#1d2430")
    ax.set_xlim(30, 180)
    ax.set_ylim(-20, 880)
    fig.text(0.09, 0.015,
             "Raw mask proposals before CellMatchR filters. *Nissl v1 trained on these images; "
             "new candidate uses source-held-out fold models.",
             fontsize=8.2, color="#525b65")
    fig.subplots_adjust(left=0.13, right=0.92, top=0.91, bottom=0.15)
    output = DEFAULT_OUTPUT.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "macaque_recall_tradeoff.png", dpi=220, facecolor="white")
    fig.savefig(output / "macaque_recall_tradeoff.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
