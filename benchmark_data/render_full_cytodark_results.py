"""Render the grouped CytoDArk0 Nissl validation comparison."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "cytodark_full_results"
OUTPUT = ROOT.parent / "docs"


def main():
    result = json.loads((RESULTS / "summary.json").read_text())
    groups = result["baseline"]["by_group"]
    sources = defaultdict(set)
    for fold in range(4):
        data = json.loads((RESULTS / f"fold_{fold}.json").read_text())
        for row in data["baseline"]["patches"]:
            sources[row["group"]].add(row["source"])
    selected_key = f"epoch_{result['selected_epochs']}"
    label_map = {
        "Bovine | Cerebellum": "Bovine cerebellum",
        "Chimpanzee | Cerebellum": "Chimpanzee cerebellum",
        "Chimpanzee | Visual Cortex": "Chimpanzee visual cortex",
        "Macaque | Visual Cortex": "Macaque visual cortex",
        "Mouse | Hippocampus": "Mouse hippocampus",
        "Tursiops Truncatus | Auditory Cortex": "Dolphin auditory cortex",
    }
    order = ["Overall", *sorted(groups, key=lambda g: -len(sources[g]))]
    labels, base, trained = [], [], []
    for group in order:
        if group == "Overall":
            labels.append("All 69 source fields")
            base.append(result["baseline"]["overall"]["f1"])
            trained.append(result[selected_key]["overall"]["f1"])
        else:
            labels.append(f"{label_map.get(group, group)}  (n={len(sources[group])})")
            base.append(groups[group]["f1"])
            trained.append(result[selected_key]["by_group"][group]["f1"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.linewidth": 1, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(order))[::-1]
    for i, (before, after) in enumerate(zip(base, trained)):
        ax.plot([before, after], [y[i], y[i]], color="#9ba2a7", lw=1.5, zorder=1)
    ax.scatter(base, y, s=62, color="#3972a5", edgecolor="white", linewidth=0.8,
               label="Original Cellpose-SAM", zorder=3)
    ax.scatter(trained, y, s=66, color="#d65a31", edgecolor="white", linewidth=0.8,
               label=f"Nissl fine-tuned ({result['selected_epochs']} epochs)", zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.55, 1.0)
    ax.set_xticks(np.arange(0.6, 1.01, 0.1))
    ax.set_xticklabels([f"{100 * value:.0f}%" for value in np.arange(0.6, 1.01, 0.1)])
    ax.set_xlabel("Cell instance F1 · field-held-out evaluation")
    ax.tick_params(axis="both", length=4, color="black", labelcolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.axhline(y[0] - 0.5, color="#d5d9dd", lw=0.9)
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              bbox_to_anchor=(1.0, -0.30), ncol=2, handletextpad=0.4,
              columnspacing=1.4)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.96, bottom=0.25)
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "Nissl_full_CytoDArk0_validation.png", dpi=240,
                facecolor="white")
    fig.savefig(OUTPUT / "Nissl_full_CytoDArk0_validation.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
