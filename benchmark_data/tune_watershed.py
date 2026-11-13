"""Tune a marker-controlled watershed on three macaque source fields.

ID68 is left out of parameter selection and used only for the test score.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, morphology, segmentation

from compare_macaque_methods import DEFAULT_OUTPUT, macaque_crops
from finetune_cytodark import score, summarize


def mask(image: np.ndarray, floor: float, separation: int, area: int) -> np.ndarray:
    stain = 1.0 - image.astype(np.float32) / 255.0
    smooth = ndi.gaussian_filter(stain, sigma=1.0)
    local = filters.threshold_local(smooth, block_size=71, offset=0.025)
    foreground = smooth > np.maximum(local, floor)
    foreground = morphology.remove_small_objects(foreground, min_size=area)
    distance = ndi.distance_transform_edt(foreground)
    peaks = feature.peak_local_max(distance, min_distance=separation,
                                   threshold_abs=1.6, labels=foreground)
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-distance, markers, mask=foreground)
    return morphology.remove_small_objects(labels, min_size=area).astype(np.uint32)


def metrics(crops: list[dict], config: dict):
    rows = []
    for crop in crops:
        prediction = mask(crop["image"], **config)
        row = score(crop["mask"], prediction)
        row["name"] = crop["name"]
        rows.append(row)
    total = summarize(rows)
    p, r = total["precision"], total["recall"]
    total["f2"] = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
    return total, rows


def main() -> None:
    crops = macaque_crops()
    tuning = [crop for crop in crops if not crop["name"].startswith("ID68_")]
    test = [crop for crop in crops if crop["name"].startswith("ID68_")]
    results = []
    for floor, separation, area in itertools.product(
            (0.30, 0.38, 0.45, 0.52, 0.60), (6, 9, 12), (20, 40, 80)):
        config = {"floor": floor, "separation": separation, "area": area}
        total, _ = metrics(tuning, config)
        results.append({"config": config, "tuning": total})
        print(config, "recall", round(total["recall"], 3),
              "f2", round(total["f2"], 3), "fp", total["fp"], flush=True)
    selected = max(results, key=lambda row: row["tuning"]["f2"])
    selected["test"], _ = metrics(test, selected["config"])
    selected["all"], rows = metrics(crops, selected["config"])
    output = DEFAULT_OUTPUT / "watershed_tuned"
    output.mkdir(parents=True, exist_ok=True)
    for crop in crops:
        np.savez_compressed(output / f"{crop['name']}.npz",
                            mask=mask(crop["image"], **selected["config"]))
    (output / "scores.json").write_text(json.dumps({"method": "watershed_tuned",
                                                    "selection": "F2 on ID66, ID67, ID69",
                                                    "selected": selected,
                                                    "patches": rows,
                                                    "grid": results}, indent=2) + "\n")
    print("SELECTED", selected, flush=True)


if __name__ == "__main__":
    main()
