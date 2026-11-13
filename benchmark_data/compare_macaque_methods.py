"""Create same-field masks and cell-level scores for Nissl model comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from finetune_cytodark import score, summarize
from train_full_cytodark import load_crops, sources

DEFAULT_OUTPUT = Path.home() / "CellMatchR/Experiments/Nissl_Recall_20261113/comparison"
METHODS = (
    "watershed_distance", "watershed_intensity", "cpsam",
    "cpsam_high_recall", "nissl_general_v1",
    "nissl_general_v1_high_recall", "cyto2_native", "cyto2_inverted",
    "rat_nissl_native", "stardist_fluo", "stardist_he",
    "micro_sam_lm", "micro_sam_histopathology",
)


def macaque_crops() -> list[dict]:
    return load_crops([item for item in sources()
                       if item["group"] == "Macaque | Visual Cortex"])


def watershed_mask(image: np.ndarray, mode: str) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage import feature, filters, morphology, segmentation

    stain = 1.0 - image.astype(np.float32) / 255.0
    smooth = ndi.gaussian_filter(stain, sigma=1.0)
    local = filters.threshold_local(smooth, block_size=71, offset=0.025)
    foreground = smooth > np.maximum(local, 0.16)
    foreground = morphology.remove_small_objects(foreground, min_size=18)
    if mode == "watershed_distance":
        distance = ndi.distance_transform_edt(foreground)
        peaks = feature.peak_local_max(distance, min_distance=6,
                                       threshold_abs=1.6, labels=foreground)
        markers = np.zeros(image.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        elevation = -distance
    else:
        maxima = morphology.h_maxima(smooth, h=0.045) & foreground
        markers, _ = ndi.label(maxima)
        elevation = -smooth
    labels = segmentation.watershed(elevation, markers, mask=foreground)
    return morphology.remove_small_objects(labels, min_size=18).astype(np.uint32)


def load_model(method: str):
    if method.startswith("watershed"):
        return None
    if method.startswith(("cpsam", "nissl_general_v1")):
        import torch
        from cellpose.models import CellposeModel
        checkpoint = "cpsam"
        if method.startswith("nissl_general_v1"):
            library = Path.home() / "CellMatchR/Models"
            records = list(library.glob("cellpose_ch0_nissl_general_v1_*/model.json"))
            if len(records) != 1:
                raise RuntimeError(f"Expected one Nissl v1 model, found {len(records)}")
            info = json.loads(records[0].read_text())
            checkpoint = str(records[0].parent / info["model_relative_path"])
        return CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=checkpoint)
    if method.startswith(("cyto2", "rat_nissl")):
        from cellmatchr.legacy_cellpose import ensure_runtime
        variant = "cyto2" if method.startswith("cyto2") else "rat_nissl_v2"
        _python, checkpoint = ensure_runtime(variant)
        from cellpose.models import CellposeModel
        return CellposeModel(gpu=False, pretrained_model=checkpoint)
    if method.startswith("stardist"):
        from stardist.models import StarDist2D
        preset = "2D_versatile_fluo" if method == "stardist_fluo" else "2D_versatile_he"
        return StarDist2D.from_pretrained(preset)
    if method.startswith("micro_sam"):
        import torch
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        from cellmatchr.segmentation import _patch_decoder_checkpoints
        preset = "vit_b_lm" if method == "micro_sam_lm" else "vit_b_histopathology"
        _patch_decoder_checkpoints(preset)
        return get_predictor_and_segmenter(model_type=preset,
                                            device="cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(method)


def predict(method: str, model, crop: dict) -> np.ndarray:
    image = crop["image"]
    if method.startswith("watershed"):
        return watershed_mask(image, method)
    if method.startswith(("cpsam", "nissl_general_v1")):
        high_recall = method.endswith("high_recall")
        flow = 1.0 if high_recall else 0.4
        cellprob = (-1.0 if method.startswith("cpsam") else -2.0) if high_recall else 0.0
        return np.asarray(model.eval(image, diameter=None, flow_threshold=flow,
                                     cellprob_threshold=cellprob)[0])
    if method.startswith(("cyto2", "rat_nissl")):
        input_image = 255 - image if method == "cyto2_inverted" else image
        return np.asarray(model.eval(input_image, channels=[0, 0], diameter=30,
                                     flow_threshold=0.4, cellprob_threshold=0.0)[0])
    if method.startswith("stardist"):
        from csbdeep.utils import normalize
        if method == "stardist_fluo":
            input_image = normalize(255 - image, 1, 99.8)
        else:
            from PIL import Image
            from train_full_cytodark import DATA
            stem, row, col = crop["name"].rsplit("_", 2)
            original = np.asarray(Image.open(DATA / "image" / f"{stem}.png").convert("RGB"))
            input_image = normalize(original[int(row)*512:(int(row)+1)*512,
                                             int(col)*512:(int(col)+1)*512], 1, 99.8)
        return np.asarray(model.predict_instances(input_image)[0])
    if method.startswith("micro_sam"):
        from PIL import Image
        predictor, segmenter = model
        if method == "micro_sam_lm":
            input_image = np.repeat((255 - image)[:, :, None], 3, axis=2)
        else:
            from train_full_cytodark import DATA
            stem, row, col = crop["name"].rsplit("_", 2)
            original = np.asarray(Image.open(DATA / "image" / f"{stem}.png").convert("RGB"))
            input_image = original[int(row)*512:(int(row)+1)*512,
                                   int(col)*512:(int(col)+1)*512]
        predictor.set_image(input_image)
        embedding = {"features": predictor.get_image_embedding(),
                     "original_size": predictor.original_size,
                     "input_size": predictor.input_size}
        segmenter.initialize(image=input_image, image_embeddings=embedding)
        result = segmenter.generate(min_size=20)
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, list):
            labels = np.zeros(image.shape, dtype=np.uint32)
            for index, item in enumerate(result, 1):
                labels[item["segmentation"]] = index
            return labels
        raise TypeError(f"Unexpected Micro-SAM output: {type(result)}")
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output / args.method
    output.mkdir(parents=True, exist_ok=True)
    crops = macaque_crops()
    model = load_model(args.method)
    rows = []
    for index, crop in enumerate(crops, 1):
        mask_path = output / f"{crop['name']}.npz"
        if mask_path.is_file():
            prediction = np.load(mask_path)["mask"]
        else:
            prediction = predict(args.method, model, crop).astype(np.uint32)
            np.savez_compressed(mask_path, mask=prediction)
        row = score(crop["mask"], prediction)
        row["name"] = crop["name"]
        rows.append(row)
        print(f"{args.method} {index}/{len(crops)} {crop['name']}: "
              f"TP={row['tp']} FN={row['fn']} FP={row['fp']}", flush=True)
    totals = summarize(rows)
    p, r = totals["precision"], totals["recall"]
    totals["f2"] = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
    (output / "scores.json").write_text(json.dumps({"method": args.method,
                                                    "overall": totals,
                                                    "patches": rows}, indent=2) + "\n")
    print(f"Overall {args.method}: {totals}", flush=True)


if __name__ == "__main__":
    main()
