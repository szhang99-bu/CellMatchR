"""
parameter_wizard_compute.py
============================
Stage 1 (headless) of the Parameter Wizard: runs UNFILTERED segmentation
on the first sample for every channel configured in Step 3, and writes
the raw channel image + raw label mask for each to disk, plus a manifest
describing what was produced.

This file deliberately has NO napari/Qt import anywhere. Combining the
segmentation stack (torch, cv2, micro_sam) with a live napari window in
one process crashes the interpreter on macOS here -- the same reason
segmentation.py (headless) and roi_editor.py (napari-only, no torch/cv2)
are already two separate scripts in this codebase rather than one. Stage
2 (parameter_wizard.py) is the napari display -- it only ever reads the
files this script writes, and never imports this module or cv2/torch/
micro_sam.

Launched by the CellMatchR launcher (Step 3 -> "Parameter Wizard"), which
runs this script first and parameter_wizard.py second. Can also be run
standalone:

    python parameter_wizard_compute.py --config ~/.cellmatchr/config.json --out-dir /path/to/scratch
"""

import argparse
import json
import os
import sys

# Standalone-subprocess bootstrap -- see parameter_wizard.py for why.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import tifffile

from cellmatchr import segmentation as seg
from cellmatchr.config import DEFAULT_CONFIG


def _load_cfg(path: str) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception as exc:
            print(f"[parameter_wizard_compute] [WARNING] Could not read {path}: {exc}")
    return cfg


def _find_first_sample(cfg: dict):
    """Same discovery logic segmentation.py uses, so "first sample" means
    exactly what it means during a real run (including confocal-only .lif
    filtering)."""
    root = cfg.get("input_root_folder", "").strip()
    files = seg.find_input_files(root)
    for fp in files:
        fb = os.path.splitext(os.path.basename(fp))[0]
        if fp.lower().endswith(".lif"):
            try:
                scenes = seg.lif_scenes_to_process(fp)
            except Exception as exc:
                print(f"[parameter_wizard_compute] Could not read '{fb}': {exc} -- trying next file.")
                continue
            if not scenes:
                continue  # slide-scanner-only .lif -- skip, same as a real run
            return fp, scenes[0][0], fb
        return fp, None, fb
    return None, None, None


def _setup_backend(cfg: dict):
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[parameter_wizard_compute] Using device: {device}")

    if cfg.get("seg_backend") == "cellpose":
        from cellpose import models as cp_models
        print("[parameter_wizard_compute] Backend: Cellpose-SAM")
        return None, None, cp_models.CellposeModel(gpu=(device != "cpu"))

    import micro_sam.automatic_segmentation
    model_type = cfg.get("seg_model_type", "vit_b_lm")
    print(f"[parameter_wizard_compute] Backend: Micro-SAM (model {model_type})")
    print(f"[parameter_wizard_compute] Loading model '{model_type}' -- if this is the first "
          "time this model has been used on this machine, it downloads once (can be a few "
          "hundred MB) and is cached at ~/Library/Caches/micro_sam/models for every future "
          "run, including normal Run Segmentation.")
    seg._patch_decoder_checkpoints(model_type)

    last_exc = None
    for attempt in range(1, 4):
        try:
            predictor, segmenter = micro_sam.automatic_segmentation.get_predictor_and_segmenter(
                model_type=model_type, device=device)
            return predictor, segmenter, None
        except Exception as exc:
            last_exc = exc
            print(f"[parameter_wizard_compute] Model load attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                print("[parameter_wizard_compute] Retrying -- this is usually a dropped "
                      "connection during the model download, not a real error.")
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="CellMatchR Parameter Wizard -- compute stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    channels = [int(x.strip()) for x in str(cfg.get("seg_channels", "")).split(",") if x.strip()]
    if not channels:
        print("[ERROR] No channels configured. Set 'Channels to predict' in Step 3 first.")
        sys.exit(1)

    image_path, lif_scene, base_name = _find_first_sample(cfg)
    if image_path is None:
        print("[ERROR] No usable input image found. Check the input folder in Step 2.")
        sys.exit(1)
    print(f"[parameter_wizard_compute] First sample: '{base_name}'  ({image_path})")
    print(f"[parameter_wizard_compute] Channels to calibrate: {channels}")

    os.makedirs(args.out_dir, exist_ok=True)
    print("[parameter_wizard_compute] Splitting channels...")
    split_paths = seg.save_raw_split_tiffs(image_path, args.out_dir, base_name, lif_scene=lif_scene)

    predictor, segmenter, cellpose_model = _setup_backend(cfg)

    # Match the configured tiling/preprocessing exactly, and disable every
    # post-filter (size/ratio/intensity) so the user sees every object the
    # backend detects and can decide where to draw the line themselves.
    # SKIP_EMPTY_TILES is deliberately left at its default (True): a real
    # Run Segmentation would skip the same blank/background tiles, so
    # leaving this on keeps the wizard's preview an accurate match for
    # what production actually produces -- and skips the same needless
    # work on large images with a lot of background.
    seg.PREPROCESS_MODE = cfg.get("seg_preprocess_mode", "local_bg_subtract")
    seg.CELLPOSE_DIAMETER = float(cfg.get("seg_cellpose_diameter", 0) or 0)
    seg.CELLPOSE_FLOW_THRESHOLD = float(cfg.get("seg_cellpose_flow_threshold", 0.4))
    seg.CELLPOSE_CELLPROB_THRESHOLD = float(cfg.get("seg_cellpose_cellprob_threshold", 0.0))
    seg.TILE_SHAPE_2D = (int(cfg.get("seg_tile_h", 2048)), int(cfg.get("seg_tile_w", 2048)))
    seg.HALO_2D = (int(cfg.get("seg_halo_h", 256)), int(cfg.get("seg_halo_w", 256)))
    seg.MIN_OBJECT_SIZE = 0
    seg.MAX_OBJECT_SIZE = 0
    seg.MAX_MAJOR_MINOR_RATIO = 0
    seg.RAW_MEAN_MIN = 0

    manifest = {"base_name": base_name, "channels": []}
    for ch in channels:
        tiff_path = split_paths.get(ch)
        if tiff_path is None:
            print(f"[parameter_wizard_compute] Channel {ch} not present in this file -- skipping.")
            manifest["channels"].append(
                {"channel": ch, "raw": None, "labels": None, "n_objects": 0})
            continue

        print(f"[parameter_wizard_compute] Channel {ch}: running unfiltered segmentation "
              "(this can take a while)...")
        image = tifffile.imread(tiff_path).astype(np.uint16)
        labels = seg.segment_by_tiling(
            predictor, segmenter, image,
            seg.TILE_SHAPE_2D, seg.HALO_2D, seg.NDIM,
            debug_folder=None, cellpose_model=cellpose_model,
            min_size=0,
        )
        n_objects = int(len(np.unique(labels[labels > 0])))
        print(f"[parameter_wizard_compute] Channel {ch}: {n_objects} unfiltered object(s) detected.")

        labels_path = os.path.join(args.out_dir, f"{base_name}_ch{ch}_unfiltered_labels.tif")
        seg.save_segmentation_tiff(labels_path, labels)

        manifest["channels"].append({
            "channel": ch, "raw": tiff_path, "labels": labels_path, "n_objects": n_objects,
        })
        seg._free_memory()

    manifest_path = os.path.join(args.out_dir, "parameter_wizard_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[parameter_wizard_compute] Wrote manifest: {manifest_path}")
    print("[parameter_wizard_compute] Done.")


if __name__ == "__main__":
    main()
