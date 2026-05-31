"""
CellMatchr configuration management.
Config is stored as a flat JSON dict at ~/.cellmatchr/config.json.
"""

import json
import os

CONFIG_PATH = os.path.expanduser("~/.cellmatchr/config.json")

DEFAULT_CONFIG: dict = {
    # --- Project identity ---
    "project_name": "",
    "project_output_folder": "",   # single root folder for all outputs

    # --- Input ---
    "input_type": "czi",           # "czi" (CZI/LIF/TIF direct) | "vsi"
    "input_root_folder": "",       # where raw CZI/LIF/TIF files live

    # --- Segmentation ---
    "seg_channels": "",
    "seg_model_type": "vit_b_lm",
    "seg_preprocess_mode": "local_bg_subtract",
    "seg_min_object_size": 0,
    "seg_max_object_size": 0,
    "seg_max_ratio": 0.0,
    "seg_raw_mean_min": 0.0,
    "seg_tile_h": 2048,
    "seg_tile_w": 2048,
    "seg_halo_h": 256,
    "seg_halo_w": 256,
    "seg_qc_downsample": 4,
    "seg_test_run": False,
    "seg_dry_run": False,
    "seg_skip_subdirs": False,

    # --- Validation ---
    "val_enabled": False,
    "val_mask_pattern": "*_predict_original_ch2*.tif",
    "val_channel_names": "",   # comma-separated channel display names, e.g. "DAPI,GAD67,AAV Cre,NeuN"
    "val_channel_colors": "",  # comma-separated colors: Blue,Green,Red,Purple,Cyan,Yellow

    # --- Analysis ---
    "ana_enabled": True,
    "ana_coloc_enabled": True,
    "ana_red_channel": "1",
    "ana_pv_channel": "3",
    "ana_iou_threshold": "0.35",
    "ana_roi_cell_rule": "centroid",
    "ana_roi_mask_folder": "",
    "ana_subfolder_name": "analysis_results",
    "ana_save_qc": True,
}


PROJECT_CONFIG_FILE = "cellmatchr_project.json"


def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            cfg.update(saved)
        except Exception as e:
            print(f"[config] Warning: could not load {CONFIG_PATH}: {e}")
    return cfg


def save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def save_project_config(cfg: dict) -> None:
    """Save config into the project output folder for later re-opening."""
    folder = cfg.get("project_output_folder", "").strip()
    if not folder:
        return
    try:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, PROJECT_CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[config] Warning: could not save project config: {e}")


def load_project_config(folder: str) -> dict | None:
    """Load config from a project folder. Returns None if not found."""
    path = os.path.join(folder, PROJECT_CONFIG_FILE)
    if not os.path.exists(path):
        return None
    try:
        cfg = DEFAULT_CONFIG.copy()
        with open(path) as f:
            saved = json.load(f)
        cfg.update(saved)
        return cfg
    except Exception as e:
        print(f"[config] Warning: could not load project config from {path}: {e}")
        return None
