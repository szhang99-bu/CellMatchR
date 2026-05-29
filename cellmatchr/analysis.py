import os
import re
import glob
import numpy as np
import pandas as pd
import tifffile
import cv2
from tqdm import tqdm
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries
from skimage.draw import polygon as sk_polygon


# =============================================================================
# USER SETTINGS
# =============================================================================

MICROSAM_OUTPUT_ROOT = ""   # input: segmentation folder containing project sub-dirs
ANALYSIS_OUTPUT_ROOT  = ""   # output: where combined CSV/Excel are written

# ROI mask folder — only used for the VSI pipeline.
# Leave empty ("") to skip ROI analysis entirely.
ROI_MASK_FOLDER = ""

# Channel indices (0-based), matching segmentation output
COI_CHANNEL = 1   # Cell of Interest channel
PM_CHANNEL  = 3   # Protein Marker channel

IOU_THRESHOLD = 0.35

# "centroid" = object inside ROI if centroid is inside
# "any_pixel" = object inside ROI if any pixel touches ROI
ROI_CELL_RULE = "centroid"

# When True, validation filtering is applied even if only some (not all) ROIs
# in a project have been reviewed.  Default False — only fully-validated
# projects are filtered automatically.
APPLY_PARTIAL_VALIDATION = False

ANALYSIS_SUBFOLDER_NAME = "colocalization_analysis"

SAVE_QC_PNG = True
ROI_BORDER_THICKNESS_PX     = 6
OBJECT_BORDER_THICKNESS_PX  = 6
MATCHED_BORDER_THICKNESS_PX = 6

SPLIT_TIFF_PATTERN = "*_split_ch{ch}.tif"
PRED_TIFF_PATTERN  = "*_predict_original_ch{ch}*.tif"

EXCLUDE_FROM_LABEL_SEARCH = [
    "coverage", "combined", "DEBUG", "debug", "UNFILTERED",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_percent(numerator, denominator):
    """Return a fraction in [0, 1] (e.g. 0.3221), not a 0-100 integer.

    Stored as a real fraction so Excel can display it with a percent number
    format (shows 32.21%, copies as 0.3221) and the CSV holds the raw 0.3221.
    """
    if denominator == 0:
        return np.nan
    return numerator / denominator


def robust_uint8(image, p_low=0.5, p_high=99.8):
    image = image.astype(np.float32)
    low, high = np.percentile(image, [p_low, p_high])
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return (image * 255).astype(np.uint8)


def find_one_file(folder, pattern, required=True):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    filtered = [p for p in paths
                if not any(excl in os.path.basename(p)
                           for excl in EXCLUDE_FROM_LABEL_SEARCH)]
    paths = filtered
    if len(paths) == 0:
        if required:
            raise FileNotFoundError(f"No file found for pattern: {pattern} in {folder}")
        return None
    if len(paths) > 1:
        print(f"  [WARNING] Multiple files found for pattern: {pattern}")
        for p in paths:
            print(f"    {os.path.basename(p)}")
        print(f"  Using: {os.path.basename(paths[-1])}")
    return paths[-1]


def get_project_folders(root):
    folders = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("DEBUG") or name.startswith("."):
            continue
        folders.append(path)
    return folders


def normalize_name_for_matching(name):
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r"_predict_original_ch\d+.*$", "", name)
    suffixes = [
        "_mask_overview", "_overview", "_vertices_overview_yx",
        "_split_ch0", "_split_ch1", "_split_ch2", "_split_ch3",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = re.sub(r"(^|_)FULLRES(_|$)", "_", name, flags=re.IGNORECASE)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# =============================================================================
# ROI MASK HANDLING
# =============================================================================

def find_roi_mask_for_project(project_name, roi_mask_folder):
    """
    Find the ROI mask file for a project.

    Searches for any .tif file in roi_mask_folder whose name contains the
    normalised project name (or vice versa). VSI-pipeline files named
    *_mask_overview.tif are matched the same way as plain TIF masks.

    The mask file must contain the original image / project name somewhere
    in its filename so the association is unambiguous.

    Returns None (instead of raising) when the folder is empty or mask is missing.
    """
    if not roi_mask_folder or not os.path.isdir(roi_mask_folder):
        return None

    candidates = sorted(
        glob.glob(os.path.join(roi_mask_folder, "*.tif")) +
        glob.glob(os.path.join(roi_mask_folder, "*.roi"))
    )
    if not candidates:
        print(f"  [WARNING] No .tif or .roi mask files found in {roi_mask_folder}")
        return None

    project_norm = normalize_name_for_matching(project_name)

    exact_matches = [p for p in candidates
                     if normalize_name_for_matching(p) == project_norm]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        print("  [WARNING] Multiple exact ROI mask matches. Using first.")
        return exact_matches[0]

    fuzzy_matches = [p for p in candidates
                     if project_norm in normalize_name_for_matching(p)
                     or normalize_name_for_matching(p) in project_norm]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        print("  [WARNING] Multiple fuzzy ROI mask matches. Using first.")
        return fuzzy_matches[0]

    print(f"  [WARNING] No ROI mask found for: {project_name} — ROI analysis skipped.")
    return None


def crop_and_resize_overview_roi_mask_to_image(overview_mask, target_shape):
    """
    1. Find the bounding box of the ROI in overview space.
    2. Crop the overview mask to that bounding box.
    3. Resize that cropped ROI mask to the full-resolution cropped TIFF shape.
    """
    mask = overview_mask > 0
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("ROI overview mask is empty.")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = mask[y0:y1, x0:x1].astype(np.uint8)

    target_h, target_w = target_shape
    resized = cv2.resize(
        cropped,
        dsize=(target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    resized = resized > 0

    return resized, {
        "overview_bounding_box_y_min": int(y0),
        "overview_bounding_box_y_max": int(y1),
        "overview_bounding_box_x_min": int(x0),
        "overview_bounding_box_x_max": int(x1),
        "overview_mask_height_pixels": int(overview_mask.shape[0]),
        "overview_mask_width_pixels":  int(overview_mask.shape[1]),
        "target_image_height_pixels":  int(target_shape[0]),
        "target_image_width_pixels":   int(target_shape[1]),
    }


def load_roi_mask_from_file(path, target_shape):
    """
    Load a binary ROI mask from a .tif or .roi (Fiji/ImageJ) file.

    Both formats are expected to be in full-image pixel coordinates
    (same dimensions as the segmentation output).

    - .tif : loaded directly; if shape matches target, used as-is.
             If shape differs (e.g. VSI overview mask), falls back to
             crop_and_resize_overview_roi_mask_to_image.
    - .roi : Fiji/ImageJ binary ROI — polygon rasterized to target_shape.
             Requires the `roifile` package (pip install roifile).

    Returns (mask_bool_array, roi_info_dict).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        raw = tifffile.imread(path)
        # Flatten to 2D (take first channel/slice if multi-dim)
        if raw.ndim > 2:
            raw = raw[0] if raw.shape[0] <= raw.shape[-1] else raw[..., 0]
        if raw.shape == tuple(target_shape):
            roi_info = {
                "target_image_height_pixels": int(target_shape[0]),
                "target_image_width_pixels":  int(target_shape[1]),
            }
            return raw > 0, roi_info
        # Shape mismatch — assume overview-space VSI mask, crop+resize
        return crop_and_resize_overview_roi_mask_to_image(raw, target_shape)

    if ext == ".roi":
        try:
            import roifile
        except ImportError:
            raise ImportError(
                "roifile is required to read .roi files. "
                "Install with: pip install roifile"
            )
        roi = roifile.ImagejRoi.fromfile(path)
        coords = roi.coordinates()  # (N, 2) array of [x, y] in image pixels
        if coords is None or len(coords) < 3:
            raise ValueError(f"ROI file has no usable polygon coordinates: {path}")
        # roifile returns (x, y) — convert to (row, col) = (y, x) for skimage
        rows_f = coords[:, 1]
        cols_f = coords[:, 0]
        rr, cc = sk_polygon(rows_f, cols_f, shape=target_shape)
        mask = np.zeros(target_shape, dtype=bool)
        mask[rr, cc] = True
        roi_info = {
            "target_image_height_pixels": int(target_shape[0]),
            "target_image_width_pixels":  int(target_shape[1]),
        }
        return mask, roi_info

    raise ValueError(f"Unsupported ROI mask format: {ext}")


# =============================================================================
# LABEL OBJECT TABLES
# =============================================================================

def get_label_props(label_img, raw_img=None, roi_mask=None, channel_name=""):
    """
    Return per-object table with area, centroid, mean intensity, and ROI membership.
    """
    label_img = label_img.astype(np.int64)
    labels = np.unique(label_img)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return pd.DataFrame(columns=[
            "label", "area", "centroid_y", "centroid_x",
            "mean_intensity", "inside_roi", "channel",
        ])

    if raw_img is not None:
        props = regionprops_table(
            label_img,
            intensity_image=raw_img,
            properties=("label", "area", "centroid", "mean_intensity"),
        )
    else:
        props = regionprops_table(label_img, properties=("label", "area", "centroid"))

    df = pd.DataFrame(props).rename(
        columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"}
    )
    if "mean_intensity" not in df.columns:
        df["mean_intensity"] = np.nan

    if roi_mask is not None:
        inside = []
        if ROI_CELL_RULE == "centroid":
            for _, row in df.iterrows():
                y = min(max(int(round(row["centroid_y"])), 0), roi_mask.shape[0] - 1)
                x = min(max(int(round(row["centroid_x"])), 0), roi_mask.shape[1] - 1)
                inside.append(bool(roi_mask[y, x]))
        elif ROI_CELL_RULE == "any_pixel":
            for label in df["label"].values:
                inside.append(bool(np.any((label_img == label) & roi_mask)))
        else:
            raise ValueError("ROI_CELL_RULE must be 'centroid' or 'any_pixel'.")
        df["inside_roi"] = inside
    else:
        df["inside_roi"] = False

    df["channel"] = channel_name
    return df


# =============================================================================
# VALIDATION TSV
# =============================================================================

def read_validation_tsv(project_folder, channel):
    """
    Read per-cell validation decisions for a specific channel.

    Looks for (in order):
        ROI Validation Results/roi_verification_results_ch{channel}.tsv  (new)
        ROI Validation Results/roi_verification_results.tsv              (legacy)

    Returns a dict with artifact_labels/yes_labels/no_labels/maybe_labels
    (all sets of int label IDs), or None if no TSV is found.
    """
    val_dir = os.path.join(project_folder, "ROI Validation Results")
    tsv_path = os.path.join(val_dir, f"roi_verification_results_ch{channel}.tsv")
    if not os.path.exists(tsv_path):
        legacy = os.path.join(val_dir, "roi_verification_results.tsv")
        if os.path.exists(legacy):
            tsv_path = legacy
        else:
            return None
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "decision" not in df.columns or "roi_id" not in df.columns:
            print(f"  [WARNING] Validation TSV missing expected columns: {list(df.columns)}")
            return None
        decisions = df["decision"].astype(str).str.upper().str.strip()
        labels    = df["roi_id"].astype(int)
        return {
            "label_decisions": dict(zip(labels, decisions)),
            "artifact_labels": set(labels[decisions == "ARTIFACT"]),
            "yes_labels":      set(labels[decisions == "YES"]),
            "no_labels":       set(labels[decisions == "NO"]),
            "maybe_labels":    set(labels[decisions == "MAYBE"]),
        }
    except Exception as exc:
        print(f"  [WARNING] Could not read validation TSV: {exc}")
        return None


def is_validation_complete(project_folder, channel):
    """True when the _validation_complete_ch{N} marker has been written."""
    val_dir = os.path.join(project_folder, "ROI Validation Results")
    if os.path.exists(os.path.join(val_dir, f"_validation_complete_ch{channel}")):
        return True
    # Legacy: single _validation_complete marker written by old validator versions
    if os.path.exists(os.path.join(val_dir, "_validation_complete")):
        return True
    return False


# =============================================================================
# IOU MATCHING
# =============================================================================

def compute_iou_pairs(source_labels, target_labels,
                      source_name="coi", target_name="pm"):
    """
    Compute candidate IoU pairs between two label images.
    Only evaluates pairs that actually overlap.
    """
    source = source_labels.astype(np.int64)
    target = target_labels.astype(np.int64)
    source_flat = source.ravel()
    target_flat = target.ravel()
    valid = (source_flat > 0) & (target_flat > 0)

    if not np.any(valid):
        return pd.DataFrame(columns=[
            f"{source_name}_label", f"{target_name}_label",
            "intersection",
            f"{source_name}_area", f"{target_name}_area",
            "union", "iou",
        ])

    pair_df = pd.DataFrame({
        f"{source_name}_label": source_flat[valid],
        f"{target_name}_label": target_flat[valid],
    })
    intersections = (
        pair_df
        .groupby([f"{source_name}_label", f"{target_name}_label"])
        .size()
        .reset_index(name="intersection")
    )
    source_areas = pd.Series(source_flat[source_flat > 0]).value_counts().to_dict()
    target_areas = pd.Series(target_flat[target_flat > 0]).value_counts().to_dict()
    intersections[f"{source_name}_area"] = intersections[f"{source_name}_label"].map(source_areas)
    intersections[f"{target_name}_area"] = intersections[f"{target_name}_label"].map(target_areas)
    intersections["union"] = (
        intersections[f"{source_name}_area"]
        + intersections[f"{target_name}_area"]
        - intersections["intersection"]
    )
    intersections["iou"] = intersections["intersection"] / intersections["union"]
    return intersections


def greedy_one_to_one_iou_matching(iou_df, source_col, target_col, iou_threshold):
    """
    Greedy one-to-one matching sorted by IoU descending.
    Each source and target label can be matched at most once.
    """
    if iou_df.empty:
        return pd.DataFrame(columns=list(iou_df.columns) + ["match_rank"])

    candidates = iou_df[iou_df["iou"] >= iou_threshold].copy()
    candidates = candidates.sort_values("iou", ascending=False)

    used_source = set()
    used_target = set()
    rows = []
    rank = 1
    for _, row in candidates.iterrows():
        s = int(row[source_col])
        t = int(row[target_col])
        if s in used_source or t in used_target:
            continue
        used_source.add(s)
        used_target.add(t)
        out = row.to_dict()
        out["match_rank"] = rank
        rows.append(out)
        rank += 1

    return pd.DataFrame(rows)


def run_directional_matching(coi_labels, pm_labels, coi_name="coi", pm_name="pm"):
    """
    Returns:
    - coi_to_pm_matches, pm_to_coi_matches
    - coi_to_pm_iou, pm_to_coi_iou
    """
    coi_to_pm_iou = compute_iou_pairs(coi_labels, pm_labels, coi_name, pm_name)
    coi_to_pm_matches = greedy_one_to_one_iou_matching(
        coi_to_pm_iou, f"{coi_name}_label", f"{pm_name}_label", IOU_THRESHOLD
    )

    pm_to_coi_iou = compute_iou_pairs(pm_labels, coi_labels, pm_name, coi_name)
    pm_to_coi_matches = greedy_one_to_one_iou_matching(
        pm_to_coi_iou, f"{pm_name}_label", f"{coi_name}_label", IOU_THRESHOLD
    )

    return coi_to_pm_matches, pm_to_coi_matches, coi_to_pm_iou, pm_to_coi_iou


# =============================================================================
# QC IMAGE OUTPUT
# =============================================================================

def thicken_boundary(binary_boundary, thickness_px=3):
    if thickness_px <= 1:
        return binary_boundary
    kernel = np.ones((thickness_px, thickness_px), dtype=np.uint8)
    thick = cv2.dilate(binary_boundary.astype(np.uint8), kernel, iterations=1)
    return thick > 0


def _save_qc_png_with_legend(rgb, png_path, coi_ch=None, pm_ch=None,
                             has_roi=False):
    """Save the QC overlay as a PNG with a colour legend using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for subprocess
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        coi_label = f"Channel {coi_ch}" if coi_ch is not None else "COI channel"
        pm_label  = f"Channel {pm_ch}"  if pm_ch  is not None else "PM channel"

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(rgb)
        ax.axis("off")

        patches = [
            mpatches.Patch(color=(1.0, 0.0, 0.0),
                           label=f"Red — {coi_label} (cell of interest)"),
            mpatches.Patch(color=(0.0, 1.0, 1.0),
                           label=f"Cyan — {pm_label} (protein marker)"),
            mpatches.Patch(color=(0.0, 1.0, 0.0),
                           label=f"Green — colocalized {coi_label} & {pm_label}"),
        ]
        if has_roi:
            patches.append(mpatches.Patch(color=(1.0, 1.0, 0.0),
                                          label="Yellow — ROI boundary"))

        legend = ax.legend(
            handles=patches,
            loc="lower right",
            fontsize=9,
            framealpha=0.75,
            facecolor="#111111",
            labelcolor="white",
            edgecolor="none",
        )
        fig.tight_layout(pad=0.2)
        fig.savefig(png_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  QC PNG saved: {os.path.basename(png_path)}")
    except Exception as exc:
        print(f"  [WARNING] Could not save QC PNG with legend: {exc}")


def make_qc_overlay(raw_coi, raw_pm, coi_labels, pm_labels,
                    roi_mask, matched_coi_labels, matched_pm_labels,
                    output_path, coi_ch=None, pm_ch=None):
    """
    RGB QC overlay:
    - grayscale base = max-projection blend of COI / PM raw images
    - COI object boundaries = red
    - PM object boundaries = cyan
    - matched COI/PM object boundaries = green
    - ROI boundary = thick yellow

    Saves as both TIFF (lossless) and PNG (with colour legend).
    """
    coi_u8 = robust_uint8(raw_coi)
    pm_u8  = robust_uint8(raw_pm)
    base   = np.maximum(coi_u8, pm_u8)
    rgb    = np.stack([base, base, base], axis=-1)

    coi_boundary = thicken_boundary(find_boundaries(coi_labels > 0, mode="outer"),
                                    OBJECT_BORDER_THICKNESS_PX)
    rgb[coi_boundary, 0] = 255
    rgb[coi_boundary, 1] = 0
    rgb[coi_boundary, 2] = 0

    pm_boundary = thicken_boundary(find_boundaries(pm_labels > 0, mode="outer"),
                                   OBJECT_BORDER_THICKNESS_PX)
    rgb[pm_boundary, 0] = 0
    rgb[pm_boundary, 1] = 255
    rgb[pm_boundary, 2] = 255

    matched_mask = (
        np.isin(coi_labels, list(matched_coi_labels)) |
        np.isin(pm_labels,  list(matched_pm_labels))
    )
    matched_boundary = thicken_boundary(find_boundaries(matched_mask, mode="outer"),
                                        MATCHED_BORDER_THICKNESS_PX)
    rgb[matched_boundary, 0] = 0
    rgb[matched_boundary, 1] = 255
    rgb[matched_boundary, 2] = 0

    if roi_mask is not None:
        roi_boundary = thicken_boundary(find_boundaries(roi_mask, mode="outer"),
                                        ROI_BORDER_THICKNESS_PX)
        rgb[roi_boundary, 0] = 255
        rgb[roi_boundary, 1] = 255
        rgb[roi_boundary, 2] = 0

    # Save lossless TIFF
    tifffile.imwrite(output_path, rgb)

    # Save PNG with colour legend
    png_path = os.path.splitext(output_path)[0] + ".png"
    _save_qc_png_with_legend(rgb, png_path, coi_ch=coi_ch, pm_ch=pm_ch,
                             has_roi=(roi_mask is not None))


def make_roi_mask_preview(raw_img, roi_mask, output_path):
    img_u8 = robust_uint8(raw_img)
    rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    if roi_mask is not None:
        b = thicken_boundary(find_boundaries(roi_mask, mode="outer"),
                             ROI_BORDER_THICKNESS_PX)
        rgb[b, 0] = 255
        rgb[b, 1] = 255
        rgb[b, 2] = 0
    tifffile.imwrite(output_path, rgb)


# =============================================================================
# PROJECT ANALYSIS
# =============================================================================

def analyze_project_folder(project_folder):
    project_name = os.path.basename(project_folder)
    print(f"\n--- Analyzing: {project_name} ---")

    analysis_folder = os.path.join(project_folder, ANALYSIS_SUBFOLDER_NAME)
    ensure_dir(analysis_folder)

    # ── Find segmentation label files ───────────────────────────────────────
    coi_label_path = find_one_file(
        project_folder, PRED_TIFF_PATTERN.format(ch=COI_CHANNEL), required=True
    )
    pm_label_path = find_one_file(
        project_folder, PRED_TIFF_PATTERN.format(ch=PM_CHANNEL), required=True
    )
    coi_raw_path = find_one_file(
        project_folder, SPLIT_TIFF_PATTERN.format(ch=COI_CHANNEL), required=True
    )
    pm_raw_path = find_one_file(
        project_folder, SPLIT_TIFF_PATTERN.format(ch=PM_CHANNEL), required=True
    )

    print(f"  COI labels: {os.path.basename(coi_label_path)}")
    print(f"  PM  labels: {os.path.basename(pm_label_path)}")

    # ── Load images ─────────────────────────────────────────────────────────
    coi_labels = tifffile.imread(coi_label_path)
    pm_labels  = tifffile.imread(pm_label_path)
    coi_raw    = tifffile.imread(coi_raw_path)
    pm_raw     = tifffile.imread(pm_raw_path)

    if coi_labels.shape != pm_labels.shape:
        raise ValueError(
            f"COI/PM label shapes do not match: {coi_labels.shape} vs {pm_labels.shape}"
        )

    # ── ROI mask (optional) ─────────────────────────────────────────────────
    roi_mask = None
    roi_info = {}
    roi_mask_path = find_roi_mask_for_project(project_name, ROI_MASK_FOLDER)
    if roi_mask_path is not None:
        print(f"  ROI mask:   {os.path.basename(roi_mask_path)}")
        roi_mask, roi_info = load_roi_mask_from_file(roi_mask_path, coi_labels.shape)
    else:
        print("  ROI mask:   (none — ROI counts will be empty)")

    # ── Channel name tokens (used throughout) ────────────────────────────────
    coi_ch = f"channel_{COI_CHANNEL}"
    pm_ch  = f"channel_{PM_CHANNEL}"

    # ── Validation — per-channel artifact filtering ───────────────────────────
    # Complete validation (marker present): always applied.
    # Partial validation (TSV exists, no marker): applied only when
    # APPLY_PARTIAL_VALIDATION is True.
    def _apply_val(label_img, channel, ch_token):
        val_data = read_validation_tsv(project_folder, channel)
        if val_data is None:
            print(f"  {ch_token}: no validation TSV — all detected cells included")
            return label_img, None, "none"

        complete = is_validation_complete(project_folder, channel)
        if not complete and not APPLY_PARTIAL_VALIDATION:
            n_reviewed = len(val_data["label_decisions"])
            print(f"  {ch_token}: [PARTIAL ~ {n_reviewed} reviewed] "
                  "filter NOT applied — enable 'apply_partial_validation' to override")
            return label_img, val_data, "partial_skipped"

        artifacts = val_data["artifact_labels"]
        if artifacts:
            label_img = label_img.copy()
            label_img[np.isin(label_img, list(artifacts))] = 0

        status = "complete" if complete else "partial_applied"
        tag = "[COMPLETE ✓]" if complete else "[PARTIAL ~ applied]"
        print(f"  {ch_token}: {tag} "
              f"{len(val_data['yes_labels'])} YES  "
              f"{len(val_data['no_labels'])} NO  "
              f"{len(val_data['maybe_labels'])} MAYBE  "
              f"{len(artifacts)} ARTIFACT removed")
        return label_img, val_data, status

    coi_labels, coi_val_data, coi_val_status = _apply_val(coi_labels, COI_CHANNEL, coi_ch)
    pm_labels,  pm_val_data,  pm_val_status  = _apply_val(pm_labels,  PM_CHANNEL,  pm_ch)

    # ── Per-object tables ────────────────────────────────────────────────────
    coi_props = get_label_props(
        coi_labels, raw_img=coi_raw, roi_mask=roi_mask,
        channel_name=f"cell_of_interest_CH{COI_CHANNEL}"
    )
    pm_props = get_label_props(
        pm_labels, raw_img=pm_raw, roi_mask=roi_mask,
        channel_name=f"protein_marker_CH{PM_CHANNEL}"
    )

    coi_to_pm_matches, pm_to_coi_matches, coi_to_pm_iou, pm_to_coi_iou = (
        run_directional_matching(coi_labels, pm_labels, coi_ch, pm_ch)
    )

    # Tag matches with ROI membership
    coi_roi_map = dict(zip(coi_props["label"].astype(int), coi_props["inside_roi"].astype(bool)))
    pm_roi_map  = dict(zip(pm_props["label"].astype(int),  pm_props["inside_roi"].astype(bool)))

    for df, coi_col, pm_col in [
        (coi_to_pm_matches, f"{coi_ch}_label", f"{pm_ch}_label"),
        (pm_to_coi_matches, f"{coi_ch}_label", f"{pm_ch}_label"),
    ]:
        if not df.empty:
            df[f"{coi_ch}_inside_roi"] = df[coi_col].astype(int).map(coi_roi_map)
            df[f"{pm_ch}_inside_roi"]  = df[pm_col].astype(int).map(pm_roi_map)
        else:
            df[f"{coi_ch}_inside_roi"] = []
            df[f"{pm_ch}_inside_roi"]  = []

    # ── Counts ───────────────────────────────────────────────────────────────
    total_coi = int(len(coi_props))
    total_pm  = int(len(pm_props))

    roi_coi = int(coi_props["inside_roi"].sum()) if (roi_mask is not None and len(coi_props)) else None
    roi_pm  = int(pm_props["inside_roi"].sum())  if (roi_mask is not None and len(pm_props))  else None

    # Average stats (all objects)
    coi_avg_area  = float(coi_props["area"].mean())            if total_coi else np.nan
    pm_avg_area   = float(pm_props["area"].mean())             if total_pm  else np.nan
    coi_avg_inten = float(coi_props["mean_intensity"].mean())  if total_coi else np.nan
    pm_avg_inten  = float(pm_props["mean_intensity"].mean())   if total_pm  else np.nan

    # Average stats (inside ROI only)
    coi_in_roi_df = coi_props[coi_props["inside_roi"]] if roi_mask is not None else pd.DataFrame()
    pm_in_roi_df  = pm_props[pm_props["inside_roi"]]   if roi_mask is not None else pd.DataFrame()
    coi_roi_avg_area  = float(coi_in_roi_df["area"].mean())           if len(coi_in_roi_df) else np.nan
    pm_roi_avg_area   = float(pm_in_roi_df["area"].mean())            if len(pm_in_roi_df)  else np.nan
    coi_roi_avg_inten = float(coi_in_roi_df["mean_intensity"].mean()) if len(coi_in_roi_df) else np.nan
    pm_roi_avg_inten  = float(pm_in_roi_df["mean_intensity"].mean())  if len(pm_in_roi_df)  else np.nan

    # Colocalization counts
    coi_pm_coloc_all = int(len(coi_to_pm_matches))
    pm_coi_coloc_all = int(len(pm_to_coi_matches))
    coi_pm_coloc_roi = (
        int(coi_to_pm_matches[f"{coi_ch}_inside_roi"].sum())
        if (roi_mask is not None and not coi_to_pm_matches.empty) else None
    )
    pm_coi_coloc_roi = (
        int(pm_to_coi_matches[f"{pm_ch}_inside_roi"].sum())
        if (roi_mask is not None and not pm_to_coi_matches.empty) else None
    )

    # ── Summary row ─────────────────────────────────────────────────────────
    summary = {
        "project_name":                                 project_name,
        "cell_of_interest_channel":                     COI_CHANNEL,
        "protein_marker_channel":                       PM_CHANNEL,
        "iou_threshold":                                IOU_THRESHOLD,
        "roi_cell_counting_rule":                       ROI_CELL_RULE,
        # All-image counts
        f"{coi_ch}_total_count":                        total_coi,
        f"{pm_ch}_total_count":                         total_pm,
        f"{coi_ch}_average_area_pixels":                round(coi_avg_area, 1),
        f"{pm_ch}_average_area_pixels":                 round(pm_avg_area, 1),
        f"{coi_ch}_average_intensity":                  round(coi_avg_inten, 2),
        f"{pm_ch}_average_intensity":                   round(pm_avg_inten, 2),
        # ROI counts (None when no ROI mask)
        f"{coi_ch}_roi_count":                          roi_coi,
        f"{pm_ch}_roi_count":                           roi_pm,
        f"{coi_ch}_roi_average_area_pixels":            round(coi_roi_avg_area, 1),
        f"{pm_ch}_roi_average_area_pixels":             round(pm_roi_avg_area, 1),
        f"{coi_ch}_roi_average_intensity":              round(coi_roi_avg_inten, 2),
        f"{pm_ch}_roi_average_intensity":               round(pm_roi_avg_inten, 2),
        # Colocalization
        f"{coi_ch}_colocalized_whole_image":            coi_pm_coloc_all,
        f"{pm_ch}_colocalized_whole_image":             pm_coi_coloc_all,
        f"{coi_ch}_colocalized_in_roi":                 coi_pm_coloc_roi,
        f"{pm_ch}_colocalized_in_roi":                  pm_coi_coloc_roi,
        f"{coi_ch}_percent_colocalized_whole_image":    round(safe_percent(coi_pm_coloc_all, total_coi), 4),
        f"{pm_ch}_percent_colocalized_whole_image":     round(safe_percent(pm_coi_coloc_all, total_pm), 4),
        f"{coi_ch}_percent_colocalized_in_roi":         round(safe_percent(coi_pm_coloc_roi or 0, roi_coi or 0), 4),
        f"{pm_ch}_percent_colocalized_in_roi":          round(safe_percent(pm_coi_coloc_roi or 0, roi_pm or 0), 4),
        # File references
        f"{coi_ch}_label_file":                         os.path.basename(coi_label_path),
        f"{pm_ch}_label_file":                          os.path.basename(pm_label_path),
        "roi_mask_file":                                os.path.basename(roi_mask_path) if roi_mask_path else "",
    }

    # ── Validation summary columns ───────────────────────────────────────────
    have_roi = roi_mask is not None

    def _add_val_columns(ch_token, val_status, val_data, props):
        summary[f"{ch_token}_validation_status"] = val_status
        if val_status in ("complete", "partial_applied") and val_data is not None:
            summary[f"{ch_token}_artifacts_removed"] = len(val_data["artifact_labels"])
            label_to_roi = dict(
                zip(props["label"].astype(int), props["inside_roi"].astype(bool))
            ) if "inside_roi" in props.columns else {}

            def _val_count(label_set, in_roi=None):
                n = 0
                for lbl in label_set:
                    if lbl in label_to_roi:
                        if in_roi is None or label_to_roi[lbl] == in_roi:
                            n += 1
                return n

            summary[f"{ch_token}_validation_yes_whole_image"]   = _val_count(val_data["yes_labels"])
            summary[f"{ch_token}_validation_no_whole_image"]     = _val_count(val_data["no_labels"])
            summary[f"{ch_token}_validation_maybe_whole_image"]  = _val_count(val_data["maybe_labels"])
            summary[f"{ch_token}_validation_yes_in_roi"]         = _val_count(val_data["yes_labels"],   in_roi=True) if have_roi else None
            summary[f"{ch_token}_validation_no_in_roi"]          = _val_count(val_data["no_labels"],    in_roi=True) if have_roi else None
            summary[f"{ch_token}_validation_maybe_in_roi"]       = _val_count(val_data["maybe_labels"], in_roi=True) if have_roi else None
        else:
            summary[f"{ch_token}_artifacts_removed"]             = 0
            summary[f"{ch_token}_validation_yes_whole_image"]    = None
            summary[f"{ch_token}_validation_no_whole_image"]     = None
            summary[f"{ch_token}_validation_maybe_whole_image"]  = None
            summary[f"{ch_token}_validation_yes_in_roi"]         = None
            summary[f"{ch_token}_validation_no_in_roi"]          = None
            summary[f"{ch_token}_validation_maybe_in_roi"]       = None

    _add_val_columns(coi_ch, coi_val_status, coi_val_data, coi_props)
    _add_val_columns(pm_ch,  pm_val_status,  pm_val_data,  pm_props)

    if roi_info:
        summary.update(roi_info)

    summary_df = pd.DataFrame([summary])

    # ── Save per-project CSVs ────────────────────────────────────────────────
    p = project_name
    summary_df.to_csv(os.path.join(analysis_folder, f"{p}_summary.csv"), index=False)
    coi_props.to_csv(os.path.join(analysis_folder, f"{p}_{coi_ch}_object_table.csv"), index=False)
    pm_props.to_csv(os.path.join(analysis_folder,  f"{p}_{pm_ch}_object_table.csv"),  index=False)
    coi_to_pm_matches.to_csv(
        os.path.join(analysis_folder, f"{p}_{coi_ch}_to_{pm_ch}_matches.csv"), index=False)
    pm_to_coi_matches.to_csv(
        os.path.join(analysis_folder, f"{p}_{pm_ch}_to_{coi_ch}_matches.csv"), index=False)
    coi_to_pm_iou.to_csv(
        os.path.join(analysis_folder, f"{p}_{coi_ch}_to_{pm_ch}_iou_candidates.csv"), index=False)
    pm_to_coi_iou.to_csv(
        os.path.join(analysis_folder, f"{p}_{pm_ch}_to_{coi_ch}_iou_candidates.csv"), index=False)

    if roi_mask is not None:
        tifffile.imwrite(
            os.path.join(analysis_folder, f"{p}_roi_mask_resized.tif"),
            roi_mask.astype(np.uint8) * 255,
        )

    # Build matched label sets for QC overlay
    matched_coi_set: set = set()
    matched_pm_set:  set = set()
    if not coi_to_pm_matches.empty:
        matched_coi_set = set(coi_to_pm_matches[f"{coi_ch}_label"].astype(int).tolist())
        matched_pm_set  = set(coi_to_pm_matches[f"{pm_ch}_label"].astype(int).tolist())

    if SAVE_QC_PNG:
        make_qc_overlay(
            raw_coi=coi_raw, raw_pm=pm_raw,
            coi_labels=coi_labels, pm_labels=pm_labels,
            roi_mask=roi_mask,
            matched_coi_labels=matched_coi_set,
            matched_pm_labels=matched_pm_set,
            output_path=os.path.join(analysis_folder, f"{p}_QC_overlay.tif"),
            coi_ch=COI_CHANNEL, pm_ch=PM_CHANNEL,
        )
        if roi_mask is not None:
            make_roi_mask_preview(
                raw_img=coi_raw, roi_mask=roi_mask,
                output_path=os.path.join(analysis_folder, f"{p}_QC_roi_preview.tif"),
            )

    print(f"  COI total: {total_coi}  |  PM total: {total_pm}")
    print(f"  COI↔PM coloc all: {coi_pm_coloc_all}/{total_coi}  |  PM↔COI: {pm_coi_coloc_all}/{total_pm}")
    if roi_mask is not None:
        print(f"  COI in ROI: {roi_coi}  |  PM in ROI: {roi_pm}")
        print(f"  COI↔PM coloc ROI: {coi_pm_coloc_roi}/{roi_coi}  |  PM↔COI ROI: {pm_coi_coloc_roi}/{roi_pm}")

    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    project_folders = get_project_folders(MICROSAM_OUTPUT_ROOT)
    if not project_folders:
        print(f"[ERROR] No project folders found in: {MICROSAM_OUTPUT_ROOT}")
        return

    print(f"Found {len(project_folders)} project folder(s).")
    all_summaries = []

    for project_folder in tqdm(project_folders, desc="Analyzing projects"):
        try:
            all_summaries.append(analyze_project_folder(project_folder))
        except Exception as exc:
            project_name = os.path.basename(project_folder)
            print(f"\n[ERROR] Failed project: {project_name}\n{exc}")
            all_summaries.append(pd.DataFrame([{"project_name": project_name, "error": str(exc)}]))

    combined = pd.concat(all_summaries, ignore_index=True)

    # Combined outputs go to ANALYSIS_OUTPUT_ROOT (the Analysis folder),
    # not MICROSAM_OUTPUT_ROOT (the Segmentation folder).
    out_dir = ANALYSIS_OUTPUT_ROOT or MICROSAM_OUTPUT_ROOT
    os.makedirs(out_dir, exist_ok=True)

    try:
        excel_path = os.path.join(out_dir, "combined_analysis_summary.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="Summary", index=False)

            # Format the colocalization-percentage columns as real Excel
            # percentages: stored as fractions (0.3221) so they show as 32.21%
            # and copy/paste as 0.3221 rather than a bare 32.21 integer.
            ws = writer.sheets["Summary"]
            pct_cols = [
                i + 1
                for i, col in enumerate(combined.columns)
                if "_percent_colocalized_" in str(col)
            ]
            for col_idx in pct_cols:
                letter = ws.cell(row=1, column=col_idx).column_letter
                for row_idx in range(2, ws.max_row + 1):
                    ws[f"{letter}{row_idx}"].number_format = "0.00%"
        print(f"\nExcel summary: {excel_path}")
    except ImportError:
        print("[WARNING] openpyxl not installed — Excel output skipped. Run: pip install openpyxl")
    except Exception as exc:
        print(f"[WARNING] Could not save Excel: {exc}")

    print("\nDONE.")


def _apply_config(config_path: str) -> None:
    """Override module-level settings from a CellMatchr config file."""
    import json
    global MICROSAM_OUTPUT_ROOT, ANALYSIS_OUTPUT_ROOT, ROI_MASK_FOLDER
    global COI_CHANNEL, PM_CHANNEL, IOU_THRESHOLD
    global ROI_CELL_RULE, ANALYSIS_SUBFOLDER_NAME, SAVE_QC_PNG

    with open(config_path) as f:
        cfg = json.load(f)

    # MICROSAM_OUTPUT_ROOT = segmentation folder (project sub-dirs live here)
    seg_root = cfg.get("ana_microsam_output_root", "").strip()
    if seg_root:
        MICROSAM_OUTPUT_ROOT = seg_root

    # ANALYSIS_OUTPUT_ROOT = Analysis folder (combined CSV/Excel go here)
    ana_root = cfg.get("output_root_folder", "").strip()
    if ana_root:
        ANALYSIS_OUTPUT_ROOT = ana_root
    elif not seg_root:
        # fallback: old configs that only set output_root_folder
        fallback = cfg.get("output_root_folder", "").strip()
        if fallback:
            MICROSAM_OUTPUT_ROOT = fallback

    roi_folder = cfg.get("ana_roi_mask_folder", "").strip()
    ROI_MASK_FOLDER = roi_folder  # may be empty — that's OK, ROI analysis will be skipped

    try:
        COI_CHANNEL = int(cfg.get("ana_red_channel", COI_CHANNEL))
    except (ValueError, TypeError):
        pass
    try:
        PM_CHANNEL = int(cfg.get("ana_pv_channel", PM_CHANNEL))
    except (ValueError, TypeError):
        pass
    try:
        IOU_THRESHOLD = float(cfg.get("ana_iou_threshold", IOU_THRESHOLD))
    except (ValueError, TypeError):
        pass

    if cfg.get("ana_roi_cell_rule") in ("centroid", "any_pixel"):
        ROI_CELL_RULE = cfg["ana_roi_cell_rule"]
    if cfg.get("ana_subfolder_name", "").strip():
        ANALYSIS_SUBFOLDER_NAME = cfg["ana_subfolder_name"].strip()
    if cfg.get("ana_save_qc") is not None:
        SAVE_QC_PNG = bool(cfg["ana_save_qc"])

    global APPLY_PARTIAL_VALIDATION
    if cfg.get("apply_partial_validation") is not None:
        APPLY_PARTIAL_VALIDATION = bool(cfg["apply_partial_validation"])


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="CellMatchr Analysis")
    _parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.cellmatchr/config.json"),
        help="Path to CellMatchr config JSON",
    )
    _args, _ = _parser.parse_known_args()
    if os.path.exists(_args.config):
        _apply_config(_args.config)
    main()
