import os
import sys
import torch
from czifile import CziFile
import numpy as np
import tifffile
from tifffile import imwrite
from tqdm import tqdm
import cv2

import micro_sam
import micro_sam.automatic_segmentation
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT_INPUT_FOLDER = "/run/user/1000/gvfs/smb-share:server=rstore.it.tufts.edu,share=tusm_bygravelab01$/Confocal data archive/Matt/20260413 RO ENS Trim 2Male"
MAIN_OUTPUT_FOLDER = "/run/user/1000/gvfs/smb-share:server=rstore.it.tufts.edu,share=tusm_bygravelab01$/Confocal data archive/Matt/Micro_SAM/ENS_Trim_BG_Deletion_2"

DRY_RUN = False
TEST_RUN = False
SKIP_SUBDIRS = False

# If TEST_RUN = True, only the first file is processed.
# Debug PNGs are saved only when TEST_RUN=True and SAVE_DEBUG_IMAGES=True.

# For slide scanner images:
# 0: DAPI-405, 1: Red-567, 2: Green-488, 3: FarRed-647
CHANNELS_TO_PREDICT = [2]

NDIM = 2

# IMPORTANT:
# Always use custom tiled micro-SAM prediction.
# Do not use full-image automatic_instance_segmentation.
ALWAYS_USE_TILING = True

# Your working settings
TILE_SHAPE_2D = (2048, 2048)
HALO_2D = (256, 256)


# =============================================================================
# MICRO-SAM SETTINGS
# =============================================================================

MODEL_TYPE = "vit_b_lm"

# Options:
# "raw_percentile"
# "local_bg_subtract"
# "local_bg_subtract_mild"
# "tile_clahe_mild"
# "tile_clahe"
PREPROCESS_MODE = "local_bg_subtract"

# This is passed into segmenter.generate()
MIN_OBJECT_SIZE = 1200

# Post-filtering settings
MAX_OBJECT_SIZE = 30000
MAX_MAJOR_MINOR_RATIO = 15.0

# Mean intensity filter only
RAW_MEAN_MIN = 5000

# Per-channel filter overrides (populated from config)
PER_CHANNEL_FILTERS_ENABLED = False
PER_CHANNEL_FILTERS = {}   # {channel_int: {"min_size", "max_size", "max_ratio", "min_intensity"}}

# Skip basically empty tiles
SKIP_EMPTY_TILES = True
BRIGHT_PIXEL_THRESHOLD = 4000
MIN_TILE_RAW_MAX = 5000
MIN_TILE_BRIGHT_PIXEL_COUNT = 20

# Debug output
SAVE_DEBUG_IMAGES = True
MAX_DEBUG_TILES = 25
DEBUG_TILE_EVERY_N = 1

# Final QC outputs. These are saved after every prediction, even when TEST_RUN=False.
SAVE_FINAL_QC_OVERLAY = True
FINAL_QC_DOWNSAMPLE = 4  # 4 = detailed; 8 or 10 = smaller/faster


# # This is example setting for using it with slide scanner data on 20x lens
# MIN_OBJECT_SIZE = 200

# # Post-filtering settings
# MAX_OBJECT_SIZE = 5000
# MAX_MAJOR_MINOR_RATIO = 15.0

# # Mean intensity filter only
# RAW_MEAN_MIN = 1000

# # Skip basically empty tiles
# SKIP_EMPTY_TILES = True
# BRIGHT_PIXEL_THRESHOLD = 4000
# MIN_TILE_RAW_MAX = 5000
# MIN_TILE_BRIGHT_PIXEL_COUNT = 20

# # Debug output
# SAVE_DEBUG_IMAGES = True
# MAX_DEBUG_TILES = 25
# DEBUG_TILE_EVERY_N = 1


# =============================================================================
# PREPROCESSING
# =============================================================================

def robust_percentile_normalize_to_uint8(image, p_low=0.5, p_high=99.8):
    img = image.astype(np.float32)

    low, high = np.percentile(img, [p_low, p_high])

    if high <= low:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    img = (img * 255).astype(np.uint8)

    return img


def apply_tile_clahe_uint8(tile_uint8, clip_limit=1.5, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size
    )
    return clahe.apply(tile_uint8)


def preprocess_tile_for_sam(tile_data, mode="raw_percentile"):
    if mode == "raw_percentile":
        tile_uint8 = robust_percentile_normalize_to_uint8(
            tile_data,
            p_low=0.5,
            p_high=99.8
        )

    elif mode == "local_bg_subtract":
        img = tile_data.astype(np.float32)

        background = cv2.GaussianBlur(
            img,
            ksize=(0, 0),
            sigmaX=35,
            sigmaY=35
        )

        corrected = img - background
        corrected[corrected < 0] = 0

        tile_uint8 = robust_percentile_normalize_to_uint8(
            corrected,
            p_low=0.3,
            p_high=99.9
        )

    elif mode == "local_bg_subtract_mild":
        img = tile_data.astype(np.float32)

        background = cv2.GaussianBlur(
            img,
            ksize=(0, 0),
            sigmaX=55,
            sigmaY=55
        )

        corrected = img - background
        corrected[corrected < 0] = 0

        tile_uint8 = robust_percentile_normalize_to_uint8(
            corrected,
            p_low=0.3,
            p_high=99.9
        )

    elif mode == "tile_clahe_mild":
        tile_uint8 = robust_percentile_normalize_to_uint8(
            tile_data,
            p_low=0.5,
            p_high=99.8
        )

        tile_uint8 = apply_tile_clahe_uint8(
            tile_uint8,
            clip_limit=1.2,
            grid_size=(8, 8)
        )

    elif mode == "tile_clahe":
        tile_uint8 = robust_percentile_normalize_to_uint8(
            tile_data,
            p_low=0.5,
            p_high=99.8
        )

        tile_uint8 = apply_tile_clahe_uint8(
            tile_uint8,
            clip_limit=2.0,
            grid_size=(8, 8)
        )

    else:
        raise ValueError(f"Unknown PREPROCESS_MODE: {mode}")

    tile_rgb = np.stack((tile_uint8,) * 3, axis=-1)

    return tile_uint8, tile_rgb


# =============================================================================
# TILE SELECTION
# =============================================================================

def tile_has_enough_signal(tile_data):
    tile_max = float(tile_data.max())
    bright_pixel_count = int(np.sum(tile_data > BRIGHT_PIXEL_THRESHOLD))

    if tile_max < MIN_TILE_RAW_MAX:
        return False

    if bright_pixel_count < MIN_TILE_BRIGHT_PIXEL_COUNT:
        return False

    return True


# =============================================================================
# DEBUG OUTPUT
# =============================================================================

def should_save_debug_images():
    return TEST_RUN and SAVE_DEBUG_IMAGES


def save_debug_pngs(
    debug_folder,
    tile_index,
    corner_coord,
    raw_tile,
    processed_uint8,
    labeled_tile,
    segmentation_results
):
    os.makedirs(debug_folder, exist_ok=True)

    y, x = corner_coord
    prefix = f"tile_{tile_index:04d}_y{y}_x{x}"

    raw_preview = robust_percentile_normalize_to_uint8(
        raw_tile,
        p_low=0.5,
        p_high=99.8
    )

    raw_path = os.path.join(debug_folder, f"{prefix}_01_raw_percentile.png")
    processed_path = os.path.join(debug_folder, f"{prefix}_02_processed_sent_to_sam.png")
    mask_path = os.path.join(debug_folder, f"{prefix}_03_sam_binary_mask.png")
    overlay_path = os.path.join(debug_folder, f"{prefix}_04_overlay_boundaries.png")
    csv_path = os.path.join(debug_folder, f"{prefix}_05_raw_sam_masks.csv")

    tifffile.imwrite(raw_path, raw_preview)
    tifffile.imwrite(processed_path, processed_uint8)

    mask_preview = np.zeros_like(processed_uint8, dtype=np.uint8)

    if labeled_tile.max() > 0:
        mask_preview = ((labeled_tile > 0) * 255).astype(np.uint8)

    tifffile.imwrite(mask_path, mask_preview)

    boundaries = find_boundaries(labeled_tile > 0, mode="outer")

    overlay = np.stack(
        [processed_uint8, processed_uint8, processed_uint8],
        axis=-1
    )

    overlay[boundaries, 0] = 255
    overlay[boundaries, 1] = 0
    overlay[boundaries, 2] = 0

    tifffile.imwrite(overlay_path, overlay)

    rows = []

    for i, mask_dict in enumerate(segmentation_results, 1):
        mask = mask_dict["segmentation"]

        rows.append({
            "raw_sam_mask_index": i,
            "area": int(mask.sum()),
            "predicted_iou": mask_dict.get("predicted_iou", None),
            "stability_score": mask_dict.get("stability_score", None),
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"      Saved debug PNGs for tile {tile_index}: {debug_folder}")


# =============================================================================
# FILTERING SAM MASKS
# =============================================================================

def filter_labeled_tile_by_raw_mean_and_shape(labeled_tile, raw_tile):
    """
    Post-filter SAM masks using:
    - area
    - raw mean intensity
    - shape

    No max-intensity filter.
    """
    if labeled_tile.max() == 0:
        return labeled_tile.astype(np.uint32)

    props = regionprops_table(
        labeled_tile,
        intensity_image=raw_tile,
        properties=(
            "label",
            "area",
            "mean_intensity",
            "major_axis_length",
            "minor_axis_length",
        )
    )

    props_df = pd.DataFrame(props)

    valid_labels = []

    for _, row in props_df.iterrows():
        label = int(row["label"])
        area = float(row["area"])
        mean_intensity = float(row["mean_intensity"])

        major = float(row["major_axis_length"])
        minor = max(float(row["minor_axis_length"]), 1e-6)
        ratio = major / minor

        if MIN_OBJECT_SIZE > 0 and area < MIN_OBJECT_SIZE:
            continue

        if MAX_OBJECT_SIZE > 0 and area > MAX_OBJECT_SIZE:
            continue

        if RAW_MEAN_MIN > 0 and mean_intensity < RAW_MEAN_MIN:
            continue

        if MAX_MAJOR_MINOR_RATIO > 0 and ratio > MAX_MAJOR_MINOR_RATIO:
            continue

        valid_labels.append(label)

    filtered = np.where(np.isin(labeled_tile, valid_labels), labeled_tile, 0)

    return filtered.astype(np.uint32)


# =============================================================================
# SMART TILING
# =============================================================================

def make_smart_tile_starts(image_size, tile_size, step_size):
    """
    Cover the full image without creating tiny edge tiles.
    Final tile is anchored to image_size - tile_size.
    """
    if image_size <= tile_size:
        return [0]

    starts = list(range(0, image_size - tile_size + 1, step_size))

    final_start = image_size - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    starts = sorted(set(starts))

    return starts


def make_tile_coordinates(image_shape, tile_shape, halo):
    step_y = max(1, tile_shape[0] - 2 * halo[0])
    step_x = max(1, tile_shape[1] - 2 * halo[1])

    y_starts = make_smart_tile_starts(
        image_size=image_shape[0],
        tile_size=tile_shape[0],
        step_size=step_y
    )

    x_starts = make_smart_tile_starts(
        image_size=image_shape[1],
        tile_size=tile_shape[1],
        step_size=step_x
    )

    coords = [(y, x) for y in y_starts for x in x_starts]

    print("    Smart tiling:")
    print(f"      Image shape: {image_shape}")
    print(f"      Tile shape: {tile_shape}")
    print(f"      Halo: {halo}")
    print(f"      Step Y/X: {step_y}, {step_x}")
    print(f"      Y starts: {y_starts}")
    print(f"      X starts: {x_starts}")
    print(f"      Total tiles: {len(coords)}")

    return coords


# =============================================================================
# MICRO-SAM TILED SEGMENTATION
# =============================================================================
def make_full_image_preview_uint8(image, p_low=0.5, p_high=99.8):
    """
    Convert full 16-bit image to uint8 for PNG preview.
    """
    return robust_percentile_normalize_to_uint8(
        image,
        p_low=p_low,
        p_high=p_high
    )


def draw_tile_boundaries_on_rgb(rgb, tile_records, color=(255, 255, 0), core_color=(0, 255, 255)):
    """
    Draw tile boundaries and tile labels onto an RGB preview.

    Yellow = full tile boundary.
    Cyan = written core boundary, if available.
    """
    out = rgb.copy()

    for rec in tile_records:
        tile_index = rec["tile_index"]

        y0 = rec["tile_y0"]
        y1 = rec["tile_y1"]
        x0 = rec["tile_x0"]
        x1 = rec["tile_x1"]

        status = rec["status"]

        # Full tile boundary
        cv2.rectangle(
            out,
            (x0, y0),
            (x1 - 1, y1 - 1),
            color,
            thickness=4
        )

        # Core boundary if present
        if (
            rec.get("core_y0") is not None
            and rec.get("core_y1") is not None
            and rec.get("core_x0") is not None
            and rec.get("core_x1") is not None
        ):
            cy0 = rec["core_y0"]
            cy1 = rec["core_y1"]
            cx0 = rec["core_x0"]
            cx1 = rec["core_x1"]

            cv2.rectangle(
                out,
                (cx0, cy0),
                (cx1 - 1, cy1 - 1),
                core_color,
                thickness=2
            )

        label_text = f"T{tile_index}: {status}"

        text_x = x0 + 20
        text_y = y0 + 45

        # Black text outline
        cv2.putText(
            out,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            thickness=6,
            lineType=cv2.LINE_AA
        )

        # White text
        cv2.putText(
            out,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return out


def save_combined_debug_pngs(
    debug_folder,
    raw_image,
    final_segmentation,
    coverage_map,
    tile_records
):
    """
    Save full-image debug PNGs:
    1. raw image with tile boundaries
    2. final combined ROI mask
    3. final ROI overlay with tile boundaries
    4. tile core coverage map
    5. tile summary CSV
    """
    if debug_folder is None:
        return

    os.makedirs(debug_folder, exist_ok=True)

    print("    Saving combined full-image debug PNGs...")

    raw_u8 = make_full_image_preview_uint8(raw_image)

    raw_rgb = np.stack(
        [raw_u8, raw_u8, raw_u8],
        axis=-1
    )

    # -------------------------------------------------------------------------
    # Final ROI mask
    # -------------------------------------------------------------------------
    roi_mask = (final_segmentation > 0).astype(np.uint8) * 255

    # -------------------------------------------------------------------------
    # ROI overlay
    # -------------------------------------------------------------------------
    roi_boundaries = find_boundaries(final_segmentation > 0, mode="outer")

    overlay_rgb = raw_rgb.copy()

    # ROI boundaries in red
    overlay_rgb[roi_boundaries, 0] = 255
    overlay_rgb[roi_boundaries, 1] = 0
    overlay_rgb[roi_boundaries, 2] = 0

    # -------------------------------------------------------------------------
    # Draw tile boundaries
    # -------------------------------------------------------------------------
    raw_with_tiles = draw_tile_boundaries_on_rgb(
        raw_rgb,
        tile_records,
        color=(255, 255, 0),
        core_color=(0, 255, 255)
    )

    overlay_with_tiles = draw_tile_boundaries_on_rgb(
        overlay_rgb,
        tile_records,
        color=(255, 255, 0),
        core_color=(0, 255, 255)
    )

    # -------------------------------------------------------------------------
    # Coverage map
    # -------------------------------------------------------------------------
    coverage_rgb = np.stack(
        [coverage_map, coverage_map, coverage_map],
        axis=-1
    ).astype(np.uint8)

    coverage_with_tiles = draw_tile_boundaries_on_rgb(
        coverage_rgb,
        tile_records,
        color=(255, 255, 0),
        core_color=(0, 255, 255)
    )

    # -------------------------------------------------------------------------
    # Save files
    # -------------------------------------------------------------------------
    tifffile.imwrite(
        os.path.join(debug_folder, "combined_01_raw_with_tile_boundaries.png"),
        raw_with_tiles
    )

    tifffile.imwrite(
        os.path.join(debug_folder, "combined_02_roi_mask.png"),
        roi_mask
    )

    tifffile.imwrite(
        os.path.join(debug_folder, "combined_03_roi_overlay_with_tile_boundaries.png"),
        overlay_with_tiles
    )

    tifffile.imwrite(
        os.path.join(debug_folder, "combined_04_tile_core_coverage.png"),
        coverage_with_tiles
    )

    pd.DataFrame(tile_records).to_csv(
        os.path.join(debug_folder, "combined_05_tile_summary.csv"),
        index=False
    )

    print(f"    Saved combined debug files to: {debug_folder}")

def segment_by_tiling(
    predictor,
    segmenter,
    image,
    tile_shape,
    halo,
    ndim,
    debug_folder=None,
    **generate_kwargs
):
    if ndim != 2:
        raise ValueError("Only 2D supported")

    image_shape = image.shape
    final_segmentation = np.zeros(image_shape, dtype=np.uint32)

    # This shows which final tile-core regions were actually written/covered.
    coverage_map = np.zeros(image_shape, dtype=np.uint8)

    max_id = 0

    coords = make_tile_coordinates(
        image_shape=image_shape,
        tile_shape=tile_shape,
        halo=halo
    )

    print(f"    Preprocess mode: {PREPROCESS_MODE}")

    if should_save_debug_images():
        print(f"    Debug folder: {debug_folder}")
    else:
        print("    Tile PNG debug saving is OFF.")

    debug_saved = 0
    skipped_empty_tiles = 0
    processed_tiles = 0
    no_result_tiles = 0
    filtered_empty_tiles = 0
    written_tiles = 0

    tile_records = []

    for tile_index, corner_coord in enumerate(
        tqdm(coords, desc="    Tiling Progress", leave=False)
    ):
        y0, x0 = corner_coord

        y1 = min(y0 + tile_shape[0], image_shape[0])
        x1 = min(x0 + tile_shape[1], image_shape[1])

        tile_slices_global = (
            slice(y0, y1),
            slice(x0, x1)
        )

        tile_data = image[tile_slices_global]
        current_tile_shape = tile_data.shape

        rec = {
            "tile_index": tile_index,
            "tile_y0": y0,
            "tile_y1": y1,
            "tile_x0": x0,
            "tile_x1": x1,
            "core_y0": None,
            "core_y1": None,
            "core_x0": None,
            "core_x1": None,
            "status": "started",
            "raw_tile_max": float(tile_data.max()),
            "raw_tile_mean": float(tile_data.mean()),
            "n_raw_sam_masks": 0,
            "n_filtered_labels": 0,
            "max_label_after_write": int(max_id),
        }

        if SKIP_EMPTY_TILES and not tile_has_enough_signal(tile_data):
            skipped_empty_tiles += 1
            rec["status"] = "skipped_empty"
            tile_records.append(rec)

            print(f"    [SKIP EMPTY] tile {tile_index}: y={y0}:{y1}, x={x0}:{x1}")
            continue

        processed_tiles += 1

        processed_uint8, tile_data_rgb = preprocess_tile_for_sam(
            tile_data,
            mode=PREPROCESS_MODE
        )

        predictor.set_image(tile_data_rgb)

        image_embeddings_dict = {
            "features": predictor.get_image_embedding(),
            "original_size": predictor.original_size,
            "input_size": predictor.input_size
        }

        segmenter.initialize(
            image=tile_data_rgb,
            image_embeddings=image_embeddings_dict
        )

        segmentation_result = segmenter.generate(**generate_kwargs)

        # micro-sam ≥1.1: InstanceSegmentationWithDecoder.generate() returns a
        # numpy labeled array (output_mode='instance_segmentation' default).
        # Older API / AMGBase returns List[Dict] with "segmentation" keys.
        if isinstance(segmentation_result, np.ndarray):
            labeled_tile = segmentation_result.astype(np.uint32)
        elif isinstance(segmentation_result, list):
            if not segmentation_result:
                no_result_tiles += 1
                rec["status"] = "no_sam_result"
                tile_records.append(rec)
                if (
                    should_save_debug_images()
                    and debug_folder is not None
                    and debug_saved < MAX_DEBUG_TILES
                    and tile_index % DEBUG_TILE_EVERY_N == 0
                ):
                    save_debug_pngs(debug_folder, tile_index, corner_coord,
                                    tile_data, processed_uint8,
                                    np.zeros(tile_data.shape, dtype=np.uint32), [])
                    debug_saved += 1
                print(f"    [NO SAM RESULT] tile {tile_index}: y={y0}:{y1}, x={x0}:{x1}")
                continue
            labeled_tile = np.zeros(tile_data_rgb.shape[:2], dtype=np.uint32)
            for i, mask_dict in enumerate(segmentation_result, 1):
                labeled_tile[mask_dict["segmentation"]] = i
        else:
            raise ValueError(f"Unexpected segmentation result type: {type(segmentation_result)}")

        if labeled_tile.max() == 0:
            no_result_tiles += 1
            rec["status"] = "no_sam_result"
            tile_records.append(rec)
            print(f"    [NO SAM RESULT] tile {tile_index}: y={y0}:{y1}, x={x0}:{x1}")
            continue

        rec["n_raw_sam_masks"] = int(labeled_tile.max())

        unfiltered_labeled_tile = labeled_tile.copy()

        labeled_tile = filter_labeled_tile_by_raw_mean_and_shape(
            labeled_tile,
            tile_data
        )

        filtered_labels = np.unique(labeled_tile)
        filtered_labels = filtered_labels[filtered_labels > 0]
        rec["n_filtered_labels"] = int(len(filtered_labels))

        if (
            should_save_debug_images()
            and debug_folder is not None
            and debug_saved < MAX_DEBUG_TILES
            and tile_index % DEBUG_TILE_EVERY_N == 0
        ):
            save_debug_pngs(debug_folder, tile_index, corner_coord,
                            tile_data, processed_uint8, labeled_tile, [])
            save_debug_pngs(debug_folder + "_UNFILTERED", tile_index, corner_coord,
                            tile_data, processed_uint8, unfiltered_labeled_tile, [])
            debug_saved += 1

        if labeled_tile.max() == 0:
            filtered_empty_tiles += 1
            rec["status"] = "filtered_empty"
            tile_records.append(rec)

            print(f"    [FILTERED EMPTY] tile {tile_index}: y={y0}:{y1}, x={x0}:{x1}")
            continue

        # ---------------------------------------------------------------------
        # EDGE-AWARE CORE REGION
        # ---------------------------------------------------------------------
        # At the image border, we do not discard the outer halo.
        # Inside the image, we discard halo regions so overlapping tiles do not
        # duplicate ROIs.
        # ---------------------------------------------------------------------

        local_y_start = 0 if y0 == 0 else halo[0]
        local_x_start = 0 if x0 == 0 else halo[1]

        local_y_end = current_tile_shape[0] if y1 == image_shape[0] else current_tile_shape[0] - halo[0]
        local_x_end = current_tile_shape[1] if x1 == image_shape[1] else current_tile_shape[1] - halo[1]

        if local_y_start >= local_y_end or local_x_start >= local_x_end:
            rec["status"] = "bad_core"
            tile_records.append(rec)

            print(f"    [BAD CORE] tile {tile_index}: y={y0}:{y1}, x={x0}:{x1}")
            continue

        global_y_start = y0 + local_y_start
        global_y_end = y0 + local_y_end
        global_x_start = x0 + local_x_start
        global_x_end = x0 + local_x_end

        rec["core_y0"] = global_y_start
        rec["core_y1"] = global_y_end
        rec["core_x0"] = global_x_start
        rec["core_x1"] = global_x_end

        core_slices_local = (
            slice(local_y_start, local_y_end),
            slice(local_x_start, local_x_end)
        )

        core_slices_global = (
            slice(global_y_start, global_y_end),
            slice(global_x_start, global_x_end)
        )

        tile_core_data = labeled_tile[core_slices_local]

        # Mark coverage even if no cells in the core.
        # This tells us whether the combined image has a true tiling hole.
        coverage_map[core_slices_global] = 255

        if tile_core_data.max() == 0:
            filtered_empty_tiles += 1
            rec["status"] = "empty_core"
            tile_records.append(rec)

            print(f"    [EMPTY CORE] tile {tile_index}: y={y0}:{y1}, x={x0}:{x1}")
            continue

        tile_core_data_renumbered = np.where(
            tile_core_data > 0,
            tile_core_data + max_id,
            0
        )

        current_core_data = final_segmentation[core_slices_global]

        mask_to_write = (
            (current_core_data == 0)
            & (tile_core_data_renumbered > 0)
        )

        final_segmentation[core_slices_global][mask_to_write] = tile_core_data_renumbered[mask_to_write]

        max_id = int(final_segmentation.max())

        written_tiles += 1
        rec["status"] = "written"
        rec["max_label_after_write"] = int(max_id)
        tile_records.append(rec)

        print(
            f"    [WRITTEN] tile {tile_index}: "
            f"tile y={y0}:{y1}, x={x0}:{x1}; "
            f"core y={global_y_start}:{global_y_end}, x={global_x_start}:{global_x_end}; "
            f"raw_masks={rec['n_raw_sam_masks']}; "
            f"filtered_labels={rec['n_filtered_labels']}; "
            f"max_label={max_id}"
        )

    print(f"    Processed tiles with signal: {processed_tiles}")
    print(f"    Skipped empty/low-signal tiles: {skipped_empty_tiles}")
    print(f"    Tiles with no SAM result: {no_result_tiles}")
    print(f"    Tiles empty after filtering/core: {filtered_empty_tiles}")
    print(f"    Tiles written into final image: {written_tiles}")
    print(f"    Final objects after tiled SAM: {max_id}")

    # Save full-image combined debug outputs.
    # This is controlled by TEST_RUN/SAVE_DEBUG_IMAGES, so it will not run in normal production.
    if should_save_debug_images() and debug_folder is not None:
        save_combined_debug_pngs(
            debug_folder=debug_folder,
            raw_image=image,
            final_segmentation=final_segmentation,
            coverage_map=coverage_map,
            tile_records=tile_records
        )

        save_combined_debug_pngs(
            debug_folder=debug_folder + "_UNFILTERED_SUMMARY",
            raw_image=image,
            final_segmentation=final_segmentation,
            coverage_map=coverage_map,
            tile_records=tile_records
        )

    return final_segmentation


# =============================================================================
# FILE HANDLING
# =============================================================================

def find_input_files(root_folder):
    input_paths = []
    exts = (".czi", ".tif", ".tiff")

    print(f"Scanning for CZI and TIF files in: {root_folder}"
          + (" (root only, subfolders skipped)" if SKIP_SUBDIRS else " (+ one level of subfolders)"))

    if not os.path.isdir(root_folder):
        print(f"[ERROR] Folder not found: {root_folder}")
        return input_paths

    for entry in sorted(os.listdir(root_folder)):
        full = os.path.join(root_folder, entry)
        if os.path.isfile(full) and entry.lower().endswith(exts):
            input_paths.append(full)
        elif os.path.isdir(full) and not SKIP_SUBDIRS:
            for fname in sorted(os.listdir(full)):
                if fname.lower().endswith(exts):
                    input_paths.append(os.path.join(full, fname))

    if input_paths:
        print(f"  Found {len(input_paths)} file(s) to process.")
    return input_paths


def _mip(arr):
    """Max-intensity-project all leading non-spatial axes until the array is 2D (Y, X)."""
    while arr.ndim > 2:
        arr = np.max(arr, axis=0)
    return arr


def _save_split_tiff(channel_data, ch_index, output_subfolder, base_name, split_tiff_paths):
    """Write one 2D channel array and record the path."""
    print(
        f"    Channel {ch_index}: shape={channel_data.shape}, "
        f"dtype={channel_data.dtype}, "
        f"min={channel_data.min()}, max={channel_data.max()}, "
        f"size_MB={channel_data.nbytes / 1024**2:.1f}"
    )
    tiff_path = os.path.join(output_subfolder, f"{base_name}_split_ch{ch_index}.tif")
    imwrite(tiff_path, channel_data, photometric="minisblack", imagej=True)
    split_tiff_paths[ch_index] = tiff_path


def save_raw_split_tiffs(file_path, output_subfolder, base_name):
    """
    Split a multi-channel (and optionally multi-Z) microscopy file into
    one 2D TIFF per channel.  Z-stacks are collapsed via max intensity
    projection (MIP) before saving — CellMatchR does not do 3D analysis.
    """
    split_tiff_paths = {}

    if file_path.lower().endswith(".czi"):
        with CziFile(file_path) as czi:
            data = czi.asarray()
            axes = czi.axes
            c_axis = axes.find("C")
            z_axis = axes.find("Z")

        print(f"    CZI axes: {axes}, shape: {data.shape}")

        if c_axis == -1:
            # No explicit C axis — treat whole array as single channel after MIP
            ch = np.squeeze(data)
            while ch.ndim > 2:
                ch = np.max(ch, axis=0)
            print(f"    No C axis — single channel after MIP: shape={ch.shape}")
            _save_split_tiff(ch.astype(np.uint16), 0, output_subfolder, base_name, split_tiff_paths)
        else:
            num_channels = data.shape[c_axis]
            num_z = data.shape[z_axis] if z_axis != -1 else 1
            print(f"    Channels: {num_channels}, Z-slices: {num_z}")
            if num_z > 1:
                print(f"    Z-stack detected — applying max intensity projection per channel")
            for ch_index in range(num_channels):
                ch = np.take(data, ch_index, axis=c_axis)
                ch = np.squeeze(ch)
                while ch.ndim > 2:      # MIP over Z and any remaining singleton dims
                    ch = np.max(ch, axis=0)
                _save_split_tiff(ch.astype(np.uint16), ch_index, output_subfolder, base_name, split_tiff_paths)

    elif file_path.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(file_path) as tif:
            data = tif.asarray()
            axes_meta = (tif.series[0].axes if tif.series else "").upper()

        print(f"    TIFF shape: {data.shape}, axes metadata: '{axes_meta or 'none'}'")

        if data.ndim == 2:
            # (Y, X) — single channel, no Z
            _save_split_tiff(data.astype(np.uint16), 0, output_subfolder, base_name, split_tiff_paths)

        elif data.ndim == 3:
            has_z = "Z" in axes_meta
            has_c = "C" in axes_meta

            if has_z and not has_c:
                # (Z, Y, X) — single-channel Z-stack → MIP
                print(f"    Single-channel Z-stack ({data.shape[0]} slices) — applying MIP")
                _save_split_tiff(np.max(data, axis=0).astype(np.uint16), 0,
                                 output_subfolder, base_name, split_tiff_paths)

            elif has_c:
                # Metadata tells us where C is
                c_pos = axes_meta.find("C")
                num_channels = data.shape[c_pos]
                print(f"    Multi-channel TIFF ({num_channels} channels, axis {c_pos})")
                for ch_index in range(num_channels):
                    cd = np.take(data, ch_index, axis=c_pos).astype(np.uint16)
                    _save_split_tiff(cd, ch_index, output_subfolder, base_name, split_tiff_paths)

            elif data.shape[0] <= 10:
                # No metadata: small first dim → treat as channels (C, Y, X)
                print(f"    Interpreting as channel-first: C={data.shape[0]}, Y, X")
                for ch_index in range(data.shape[0]):
                    _save_split_tiff(data[ch_index].astype(np.uint16), ch_index,
                                     output_subfolder, base_name, split_tiff_paths)

            elif data.shape[-1] <= 10:
                # No metadata: small last dim → treat as channels (Y, X, C)
                print(f"    Interpreting as channel-last: Y, X, C={data.shape[-1]}")
                for ch_index in range(data.shape[-1]):
                    _save_split_tiff(data[..., ch_index].astype(np.uint16), ch_index,
                                     output_subfolder, base_name, split_tiff_paths)

            else:
                # Large first dim, no metadata → assume Z-stack, single channel
                print(f"    Large first dim ({data.shape[0]}) — treating as Z-stack, applying MIP")
                _save_split_tiff(np.max(data, axis=0).astype(np.uint16), 0,
                                 output_subfolder, base_name, split_tiff_paths)

        elif data.ndim == 4:
            # Multi-channel Z-stack: determine C and Z positions
            if "C" in axes_meta and "Z" in axes_meta:
                c_pos = axes_meta.find("C")
                z_pos = axes_meta.find("Z")
            else:
                # Heuristic: smaller of first two dims = C, larger = Z
                c_pos, z_pos = (0, 1) if data.shape[0] <= data.shape[1] else (1, 0)

            num_channels = data.shape[c_pos]
            num_z        = data.shape[z_pos]
            print(f"    4D TIFF — {num_channels} channels, {num_z} Z-slices — applying MIP per channel")

            for ch_index in range(num_channels):
                ch = np.take(data, ch_index, axis=c_pos)   # (Z, Y, X) or similar
                channel_data = _mip(ch).astype(np.uint16)
                _save_split_tiff(channel_data, ch_index, output_subfolder, base_name, split_tiff_paths)

        else:
            raise ValueError(
                f"Unsupported image shape: {data.shape} (ndim={data.ndim}). "
                "Expected 2D, 3D, or 4D array."
            )

    return split_tiff_paths


def save_segmentation_tiff(output_path, result):
    """
    Save segmentation in ImageJ-compatible format when possible.
    ImageJ TIFF does not support uint32 labels.
    """
    max_label = int(result.max())

    print(f"    Max label in result: {max_label}")

    if max_label <= 65535:
        result_to_save = result.astype(np.uint16)

        imwrite(
            output_path,
            result_to_save,
            photometric="minisblack",
            imagej=True
        )

    else:
        print("    [WARNING] More than 65535 objects.")
        print("    Saving as uint32 non-ImageJ TIFF.")

        result_to_save = result.astype(np.uint32)

        imwrite(
            output_path,
            result_to_save,
            photometric="minisblack"
        )




# =============================================================================
# FINAL QC PNG OUTPUT
# =============================================================================

def make_final_qc_overlay(raw_image, label_image, output_png_path, downsample=4):
    """
    Save a full-image PNG showing the raw channel image with predicted ROI boundaries.
    This runs independently of TEST_RUN/debug mode.
    """
    print(f"    Creating final QC overlay PNG: {output_png_path}")

    if downsample < 1:
        downsample = 1

    raw_small = raw_image[::downsample, ::downsample]
    label_small = label_image[::downsample, ::downsample]

    raw_u8 = robust_percentile_normalize_to_uint8(
        raw_small,
        p_low=0.5,
        p_high=99.8
    )

    rgb = np.stack([raw_u8, raw_u8, raw_u8], axis=-1)

    boundaries = find_boundaries(label_small > 0, mode="outer")

    # Red ROI boundaries
    rgb[boundaries, 0] = 255
    rgb[boundaries, 1] = 0
    rgb[boundaries, 2] = 0

    tifffile.imwrite(output_png_path, rgb)
    print(f"    Saved final QC overlay PNG: {output_png_path}")


def make_final_qc_mask_png(label_image, output_png_path, downsample=4):
    """
    Save a simple binary ROI mask PNG.
    This is useful because Fiji/ImageJ can display label TIFFs poorly when there are many objects.
    """
    print(f"    Creating final binary ROI mask PNG: {output_png_path}")

    if downsample < 1:
        downsample = 1

    label_small = label_image[::downsample, ::downsample]
    mask_png = ((label_small > 0) * 255).astype(np.uint8)

    tifffile.imwrite(output_png_path, mask_png)
    print(f"    Saved final binary ROI mask PNG: {output_png_path}")


# =============================================================================
# CHECKPOINT COMPATIBILITY PATCH
# =============================================================================

def _patch_decoder_checkpoints(model_type: str) -> None:
    """Rename old torch_em sampler key names in cached decoder checkpoints.

    torch_em 0.8.0 renamed the Upsampler internal attribute from
    ``self.block`` to ``self.conv``, which changed the state-dict key from
    ``decoder.samplers.N.block.{weight,bias}`` to
    ``decoder.samplers.N.conv.{weight,bias}``.  The published bioimage.io
    checkpoints pre-date that rename.  This function detects and fixes the
    mismatch in-place the first time segmentation is run.
    """
    import re, shutil
    try:
        from micro_sam.util import get_cache_directory
        cache_dir = os.path.join(get_cache_directory(), "models")
    except Exception:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "micro_sam", "models")

    decoder_path = os.path.join(cache_dir, f"{model_type}_decoder")
    if not os.path.isfile(decoder_path):
        return  # not yet downloaded — nothing to patch

    import torch
    state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    old_keys = [k for k in state if re.match(r"decoder\.samplers\.\d+\.block\.(weight|bias)$", k)]
    if not old_keys:
        return  # already up-to-date

    print(f"[INFO] Patching decoder checkpoint for '{model_type}' "
          f"(renaming {len(old_keys)} sampler keys: block → conv) …")
    shutil.copy2(decoder_path, decoder_path + ".bak")
    new_state = {
        re.sub(r"(decoder\.samplers\.\d+\.)block\.", r"\1conv.", k): v
        for k, v in state.items()
    }
    torch.save(new_state, decoder_path)
    print(f"[INFO] Checkpoint patched successfully.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global MIN_OBJECT_SIZE, MAX_OBJECT_SIZE, MAX_MAJOR_MINOR_RATIO, RAW_MEAN_MIN
    is_dry_run = DRY_RUN and not TEST_RUN

    all_files = find_input_files(ROOT_INPUT_FOLDER)

    if TEST_RUN and all_files:
        all_files = all_files[:1]

    if not all_files:
        print("[ERROR] No input files found. Check that input_root_folder is set correctly.")
        sys.exit(1)

    print(f"Found {len(all_files)} file(s) to process.")

    predictor, segmenter = None, None

    if not is_dry_run:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")

        # Patch cached decoder checkpoints if they were saved with old torch_em naming
        # (torch_em 0.8+ renamed Upsampler.block → Upsampler.conv; published bioimage.io
        # checkpoints pre-date this rename, so we fix the keys on first run).
        _patch_decoder_checkpoints(MODEL_TYPE)

        predictor, segmenter = micro_sam.automatic_segmentation.get_predictor_and_segmenter(
            model_type=MODEL_TYPE,
            device=device,
        )

    generate_kwargs = {
        "min_size": MIN_OBJECT_SIZE,
    }

    for file_path in all_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        print(f"\n--- Planning for: {base_name} ---")
        print(f"Input file: {file_path}")

        output_subfolder = os.path.join(MAIN_OUTPUT_FOLDER, base_name)

        try:
            if not is_dry_run:
                os.makedirs(output_subfolder, exist_ok=True)

            split_tiff_paths = save_raw_split_tiffs(
                file_path,
                output_subfolder,
                base_name
            )

            if is_dry_run:
                continue

            for ch_to_predict in CHANNELS_TO_PREDICT:
                if ch_to_predict not in split_tiff_paths:
                    print(f"  [SKIP] Channel {ch_to_predict} not found.")
                    continue

                # Apply per-channel filter overrides if configured
                _saved_filters = (MIN_OBJECT_SIZE, MAX_OBJECT_SIZE,
                                  MAX_MAJOR_MINOR_RATIO, RAW_MEAN_MIN)
                if PER_CHANNEL_FILTERS_ENABLED and ch_to_predict in PER_CHANNEL_FILTERS:
                    pcf = PER_CHANNEL_FILTERS[ch_to_predict]
                    MIN_OBJECT_SIZE      = int(pcf.get("min_size",     MIN_OBJECT_SIZE))
                    MAX_OBJECT_SIZE      = int(pcf.get("max_size",     MAX_OBJECT_SIZE))
                    MAX_MAJOR_MINOR_RATIO = float(pcf.get("max_ratio", MAX_MAJOR_MINOR_RATIO))
                    RAW_MEAN_MIN         = float(pcf.get("min_intensity", RAW_MEAN_MIN))
                    print(f"  Per-channel filters ch{ch_to_predict}: "
                          f"min={MIN_OBJECT_SIZE}, max={MAX_OBJECT_SIZE}, "
                          f"ratio={MAX_MAJOR_MINOR_RATIO}, intensity={RAW_MEAN_MIN}")

                print(f"\n  - Predicting on Channel {ch_to_predict}...")

                tiff_path = split_tiff_paths[ch_to_predict]
                image_data = tifffile.imread(tiff_path).astype(np.uint16)

                print(
                    f"    Loaded split TIFF: shape={image_data.shape}, "
                    f"dtype={image_data.dtype}, "
                    f"min={image_data.min()}, max={image_data.max()}"
                )

                debug_folder = os.path.join(
                    output_subfolder,
                    f"DEBUG_microSAM_ch{ch_to_predict}_{PREPROCESS_MODE}_tile2048"
                )

                print("    Running custom tiled micro-SAM prediction.")
                print("    Full-image prediction is disabled.")
                print(f"    Debug folder: {debug_folder}")

                result = segment_by_tiling(
                    predictor,
                    segmenter,
                    image_data,
                    TILE_SHAPE_2D,
                    HALO_2D,
                    NDIM,
                    debug_folder=debug_folder,
                    **generate_kwargs
                )

                output_path = os.path.join(
                    output_subfolder,
                    f"{base_name}_predict_original_ch{ch_to_predict}_{PREPROCESS_MODE}_tile2048.tif"
                )

                save_segmentation_tiff(output_path, result)

                print(f"    Saved prediction to: {output_path}")

                # Always save PNG QC outputs so segmentation can be checked immediately,
                # even during full batch runs.
                if SAVE_FINAL_QC_OVERLAY:
                    qc_overlay_path = os.path.join(
                        output_subfolder,
                        f"{base_name}_QC_overlay_ch{ch_to_predict}_{PREPROCESS_MODE}_tile2048.png"
                    )

                    qc_mask_path = os.path.join(
                        output_subfolder,
                        f"{base_name}_QC_binary_mask_ch{ch_to_predict}_{PREPROCESS_MODE}_tile2048.png"
                    )

                    make_final_qc_overlay(
                        raw_image=image_data,
                        label_image=result,
                        output_png_path=qc_overlay_path,
                        downsample=FINAL_QC_DOWNSAMPLE
                    )

                    make_final_qc_mask_png(
                        label_image=result,
                        output_png_path=qc_mask_path,
                        downsample=FINAL_QC_DOWNSAMPLE
                    )

                # Restore global filter values for the next channel
                (MIN_OBJECT_SIZE, MAX_OBJECT_SIZE,
                 MAX_MAJOR_MINOR_RATIO, RAW_MEAN_MIN) = _saved_filters

        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"--- Finished: {base_name} ---\n")


def _apply_config(config_path: str) -> None:
    """Override module-level settings from a CellMatchr config file."""
    import json
    global ROOT_INPUT_FOLDER, MAIN_OUTPUT_FOLDER, CHANNELS_TO_PREDICT
    global MODEL_TYPE, PREPROCESS_MODE
    global MIN_OBJECT_SIZE, MAX_OBJECT_SIZE, MAX_MAJOR_MINOR_RATIO, RAW_MEAN_MIN
    global TILE_SHAPE_2D, HALO_2D, FINAL_QC_DOWNSAMPLE
    global DRY_RUN, TEST_RUN, SKIP_SUBDIRS
    global PER_CHANNEL_FILTERS_ENABLED, PER_CHANNEL_FILTERS

    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("input_root_folder"):
        ROOT_INPUT_FOLDER = cfg["input_root_folder"]
    if cfg.get("output_root_folder"):
        MAIN_OUTPUT_FOLDER = cfg["output_root_folder"]

    raw_ch = cfg.get("seg_channels", "")
    if raw_ch:
        CHANNELS_TO_PREDICT = [
            int(x.strip()) for x in str(raw_ch).split(",") if x.strip()
        ]

    if cfg.get("seg_model_type"):
        MODEL_TYPE = cfg["seg_model_type"]
    if cfg.get("seg_preprocess_mode"):
        PREPROCESS_MODE = cfg["seg_preprocess_mode"]

    for src, dst in [
        ("seg_min_object_size", "MIN_OBJECT_SIZE"),
        ("seg_max_object_size", "MAX_OBJECT_SIZE"),
        ("seg_qc_downsample",   "FINAL_QC_DOWNSAMPLE"),
    ]:
        if cfg.get(src) is not None:
            globals()[dst] = int(cfg[src])

    for src, dst in [
        ("seg_max_ratio",    "MAX_MAJOR_MINOR_RATIO"),
        ("seg_raw_mean_min", "RAW_MEAN_MIN"),
    ]:
        if cfg.get(src) is not None:
            globals()[dst] = float(cfg[src])

    th = int(cfg.get("seg_tile_h", TILE_SHAPE_2D[0]))
    tw = int(cfg.get("seg_tile_w", TILE_SHAPE_2D[1]))
    TILE_SHAPE_2D = (th, tw)

    hh = int(cfg.get("seg_halo_h", HALO_2D[0]))
    hw = int(cfg.get("seg_halo_w", HALO_2D[1]))
    HALO_2D = (hh, hw)

    if cfg.get("seg_dry_run") is not None:
        DRY_RUN = bool(cfg["seg_dry_run"])
    if cfg.get("seg_test_run") is not None:
        TEST_RUN = bool(cfg["seg_test_run"])
    if cfg.get("seg_skip_subdirs") is not None:
        SKIP_SUBDIRS = bool(cfg["seg_skip_subdirs"])

    PER_CHANNEL_FILTERS_ENABLED = bool(cfg.get("seg_per_channel_filters_enabled", False))
    raw_pcf = cfg.get("seg_per_channel_filters", "{}")
    if PER_CHANNEL_FILTERS_ENABLED and raw_pcf:
        try:
            parsed = json.loads(raw_pcf)
            PER_CHANNEL_FILTERS = {int(k): v for k, v in parsed.items()}
        except Exception:
            PER_CHANNEL_FILTERS = {}


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="CellMatchr Segmentation")
    _parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.cellmatchr/config.json"),
        help="Path to CellMatchr config JSON",
    )
    _args, _ = _parser.parse_known_args()
    if os.path.exists(_args.config):
        _apply_config(_args.config)
    main()