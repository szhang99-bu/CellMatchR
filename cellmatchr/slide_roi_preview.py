"""Find and draw the scan region associated with a converted image."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

if __package__:
    from .file_paths import read_tiff
    from .roi_mask_matching import roi_mask_files, roi_match_candidates
    from .slide_signal import read_run_progress
else:
    from file_paths import read_tiff
    from roi_mask_matching import roi_mask_files, roi_match_candidates
    from slide_signal import read_run_progress

MASK_FOLDER = "DONOTDELETE-MASK_FOR_VSI"


def slide_roi_for_sample(sample, *, run=None, source=None):
    """Use the exact paired region. Older runs can match a unique export."""
    if source:
        folder = Path(source).parent
        exporter = folder.parent if folder.name == "exports" else folder
        folders = [exporter / MASK_FOLDER]
    elif run:
        run = Path(run)
        inputs = read_run_progress(run).get("input_sources")
        converter = inputs.get("converter_run") if isinstance(inputs, dict) else None
        if converter == "":
            return None
        if isinstance(converter, str) and converter and Path(converter).name == converter:
            folders = [run.parent / converter / MASK_FOLDER]
        else:
            folders = [path / MASK_FOLDER for path in run.parent.glob("VSI_Converted_TIFF*")]
    else:
        return None
    masks = [mask for folder in folders for mask in roi_mask_files(folder)
             if mask.lower().endswith(("_mask_overview.tif", "_mask_overview.tiff"))]
    match, _ = roi_match_candidates(sample, masks)
    return match


def slide_outline(path, size):
    """Rasterize the boundary in cropped-image coordinates for a preview."""
    import numpy as np

    mask_path = Path(path)
    vector_path = mask_path.with_name(mask_path.name.rsplit("_mask_overview.", 1)[0] + "_overview.geojson")
    try:
        with vector_path.open(encoding="utf-8") as source:
            feature = json.load(source)["features"][0]
        properties = feature["properties"]
        base, scene = properties["vsi_base"], int(properties["slice_one_indexed"])
        if Path(base).name != base:
            raise ValueError("Invalid slide name")
        meta = mask_path.parent.parent / "intermediate_files" / f"{base}_slice{scene:02d}_overview_meta.json"
        with meta.open(encoding="utf-8") as source:
            geometry = json.load(source)
        scale = float(geometry["scale_overview_per_full"])
        vertices = np.asarray(feature["geometry"]["coordinates"][0], dtype=float)
        if scale <= 0 or vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
            raise ValueError("Invalid slide coordinates")
        overview_limit = np.asarray(geometry["overview_size_wh"]) - 1
        full_limit = np.asarray(geometry["full_size_wh"]) - 1
        start = np.clip(np.floor(np.clip(np.floor(vertices.min(axis=0)), 0, overview_limit) / scale),
                        0, full_limit)
        end = np.clip(np.ceil(np.clip(np.ceil(vertices.max(axis=0)), 0, overview_limit) / scale),
                      0, full_limit)
        display = (vertices / scale - start) * np.asarray(size) / (end - start + 1)
        display = np.clip(display, 0, np.asarray(size) - 1)
        outline = Image.new("1", size)
        ImageDraw.Draw(outline).line([tuple(point) for point in display], fill=1, width=2)
        return np.asarray(outline)
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        pass  # Early projects can have only the raster mask.

    mask = Image.fromarray((read_tiff(path) > 0).astype(np.uint8) * 255)
    bounds = mask.getbbox()
    if not bounds:
        return None
    mask = mask.crop(bounds).resize(size, Image.Resampling.NEAREST)
    padded = ImageOps.expand(mask, border=2, fill=0)
    eroded = padded.filter(ImageFilter.MinFilter(5)).crop((2, 2, size[0] + 2, size[1] + 2))
    outline = (np.asarray(mask) > 0) & (np.asarray(eroded) == 0)
    return outline


def overlay_slide_outline(image, path):
    if not path:
        return image
    outline = slide_outline(path, image.size)
    if outline is None:
        return image
    image = image.convert("RGB")
    image.paste((56, 228, 237), mask=Image.fromarray(outline))
    return image
