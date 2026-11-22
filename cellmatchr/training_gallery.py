"""Small multichannel previews for the inline retraining image library."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .slide_roi_preview import overlay_slide_outline
from .slide_signal import run_is_brightfield


CHANNEL_COLORS = {
    "blue": (0.33, 0.56, 1.0),
    "green": (0.22, 1.0, 0.40),
    "red": (1.0, 0.31, 0.36),
    "purple": (0.80, 0.42, 1.0),
    "cyan": (0.22, 0.95, 1.0),
    "yellow": (1.0, 0.85, 0.30),
}
DEFAULT_COLORS = ("blue", "green", "red", "purple", "cyan", "yellow")


def unavailable_gallery_samples(sources, examples, skipped):
    """Show every segmented sample while keeping training eligibility separate."""
    from .retraining import _review_complete, example_id
    from .roi_editor import _find_mask
    from .validation_state import get_change_state, get_validation_tsv_path

    present = {item["key"] for item in examples}
    unavailable = []
    for source in sources:
        project = Path(source["project"]).expanduser().resolve()
        run = source["seg_run"]
        channel = int(source["channel"])
        title = source.get("name") or project.name
        for sample in sorted((project / run).iterdir()):
            if not sample.is_dir() or sample.name.startswith((".", "DEBUG")):
                continue
            if not any(path.is_file() and path.suffix.lower() in (".tif", ".tiff")
                       for path in sample.iterdir()):
                continue
            key = example_id(str(project), run, channel, sample.name)
            if key in present:
                continue
            present.add(key)
            prefix = f"{title} / {run} / ch{channel}: {sample.name}: "
            reason = next((note[len(prefix):] for note in skipped if note.startswith(prefix)),
                          "No reviewed mask is available for this channel.")
            mask = _find_mask(str(sample), channel)
            verified = (
                mask is not None and _review_complete(sample, channel)
                and get_change_state(str(sample), channel, mask, verify_hash=False)["status"]
                in ("current", "legacy")
                and get_validation_tsv_path(str(sample), channel) is not None
            )
            unavailable.append({
                "key": key, "sample": sample.name, "project": str(project),
                "project_name": title, "seg_run": run, "channel": channel,
                "folder": str(sample), "unavailable": reason, "label_ids": {},
                "verified": verified,
                "final_count": 0, "original_count": 0, "incorrect_count": 0,
            })
    return unavailable


def project_preview_settings(project: str, active_project: str = "",
                             active_config: dict | None = None, *,
                             run: str = "") -> tuple[dict[int, str], bool]:
    """Use each source project's own visible colors and brightfield choice."""
    if (active_project and
            os.path.normcase(os.path.realpath(project)) ==
            os.path.normcase(os.path.realpath(active_project))):
        settings = active_config or {}
    else:
        try:
            with (Path(project) / "cellmatchr_project.json").open(encoding="utf-8") as source:
                settings = json.load(source)
        except (OSError, ValueError, TypeError):
            settings = {}
    if not isinstance(settings, dict):
        settings = {}
    raw_colors = settings.get("val_channel_colors", "")
    names = raw_colors.split(",") if isinstance(raw_colors, str) else raw_colors
    colors = {index: name.strip() for index, name in enumerate(names)
              if isinstance(name, str) and name.strip()} if isinstance(names, (list, tuple)) else {}
    brightfield = run_is_brightfield(os.path.join(project, run) if run else "", settings)
    return colors, brightfield


def _channel_number(path: Path) -> int:
    match = re.search(r"_split_ch(\d+)\.tif{1,2}$", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _small_array(path: Path, max_side: int) -> np.ndarray:
    """Read a sparse grid from a TIFF, avoiding full allocation when possible."""
    with tifffile.TiffFile(str(path)) as tif:
        shape = tif.series[0].shape
        if len(shape) != 2:
            raise ValueError(f"Expected a split 2D channel: {path}")
        stride = max(1, int(np.ceil(max(shape) / max_side)))
        try:
            array = tifffile.memmap(str(path))
            return np.asarray(array[::stride, ::stride])
        except (OSError, ValueError, TypeError):
            pass
        try:
            import zarr
            store = tif.series[0].aszarr()
            try:
                array = zarr.open(store, mode="r")
                return np.asarray(array[::stride, ::stride])
            finally:
                store.close()
        except (ImportError, OSError, ValueError, TypeError, AttributeError):
            expanded_bytes = int(np.prod(shape)) * np.dtype(tif.series[0].dtype).itemsize
            if expanded_bytes > 256 * 1024 * 1024:
                raise ValueError("Preview needs more than 256 MB to decode this "
                                 "compressed image; the sample remains available for training")
            return np.asarray(tif.asarray()[::stride, ::stride])


def composite_preview(sample_dir: str, size: tuple[int, int] = (340, 230),
                      color_names: dict[int, str] | None = None,
                      brightfield: bool = False, roi_path=None) -> Image.Image:
    """Overlay all split channels with their display colors, without masks."""
    paths = sorted(Path(sample_dir).glob("*_split_ch*.tif"), key=_channel_number)
    if not paths:
        raise FileNotFoundError(f"No split channels in {sample_dir}")
    channels = []
    max_side = max(size) * 2
    for path in paths:
        image = _small_array(path, max_side)
        lo, hi = np.percentile(image, (1, 99.7))
        scaled = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        channels.append((_channel_number(path), scaled))
    h = min(item.shape[0] for _, item in channels)
    w = min(item.shape[1] for _, item in channels)
    if brightfield and len(channels) == 1:
        gray = np.uint8(channels[0][1][:h, :w] * 255)
        preview = Image.fromarray(gray).convert("RGB")
        preview.thumbnail(size, Image.Resampling.LANCZOS)
        return overlay_slide_outline(preview, roi_path)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for position, (number, image) in enumerate(channels):
        name = (color_names or {}).get(number, DEFAULT_COLORS[position % len(DEFAULT_COLORS)])
        rgb += image[:h, :w, None] * CHANNEL_COLORS.get(name.lower(), CHANNEL_COLORS["blue"])
    rgb = np.uint8(np.clip(rgb, 0, 1) * 255)
    preview = Image.fromarray(rgb, "RGB")
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    return overlay_slide_outline(preview, roi_path)
