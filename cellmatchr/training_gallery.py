"""Small multichannel previews for the inline retraining image library."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


CHANNEL_COLORS = {
    "blue": (0.33, 0.56, 1.0),
    "green": (0.22, 1.0, 0.40),
    "red": (1.0, 0.31, 0.36),
    "purple": (0.80, 0.42, 1.0),
    "cyan": (0.22, 0.95, 1.0),
    "yellow": (1.0, 0.85, 0.30),
}
DEFAULT_COLORS = ("blue", "green", "red", "purple", "cyan", "yellow")


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
            return np.asarray(tif.asarray()[::stride, ::stride])


def composite_preview(sample_dir: str, size: tuple[int, int] = (340, 230),
                      color_names: dict[int, str] | None = None) -> Image.Image:
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
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for position, (number, image) in enumerate(channels):
        name = (color_names or {}).get(number, DEFAULT_COLORS[position % len(DEFAULT_COLORS)])
        rgb += image[:h, :w, None] * CHANNEL_COLORS.get(name.lower(), CHANNEL_COLORS["blue"])
    rgb = np.uint8(np.clip(rgb, 0, 1) * 255)
    preview = Image.fromarray(rgb, "RGB")
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    return preview
