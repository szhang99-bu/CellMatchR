"""Read full-resolution TIFF regions without loading an entire slide."""

from itertools import product
from math import ceil, prod
from pathlib import Path

import numpy as np
import tifffile

if __package__:
    from .file_paths import filesystem_path
else:
    from file_paths import filesystem_path


TIFF_EXTENSIONS = (".tif", ".tiff", ".btf", ".bigtiff")


def resize_overview(image, height, width):
    """Sample a navigation preview without casting the full image to float."""
    rows = np.linspace(0, image.shape[0] - 1, height).round().astype(np.intp)
    cols = np.linspace(0, image.shape[1] - 1, width).round().astype(np.intp)
    return image[np.ix_(rows, cols)]


class TiffSlide:
    """Expose TIFF image series and unlabelled pages as slide scenes."""

    def __init__(self, filename):
        self.path = Path(filesystem_path(filename))
        self._file = tifffile.TiffFile(self.path)
        try:
            self._scenes = []
            for index, series in enumerate(self._file.series):
                axes = series.axes
                if axes.count("Y") != 1 or axes.count("X") != 1:
                    continue
                # Generic multi-page TIFF has no channel or Z labels. Preserve
                # those pages as separate scenes so no fluorescence is combined.
                frame_axes = [axis for axis in range(len(axes)) if axes[axis] not in "TZCYXS"]
                frames = [range(series.shape[axis]) for axis in frame_axes]
                for frame in product(*frames):
                    self._scenes.append(_TiffScene(self.path, index, series,
                                                   dict(zip(frame_axes, frame))))
            if not self._scenes:
                raise ValueError("The TIFF has no images with X and Y axes.")
        except Exception:
            self.close()
            raise

    @property
    def num_scenes(self):
        return len(self._scenes)

    def get_scene(self, index):
        return self._scenes[index]

    def close(self):
        self._file.close()


class _TiffScene:
    def __init__(self, path, series_index, series, frame):
        self._path = path
        self._series_index = series_index
        self._series = series
        self._frame = frame
        self.size = self._size(series)
        self.num_channels = prod(dim for axis, dim in zip(series.axes, series.shape) if axis in "CS")
        self.name = series.name or f"TIFF series {series_index + 1}"

    @staticmethod
    def _size(level):
        return level.shape[level.axes.index("X")], level.shape[level.axes.index("Y")]

    def _sample(self, array, axes, xs, ys):
        index = []
        kept_axes = []
        for number, axis in enumerate(axes):
            if axis in "XYCS":
                index.append(xs if axis == "X" else ys if axis == "Y" else slice(None))
                kept_axes.append(axis)
            else:
                index.append(self._frame.get(number, 0))

        result = np.array(array[tuple(index)], copy=True)
        if "Z" in axes:
            depth = axes.index("Z")
            for plane in range(1, array.shape[depth]):
                index[depth] = plane
                np.maximum(result, array[tuple(index)], out=result)
        order = [kept_axes.index("Y"), kept_axes.index("X")]
        order.extend(i for i, axis in enumerate(kept_axes) if axis in "CS")
        result = result.transpose(order)
        return result.reshape(*result.shape[:2], -1) if result.ndim > 3 else result

    def read_block(self, rect=None, size=None):
        full_width, full_height = self.size
        x, y, width, height = rect or (0, 0, full_width, full_height)
        out_width, out_height = size or (width, height)
        if (min(x, y) < 0 or min(width, height, out_width, out_height) < 1
                or x + width > full_width or y + height > full_height):
            raise ValueError(f"TIFF read rectangle {rect} or output size {size} is out of bounds.")

        level_index = 0
        if size is not None:
            for i, level in enumerate(self._series.levels[1:], start=1):
                if level.axes != self._series.axes:
                    continue
                level_width, level_height = self._size(level)
                if (level_width * width >= out_width * full_width
                        and level_height * height >= out_height * full_height):
                    level_index = i
        level = self._series.levels[level_index]
        level_width, level_height = self._size(level)
        left = x * level_width // full_width
        top = y * level_height // full_height
        right = ceil((x + width) * level_width / full_width)
        bottom = ceil((y + height) * level_height / full_height)
        xs = slice(left, right, max(1, (right - left) // out_width))
        ys = slice(top, bottom, max(1, (bottom - top) // out_height))

        cropped = self._read_sampled(level_index, xs, ys)
        if cropped.shape[:2] != (out_height, out_width):
            return resize_overview(cropped, out_height, out_width)
        return cropped

    def read_strided(self, stride):
        """Read level-zero pixels on the same grid as a strided label mask."""
        return self._read_sampled(0, slice(None, None, stride), slice(None, None, stride))

    def _read_sampled(self, level_index, xs, ys):
        level = self._series.levels[level_index]
        try:
            mapped = tifffile.memmap(self._path, series=self._series_index, level=level_index, mode="r")
        except ValueError:
            try:
                import zarr
                store = self._series.aszarr(level=level_index)
            except (ImportError, ValueError) as exc:
                # tifffile wraps the missing Zarr 3 store module in a ValueError.
                missing_module = getattr(exc.__cause__, "name", None)
                if isinstance(exc, ValueError) and missing_module not in ("zarr.abc", "zarr.abc.store"):
                    raise
                raise RuntimeError("Reading tiled or compressed TIFF slides requires Zarr 3. "
                                   "Update the CellMatchR environment from its environment file.") from exc
            try:
                return self._sample(zarr.open_array(store, mode="r"), level.axes, xs, ys)
            finally:
                store.close()
        return self._sample(mapped, level.axes, xs, ys)
