"""Filesystem path and channel filename handling shared by pipeline steps."""

import glob
import json
import os
import re
from pathlib import Path
from uuid import uuid4


def filesystem_path(path):
    """Use Windows extended-length syntax for deep paths at the I/O boundary."""
    path = os.fspath(path)
    if os.name != "nt" or path.startswith(("\\\\?\\", "\\\\.\\")):
        return path
    absolute = os.path.abspath(path)
    if len(absolute) < 248:
        return path
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def matching_paths(folder, pattern, *, recursive=False):
    """Expand a file pattern inside a literal directory."""
    pattern_path = os.path.join(glob.escape(filesystem_path(folder)), pattern)
    return [filesystem_path(path) for path in glob.glob(pattern_path, recursive=recursive)]


def write_json_atomically(path, data):
    """Keep the previous record if a write to the same directory is interrupted."""
    path = os.fspath(path)
    temporary = filesystem_path(os.path.join(os.path.dirname(path), f".cellmatchr-{uuid4().hex}.tmp"))
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
        os.replace(temporary, filesystem_path(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_tiff(path):
    """Read one literal TIFF path, including Windows paths with a \\? prefix."""
    from tifffile import imread

    return imread(Path(filesystem_path(path)))


def prediction_channel(filename):
    """Read the complete channel number from a prediction filename."""
    channels = re.findall(r"_predict_original_ch(\d+)", os.path.basename(filename), re.IGNORECASE)
    return int(channels[-1]) if channels else None
