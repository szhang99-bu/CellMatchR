"""Prepare a Model Selection Wizard preview and run production segmentation on its crop."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import sleep
import traceback

import numpy as np
from PIL import Image, ImageFilter
import tifffile

from cellmatchr.file_paths import filesystem_path
from cellmatchr.legacy_cellpose import segmentation_command


def write_json(path, data):
    # The launcher polls for progress while this process is writing it.
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data), encoding="utf-8")
    for attempt in range(5):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            sleep(0.04)


def file_identity(path):
    path = Path(filesystem_path(path)).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def crop_box(x, y, size, width, height):
    side = min(max(1, size), width, height)
    left = max(0, min(int(x) - side // 2, width - side))
    top = max(0, min(int(y) - side // 2, height - side))
    return left, top, left + side, top + side


def source_pixels(source):
    """Map a source channel or original stain RGB without reading the whole image."""
    if "shape" in source:
        return np.memmap(source["path"], mode="r", dtype=source["dtype"], shape=tuple(source["shape"]))
    data = tifffile.memmap(source["path"], mode="r")
    axis = source.get("axis")
    if axis == 0:
        return data[source["index"]]
    if axis == 2:
        return data[..., source["index"]]
    return data


def display_image(pixels):
    """Apply display contrast only. Color stain is always shown in its original polarity."""
    if pixels.ndim == 3 and pixels.dtype == np.uint8:
        return Image.fromarray(pixels)
    stride = max(1, max(pixels.shape[:2]) // 600)
    low, high = np.percentile(pixels[::stride, ::stride], (0.5, 99.8))
    scaled = (pixels.astype(np.float32) - low) * (255 / max(high - low, 1))
    return Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8)).convert("RGB")


def _direct_tiff_sources(path, brightfield, scratch, seg):
    """Use disk-backed pixels for standard slide exports and preserve stain RGB."""
    path = filesystem_path(path)
    with tifffile.TiffFile(path) as tif:
        axes = tif.series[0].axes.upper()
        if not brightfield and axes not in {"YX", "CYX", "YXC"}:
            return None
        try:
            data = tifffile.memmap(path, mode="r")
        except ValueError:
            if not brightfield:
                return None
            data = tif.asarray(out=str(scratch / "decoded-stain.bin"))
    if brightfield:
        native = seg._brightfield_pixels(data, axes)
        if native is data and isinstance(data, np.memmap):
            return {0: {"path": path}}
        # Preserve the exact channel order and dtype even for planar or compressed stain TIFFs.
        cache = scratch / "stain.bin"
        mapped = np.memmap(cache, mode="w+", dtype=native.dtype, shape=native.shape)
        for row in range(0, native.shape[0], 1024):
            mapped[row:row + 1024] = native[row:row + 1024]
        mapped.flush()
        return {0: {"path": str(cache), "dtype": native.dtype.str, "shape": native.shape}}
    if axes == "YX":
        return {0: {"path": path}}
    axis = axes.index("C")
    return {index: {"path": path, "axis": axis, "index": index}
            for index in range(data.shape[axis])}


def prepare(config, scratch):
    from cellmatchr import segmentation as seg
    from cellmatchr.parameter_wizard_compute import _find_first_sample

    seg.SKIP_SUBDIRS = bool(config.get("seg_skip_subdirs"))
    path, scene, name = _find_first_sample(config)
    if path is None:
        raise ValueError("No input image found. Check the input folder in Step 2.")
    brightfield = config.get("input_type") == "vsi" and config.get("slide_signal_type") == "brightfield"
    sources = (_direct_tiff_sources(path, brightfield, scratch, seg)
               if path.lower().endswith((".tif", ".tiff")) else None)
    if sources is None:
        split = seg.save_raw_split_tiffs(path, str(scratch), "preview", lif_scene=scene,
                                        brightfield=brightfield)
        sources = {channel: {"path": channel_path} for channel, channel_path in split.items()}
    channels = {}
    for channel, source in sorted(sources.items()):
        pixels = source_pixels(source)
        height, width = pixels.shape[:2]
        stride = max(1, (max(width, height) + 899) // 900)
        overview = scratch / f"channel-{channel}.png"
        display_image(pixels[::stride, ::stride]).save(overview)
        channels[channel] = dict(source, width=width, height=height, overview=str(overview))
    if not channels:
        raise ValueError("The first image has no readable channels.")
    title = name if scene is None else f"{name} • Scene {scene + 1}"
    write_json(scratch / "preview.json", {
        "sample": title, "channels": channels, "brightfield": brightfield,
        "source_path": file_identity(path), "scene": scene,
    })


def patch_pixels(source, box, brightfield):
    left, top, right, bottom = box
    pixels = source_pixels(source)[top:bottom, left:right]
    return np.array(pixels) if brightfield else pixels.astype(np.uint16)


def _overlay(raw, prediction, destination):
    from skimage.segmentation import find_boundaries

    labels = tifffile.imread(prediction)
    if labels.shape != raw.shape[:2]:
        raise ValueError("The prediction mask does not match the selected patch.")
    boundary = Image.fromarray((find_boundaries(labels, mode="inner") * 255).astype(np.uint8))
    image = display_image(raw)
    image.paste((255, 207, 76), mask=boundary.filter(ImageFilter.MaxFilter(3)))
    image.save(destination)
    return int(np.unique(labels[labels > 0]).size)


def compare(request, scratch):
    raw = patch_pixels(request["source"], request["box"], request["brightfield"])
    inputs = scratch / "input"
    inputs.mkdir()
    patch_path = inputs / "patch.tif"
    tifffile.imwrite(patch_path, raw, photometric="rgb" if raw.ndim == 3 else "minisblack")
    display_image(raw).save(scratch / "original.png")
    choices = request["models"]
    progress = {"current": "", "results": [], "total": len(choices)}
    for index, (name, settings) in enumerate(choices):
        if (scratch / "stop").exists():
            break
        progress["current"] = name
        write_json(scratch / "progress.json", progress)

        model_dir = scratch / f"model-{index}"
        model_dir.mkdir()
        config = dict(request["config"], **settings)
        output = model_dir / "output"
        config.update(input_root_folder=str(inputs), output_root_folder=str(output),
                      seg_channels="0", seg_tile_h=raw.shape[0], seg_tile_w=raw.shape[1],
                      seg_halo_h=0, seg_halo_w=0, seg_skip_subdirs=True,
                      seg_min_object_size=0, seg_max_object_size=0, seg_max_ratio=0,
                      seg_raw_mean_min=0, seg_raw_mean_max=0, seg_per_channel_filters_enabled=False,
                      seg_test_run=False, seg_resume=False, seg_dry_run=False)
        config_path = model_dir / "settings.json"
        write_json(config_path, config)
        log = model_dir / "run.log"
        result = {"name": name, "log": str(log)}
        overlay = model_dir / "overlay.png"
        try:
            with log.open("wb") as stream:
                # Route legacy Cellpose and its model weights through the same setup as Step 3.
                command, env = segmentation_command(
                    str(Path(__file__).with_name("segmentation.py")), str(config_path), config)
                command.insert(1, "-u")
                completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                if settings["seg_backend"] == "stardist" and b"No module named 'stardist'" in log.read_bytes():
                    raise RuntimeError("StarDist is not installed in this environment.")
                raise RuntimeError("Segmentation failed. Open the log for details.")
            sample = output / "patch"
            prediction, = sample.glob("patch_predict_original_ch0_*.tif")
            result["raw"] = str(sample / "patch_split_ch0.tif")
            result["labels"] = str(prediction)
            if raw.ndim == 3:
                result["source_rgb"] = str(patch_path)
            result["count"] = _overlay(raw, prediction, overlay)
            result["overlay"] = str(overlay)
        except Exception as error:
            with log.open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            result["error"] = str(error)
        progress["results"].append(result)
    progress["current"] = ""
    write_json(scratch / "progress.json", progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compare"))
    parser.add_argument("settings", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    action = prepare if args.action == "prepare" else compare
    action(settings, args.settings.parent)


if __name__ == "__main__":
    main()
