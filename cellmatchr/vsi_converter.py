# VSI file processing code, manually select a small region of the brain to create TIF
# Shiyu Zhang
# 02/01/2026

import os
import io
import json
import traceback
import numpy as np
import tifffile as tiff
import sys
import warnings
import napari
import slideio

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QGroupBox,
    QLineEdit, QHBoxLayout, QScrollArea, QSizePolicy, QProgressBar,
    QApplication, QFileDialog
)
from qtpy.QtCore import QTimer, QEventLoop, Qt
from qtpy.QtGui import QImage, QPixmap


# ----------------------------
# USER SETTINGS
# ----------------------------
OUTPUT_DIR_DEFAULT = os.path.expanduser("~/CellMatchR_VSI_Output")  # used only if the
# launcher's config is missing (e.g. running this script standalone); the
# launcher itself always supplies vsi_output_dir before spawning this app.

OVERVIEW_WIDTH = 2500          # overview width (navigation only)
START_SCENE_ID = 0             # start slice (0-index internally)
MASK_DTYPE = np.uint8          # saved mask dtype (0/255 preview mask)
SHOW_DEBUG_PANEL = False

DOCK_IMG_WIDTH = 270           # max pixel width for label / overview thumbnails in dock


# ----------------------------
# DEBUG PRINT
# ----------------------------
def debug(msg: str):
    print(f"[DEBUG] {msg}", flush=True)


def safe_slot(fn):
    """Wrap Qt callbacks so they never fail silently (and ignore Qt args like 'checked')."""
    def wrapped(*args, **kwargs):
        debug(f"BUTTON CALLBACK START: {fn.__name__}")
        try:
            return fn()
        except Exception:
            debug(f"EXCEPTION in {fn.__name__}:")
            traceback.print_exc()
            return None
    return wrapped


# ----------------------------
# UTILITIES
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def polygon_to_bbox_yx(poly_yx: np.ndarray):
    ys = poly_yx[:, 0]
    xs = poly_yx[:, 1]
    y0 = int(np.floor(ys.min()))
    y1 = int(np.ceil(ys.max()))
    x0 = int(np.floor(xs.min()))
    x1 = int(np.ceil(xs.max()))
    return y0, x0, y1, x1


def rasterize_polygon_mask(poly_yx: np.ndarray, shape_hw):
    from matplotlib.path import Path
    h, w = shape_hw

    poly = poly_yx.copy()
    poly[:, 0] = np.clip(poly[:, 0], 0, h - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, w - 1)

    yy, xx = np.mgrid[0:h, 0:w]
    points = np.vstack((yy.ravel(), xx.ravel())).T
    mask = Path(poly).contains_points(points).reshape(h, w)
    return mask.astype(MASK_DTYPE)


def get_channel_label(scene, c: int) -> str:
    for attr in ("get_channel_name", "getChannelName", "channel_name", "channelName"):
        if hasattr(scene, attr):
            try:
                fn = getattr(scene, attr)
                return str(fn(c)) if callable(fn) else str(fn)
            except Exception:
                pass
    for attr in ("channel_names", "channels", "get_channels", "getChannels"):
        if hasattr(scene, attr):
            try:
                obj = getattr(scene, attr)
                obj = obj() if callable(obj) else obj
                if isinstance(obj, (list, tuple)) and c < len(obj):
                    return str(obj[c])
            except Exception:
                pass
    return f"ch{c}"


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert a uint8 numpy array (H,W,3) or (H,W) to QPixmap."""
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w = arr.shape[:2]
        qimg = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        h, w = arr.shape[:2]
        qimg = QImage(arr.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        raise ValueError(f"Unsupported array shape {arr.shape}")
    return QPixmap.fromImage(qimg)


def _parse_label_image_size(raw_meta: str) -> "tuple[int,int] | tuple[None,None]":
    """
    Extract the pixel size (W, H) of the 'Label' stack from raw VSI JSON metadata.
    Returns (W, H) or (None, None) on failure.
    """
    import re
    try:
        # Locate the "Label" stack-name value, then search backward for the
        # nearest "Image size" tag (format: "(0, 0, W, H)").
        label_pos = raw_meta.find('"value":"Label"')
        if label_pos < 0:
            label_pos = raw_meta.find('"value": "Label"')
        if label_pos < 0:
            return None, None

        # Search the full metadata before the Label tag so we catch files
        # where "Image size" is thousands of characters earlier (First_run style).
        window = raw_meta[0 : label_pos]
        img_size_pos = window.rfind('"Image size"')
        if img_size_pos < 0:
            return None, None

        snippet = window[img_size_pos : img_size_pos + 200]
        m = re.search(
            r'"value"\s*:\s*"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"',
            snippet,
        )
        if not m:
            return None, None

        x0, y0, x1, y1 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        W = x1 if x0 == 0 else (x1 - x0)
        H = y1 if y0 == 0 else (y1 - y0)
        return W, H
    except Exception:
        return None, None


def _extract_label_image_from_vsi(
    vsi_path: str,
    label_wh: "tuple[int,int] | None" = None,
) -> "np.ndarray | None":
    """
    Read the slide-label photo from the Olympus VSI companion directory.

    Olympus VS200 VSI files keep the label image in:
        <vsi_dir>/_<vsi_base>_/stack1/frame_t.ets
    as a multi-level JPEG-tile pyramid.

    When ``label_wh`` = (W, H) is provided the function reconstructs the
    optimal pyramid level (targeting ~500 px in the shorter dimension) so
    the returned thumbnail has the correct aspect ratio and detail.
    Without it, a single coarsest-level tile is returned as a fallback.

    Returns an (H, W, 3) uint8 RGB array, or None if unavailable.
    """
    import math
    import struct
    try:
        vsi_dir = os.path.dirname(vsi_path)
        vsi_base = os.path.splitext(os.path.basename(vsi_path))[0]
        ets_path = os.path.join(vsi_dir, f"_{vsi_base}_", "stack1", "frame_t.ets")
        if not os.path.exists(ets_path):
            debug(f"Label ETS not found: {ets_path}")
            return None

        with open(ets_path, "rb") as fh:
            data = fh.read()

        from PIL import Image as _PILImage

        # ── Parse the ETS binary tile-index ("used chunks") table ─────────────
        # The sequential JPEG order in the file is NOT a reliable pyramid
        # layout (it differs between captures), so we MUST use the authoritative
        # per-tile coordinates recorded in this index.  Each chunk records its
        # dims = (x_tile_index, y_tile_index, z, pyramid_level) plus the byte
        # offset/size of its JPEG payload.
        TILE = 512
        chunks: list = []  # each: (x, y, level, offset, nbytes)
        if data[:4] == b"SIS\x00":
            try:
                ndims, = struct.unpack_from("<I", data, 12)
                used_chunk_off, = struct.unpack_from("<Q", data, 32)
                n_used_chunks, = struct.unpack_from("<I", data, 40)
                p = used_chunk_off
                for _ in range(n_used_chunks):
                    dims = struct.unpack_from(f"<{ndims}i", data, p + 4)
                    offset, = struct.unpack_from("<Q", data, p + 4 + ndims * 4)
                    nbytes, = struct.unpack_from("<I", data, p + 4 + ndims * 4 + 8)
                    # dims layout: x, y, (z...), level — level is the last dim
                    x_idx = dims[0]
                    y_idx = dims[1]
                    level = dims[-1]
                    chunks.append((x_idx, y_idx, level, offset, nbytes))
                    p += 4 + ndims * 4 + 8 + 4 + 4
            except Exception as exc:
                debug(f"ETS index parse failed ({exc}); falling back to JPEG scan.")
                chunks = []

        # ── Reconstruct using the index when dimensions + index are available ──
        if chunks and label_wh is not None and label_wh[0] and label_wh[1]:
            W_label, H_label = label_wh

            # Choose the pyramid level whose reconstructed image is closest to
            # ~500 px on the shorter side (but ≥ 300 px), matching prior behaviour.
            best_k = 0
            for k in range(1, 10):
                scale = 2 ** k
                cw = math.ceil(W_label / scale)
                ch = math.ceil(H_label / scale)
                if min(cw, ch) < 300:
                    break
                best_k = k

            # Clamp to a level that actually exists in the index.
            levels_present = sorted({c[2] for c in chunks})
            if best_k not in levels_present:
                best_k = min(levels_present, key=lambda lv: abs(lv - best_k))

            k = best_k
            scale = 2 ** k
            lvl_chunks = [c for c in chunks if c[2] == k]
            if lvl_chunks:
                n_cols = max(c[0] for c in lvl_chunks) + 1
                n_rows = max(c[1] for c in lvl_chunks) + 1
                content_w = math.ceil(W_label / scale)
                content_h = math.ceil(H_label / scale)

                debug(
                    f"Label (ETS index): W={W_label}, H={H_label}, level={k}, "
                    f"grid={n_cols}×{n_rows}, tiles={len(lvl_chunks)}, "
                    f"content={content_w}×{content_h}"
                )

                canvas = _PILImage.new(
                    "RGB", (n_cols * TILE, n_rows * TILE), (255, 255, 255)
                )
                for x_idx, y_idx, _lv, off, nb in lvl_chunks:
                    try:
                        tile_img = _PILImage.open(
                            io.BytesIO(data[off : off + nb])
                        ).convert("RGB")
                        canvas.paste(tile_img, (x_idx * TILE, y_idx * TILE))
                    except Exception:
                        pass
                result = canvas.crop((0, 0, content_w, content_h))
                arr = np.array(result.convert("RGB"))
                h80 = max(1, int(arr.shape[0] * 0.8))
                return arr[:h80, :, :]

        # ── Fallback: scan sequential JPEG tiles (legacy behaviour) ────────────
        jpeg_soi = b"\xff\xd8\xff"
        jpeg_eoi = b"\xff\xd9"
        all_tiles: list = []
        pos = 0
        while True:
            idx = data.find(jpeg_soi, pos)
            if idx < 0:
                break
            end = data.find(jpeg_eoi, idx + 3)
            if end >= 0:
                jpeg_bytes = data[idx : end + 2]
                try:
                    img = _PILImage.open(io.BytesIO(jpeg_bytes))
                    if img.size == (512, 512):
                        all_tiles.append({"offset": idx, "end": end + 2})
                except Exception:
                    pass
            pos = idx + 1

        if not all_tiles:
            debug("No valid 512×512 JPEG tiles found in label ETS.")
            return None

        debug("Label: using single coarsest pyramid tile as fallback.")
        t = all_tiles[-1]
        img = _PILImage.open(io.BytesIO(data[t["offset"] : t["end"]])).convert("RGB")
        return np.array(img)

    except Exception as exc:
        debug(f"Label image extraction failed: {exc}")
        return None


def _parse_scene_physical_coords(slide) -> dict:
    """
    Parse physical stage coordinates (µm) for each stack from VSI metadata.

    Returns dict mapping stack-name → {'ox': µm, 'oy': µm, 'sx': µm/px, 'sy': µm/px}.
    Each 'ox','oy' is the top-left corner; 'sx','sy' are pixel sizes in µm.
    Returns {} on any failure.

    Strategy: iterate over every "Stack name" entry and search *backward* for
    the nearest "RWC frame origin" (within 25 000 chars).  This handles both
    metadata layouts observed in the wild:
      • Compact (Third_Run): origin appears ~90 chars before Stack name.
      • Spread (First_run): origin appears ~17 000 chars before Stack name.
    Searching backward from the Stack name always finds the correct origin
    without crossing into a neighbouring stack's block.
    """
    try:
        meta = slide.raw_metadata
        coords: dict = {}
        search_start = 0

        while True:
            sn_tag = meta.find('"Stack name"', search_start)
            if sn_tag < 0:
                break

            # Parse the stack name value
            sv0 = meta.find('"value":', sn_tag)
            if sv0 < 0 or sv0 > sn_tag + 200:
                search_start = sn_tag + 1
                continue
            sq1 = meta.find('"', sv0 + 8) + 1
            sq2 = meta.find('"', sq1)
            stack_name = meta[sq1:sq2]

            # Search backward (up to 25 000 chars) for the nearest origin
            window_start = max(0, sn_tag - 25_000)
            window = meta[window_start : sn_tag]
            orig_local = window.rfind('"RWC frame origin"')
            if orig_local < 0:
                search_start = sn_tag + 1
                continue
            origin_tag = window_start + orig_local

            # Parse origin value
            v0 = meta.find('"value":', origin_tag)
            if v0 < 0 or v0 > origin_tag + 200:
                search_start = sn_tag + 1
                continue
            q1 = meta.find('"', v0 + 8) + 1
            q2 = meta.find('"', q1)
            origin_str = meta[q1:q2]

            # Scale is always close to the origin (within 500 chars forward)
            sc_tag = meta.find('"RWC frame scale"', origin_tag)
            if sc_tag < 0 or sc_tag > origin_tag + 500:
                search_start = sn_tag + 1
                continue
            sc0 = meta.find('"value":', sc_tag)
            scq1 = meta.find('"', sc0 + 8) + 1
            scq2 = meta.find('"', scq1)
            scale_str = meta[scq1:scq2]

            try:
                ox, oy = [float(v.strip()) for v in origin_str.strip("()").split(",")]
                sx, sy = [float(v.strip()) for v in scale_str.strip("()").split(",")]
                coords[stack_name] = {"ox": ox, "oy": oy, "sx": sx, "sy": sy}
            except Exception:
                pass

            search_start = sn_tag + 1

        return coords

    except Exception as exc:
        debug(f"Scene coord parsing failed: {exc}")
        return {}


def compute_overview(scene, overview_width: int):
    W_full, H_full = scene.size
    scale = overview_width / W_full
    overview_h = int(round(H_full * scale))

    debug(f"Scene full size W,H = {W_full},{H_full}")
    debug(f"Overview requested W,H = {overview_width},{overview_h} (scale={scale:.8f})")

    img = scene.read_block((0, 0, W_full, H_full), (overview_width, overview_h))
    debug(f"Overview read OK: dtype={img.dtype}, shape={img.shape}")

    return img, scale


def write_multichannel_tiff(path: str, img_yxc: np.ndarray):
    """Save (Y,X,C) as multichannel TIFF readable by Fiji (stores as (C,Y,X), imagej=True)."""
    if img_yxc.ndim == 2:
        tiff.imwrite(path, img_yxc, photometric="minisblack", compression=None)
        return
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {img_yxc.shape}")

    img_cyx = np.moveaxis(img_yxc, -1, 0)  # (C,Y,X)
    tiff.imwrite(path, img_cyx, imagej=True, compression=None)


# ----------------------------
# LEICA LIF BACKEND (slideio-compatible duck types)
# ----------------------------
# Leica writes slide-scanner data as .lif (same extension as confocal). We wrap
# its merged/stitched image(s) in objects that quack like slideio's Slide/Scene
# so the rest of the converter (overview, ROI drawing, full-res export) works
# unchanged.

def _resize_overview(arr: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """Resize a large (H,W,C) image to (oh,ow,C) for the navigation overview.

    Uses nearest-neighbour index sampling — pure numpy, no OpenCV (which would
    pull in a second Qt and crash the napari window), and memory-light (it never
    casts the whole gigabyte-scale image to float). Quality is fine for an
    overview; full-resolution ROI export uses an un-resized crop.
    """
    H, W = arr.shape[:2]
    ys = np.linspace(0, H - 1, oh).round().astype(np.intp)
    xs = np.linspace(0, W - 1, ow).round().astype(np.intp)
    return arr[ys][:, xs]


class _LifScene:
    """slideio-Scene-compatible wrapper over one readlif image (a stitched scene)."""

    def __init__(self, lif_img):
        self._img = lif_img
        self._W = int(lif_img.dims.x)
        self._H = int(lif_img.dims.y)
        self._C = int(lif_img.channels)
        self._nz = max(int(lif_img.nz), 1)
        self.name = lif_img.name
        self._cache = None  # full image (H, W, C), loaded once on first read

    @property
    def size(self):
        return (self._W, self._H)

    @property
    def num_channels(self):
        return self._C

    def _full(self) -> np.ndarray:
        if self._cache is None:
            debug(f"[LIF] Loading full scene '{self.name}' ({self._W}x{self._H}, "
                  f"{self._C}ch, {self._nz}z) into memory…")
            chans = []
            for c in range(self._C):
                if self._nz == 1:
                    a = np.asarray(self._img.get_frame(z=0, t=0, c=c))
                else:
                    a = np.max(np.stack(
                        [np.asarray(self._img.get_frame(z=z, t=0, c=c))
                         for z in range(self._nz)], axis=0), axis=0)
                chans.append(a)
            self._cache = np.stack(chans, axis=-1)  # (H, W, C)
            debug(f"[LIF] Scene cached: shape={self._cache.shape} dtype={self._cache.dtype}")
        return self._cache

    def read_block(self, rect=None, size=None, **kwargs):
        """Mimic slideio Scene.read_block: optional rect=(x,y,w,h) crop and
        optional size=(out_w,out_h) resize. Returns (H, W, C).

        NOTE: resize uses a pure-numpy area/nearest downsample on purpose —
        cv2.resize would import OpenCV's Qt, which collides with the PyQt5/Qt
        the napari window already loaded and crashes the app. The resize is only
        used for the navigation overview; full-res export uses a plain crop.
        """
        arr = self._full()
        if rect is not None:
            x, y, w, h = (int(v) for v in rect)
            arr = arr[y:y + h, x:x + w, :]
        if size is not None:
            ow, oh = int(size[0]), int(size[1])
            H, W = arr.shape[:2]
            if (oh, ow) != (H, W) and oh > 0 and ow > 0:
                arr = _resize_overview(arr, oh, ow)
        return np.ascontiguousarray(arr)


class _LifSlide:
    """slideio-Slide-compatible wrapper over a Leica .lif container.

    Scenes = the non-mosaic (m == 1) images: a slide-scanner file's stitched
    image(s), or each confocal acquisition. Raw mosaic tiles (m > 1) are skipped
    when a stitched image exists.
    """
    raw_metadata = ""   # VSI-only; empty for LIF

    def __init__(self, path: str):
        from readlif.reader import LifFile
        lif = LifFile(path)
        imgs = list(lif.get_iter_image())
        singles = [im for im in imgs if int(im.dims.m) <= 1]
        chosen = singles if singles else imgs
        if not singles:
            debug("[LIF] No stitched (m==1) image found — using images as-is "
                  "(mosaic stitching not supported).")
        self._scenes = [_LifScene(im) for im in chosen]

    @property
    def num_scenes(self):
        return len(self._scenes)

    def get_scene(self, i):
        return self._scenes[i]


def wait_for_viewer_close(viewer: napari.Viewer):
    """
    Block until the napari window is actually closed.
    This fixes the 'file picker pops up again while napari is open' bug.
    """
    qtwin = viewer.window._qt_window
    loop = QEventLoop()
    debug("Entering wait loop: will resume only after napari window closes.")
    qtwin.destroyed.connect(loop.quit)
    loop.exec_()
    debug("Napari window destroyed -> leaving wait loop.")


# ----------------------------
# MAIN APP
# ----------------------------
class VSINapariROIApp:
    def __init__(self, vsi_path: str, output_dir: str):
        self.vsi_path = vsi_path
        self.vsi_base = os.path.splitext(os.path.basename(vsi_path))[0]
        self.output_dir = output_dir

        self.intermediate_dir = os.path.join(self.output_dir, "intermediate_files")
        self.mask_dir = os.path.join(self.output_dir, "DONOTDELETE-MASK_FOR_VSI")
        self.export_dir = os.path.join(self.output_dir, "exports")

        ensure_dir(self.output_dir)
        ensure_dir(self.intermediate_dir)
        ensure_dir(self.mask_dir)
        ensure_dir(self.export_dir)

        self.slide = None
        self.is_lif = self.vsi_path.lower().endswith(".lif")
        self.scene_id = START_SCENE_ID
        self.scene = None
        self.total_scenes = 0
        self.is_last_scene = False

        self.overview = None
        self.scale_overview_per_full = None  # overview = full * scale

        self.viewer = None
        self.shapes = None

        # Header UI
        self.lbl_file = None
        self.lbl_slice = None
        self.lbl_status = None
        self.progress = None

        # Processing buttons
        self.btn_next = None
        self.btn_close = None
        self.btn_skip = None
        self.btn_debug = None

        # ROI naming UI
        self.roi_names = []
        self.roi_colors = []
        self.roi_name_layout = None
        self._roi_rows = []
        self._dock_panel_ref = None
        self._syncing_ui = False

        # Slide label & overview panel
        self._label_img: "np.ndarray | None" = None     # RGB thumbnail of slide label
        self._scene_coords: dict = {}                    # name -> {ox,oy,sx,sy} in µm
        self._overview_lbl_widget: "QLabel | None" = None  # dock QLabel for scene overview
        self._tools_panel_ref = None                     # left-dock editing panel

    # ---------- Colors ----------
    def _pick_roi_color(self, idx: int):
        palette = [
            (1.0, 0.2, 0.2, 1.0),
            (0.2, 0.8, 0.2, 1.0),
            (0.2, 0.6, 1.0, 1.0),
            (1.0, 0.7, 0.2, 1.0),
            (0.8, 0.2, 1.0, 1.0),
            (0.2, 1.0, 0.9, 1.0),
            (1.0, 1.0, 0.2, 1.0),
            (0.9, 0.9, 0.9, 1.0),
        ]
        return palette[idx % len(palette)]

    def _rgba_to_css(self, rgba):
        r, g, b, a = rgba
        return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.3f})"

    # ---------- Slide / scenes ----------
    def open_slide(self):
        if self.is_lif:
            debug(f"Opening LIF with readlif: {self.vsi_path}")
            self.slide = _LifSlide(self.vsi_path)
        else:
            debug(f"Opening VSI with slideio: {self.vsi_path}")
            self.slide = slideio.open_slide(self.vsi_path, "VSI")
        debug("Opened slide successfully.")
        self.total_scenes = self.slide.num_scenes
        debug(f"Number of scenes: {self.total_scenes}")
        for i in range(self.total_scenes):
            s = self.slide.get_scene(i)
            debug(f"Scene {i}: size=({s.size[0]} x {s.size[1]}), channels={s.num_channels}")

        if self.is_lif:
            # LIF has no slide-label photo or per-scene stage coordinates.
            self._label_img = None
            self._scene_coords = {}
        else:
            # Extract slide-label thumbnail and stage coordinates (non-fatal)
            _label_w, _label_h = _parse_label_image_size(self.slide.raw_metadata)
            debug(f"Label image dimensions from metadata: {_label_w}×{_label_h}")
            self._label_img = _extract_label_image_from_vsi(
                self.vsi_path, label_wh=(_label_w, _label_h) if _label_w else None
            )
            debug(f"Label image: {'OK ' + str(self._label_img.shape) if self._label_img is not None else 'not found'}")
            self._scene_coords = _parse_scene_physical_coords(self.slide)
            debug(f"Scene coords parsed: {list(self._scene_coords.keys())}")

    def load_scene(self, scene_id: int):
        if scene_id < 0 or scene_id >= self.total_scenes:
            raise ValueError(f"scene_id {scene_id} out of range 0..{self.total_scenes-1}")

        self.scene_id = scene_id
        self.scene = self.slide.get_scene(scene_id)
        self.is_last_scene = (scene_id == self.total_scenes - 1)

        W_full, H_full = self.scene.size
        C = self.scene.num_channels
        debug(f"Using SCENE_ID={scene_id}: full={W_full}x{H_full}, channels={C} | last_scene={self.is_last_scene}")

        self.overview, self.scale_overview_per_full = compute_overview(self.scene, OVERVIEW_WIDTH)

        meta = {
            "vsi_path": self.vsi_path,
            "vsi_base": self.vsi_base,
            "scene_id_zero_indexed": int(scene_id),
            "brain_slice_one_indexed": int(scene_id + 1),
            "total_slices": int(self.total_scenes),
            "full_size_wh": [int(W_full), int(H_full)],
            "overview_size_wh": [int(self.overview.shape[1]), int(self.overview.shape[0])],
            "scale_overview_per_full": float(self.scale_overview_per_full),
            "scale_full_per_overview": float(1.0 / self.scale_overview_per_full),
        }
        meta_path = os.path.join(self.intermediate_dir, f"{self.vsi_base}_slice{scene_id+1:02d}_overview_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        debug(f"Wrote overview metadata: {meta_path}")

        self._update_header_ui()

    # ---------- Napari ----------
    def setup_viewer(self):
        file_name = os.path.basename(self.vsi_path)
        self.viewer = napari.Viewer(title=f"VSI ROI Tool — {file_name}")
        debug("Launching napari...")

        self._add_overview_layers()

        self.shapes = self.viewer.add_shapes(
            name="ROI",
            shape_type="polygon",
            edge_width=2,
        )
        try:
            self.shapes.blending = "translucent"
        except Exception:
            pass

        self._force_polygon_mode()
        self._ensure_roi_on_top(force_later=True)

        self.shapes.events.data.connect(self._on_shapes_data_changed)
        debug("Connected shapes data change event.")

        # Right dock: slice info (label photo + overview) + processing buttons
        panel = self._build_right_panel()
        self._dock_panel_ref = panel
        self.viewer.window.add_dock_widget(self._dock_panel_ref, name="ROI Controls", area="right")

        # Left dock: editing tools + ROI name fields
        tools = self._build_tools_panel()
        self._tools_panel_ref = tools
        self.viewer.window.add_dock_widget(self._tools_panel_ref, name="ROI Editor", area="left")

        debug("Dock widgets added.")
        debug("Napari launched.")

    def _force_polygon_mode(self):
        try:
            self.viewer.layers.selection.active = self.shapes
            self.shapes.mode = "add_polygon"
            self.shapes.current_shape_type = "polygon"
            debug("ROI layer set to Add Polygon mode by default.")
        except Exception:
            debug("Failed to force polygon mode (non-fatal).")

    def _ensure_roi_on_top(self, force_later: bool = False):
        def do_move():
            try:
                if self.viewer is None or self.shapes is None:
                    return
                layers = self.viewer.layers
                n_layers = len(layers)
                target_idx = n_layers  # <- LAST = top in your napari

                # Find current ROI layer index
                try:
                    idx = list(layers).index(self.shapes)
                except ValueError:
                    debug(f"ROI layer not found in viewer.layers (n_layers={n_layers}) (non-fatal).")
                    return

                debug(f"Layer stack: n_layers={n_layers}, ROI idx={idx}, target(top)={target_idx}")

                if idx != target_idx:
                    layers.move(idx, target_idx)
                    debug(f"Moved ROI layer to top (last): {idx} -> {target_idx}")
                else:
                    debug("ROI layer already at top (last index).")

            except Exception as e:
                debug(f"Could not move ROI layer to top (non-fatal): {repr(e)}")

        do_move()
        if force_later:
            QTimer.singleShot(0, do_move)
            QTimer.singleShot(50, do_move)
            QTimer.singleShot(200, do_move)



    def _sync_skip_button(self):
        if self.btn_skip is None:
            return
        n = len(self.shapes.data) if self.shapes is not None else 0
        self.btn_skip.setEnabled(n == 0)
        if self.is_last_scene:
            self.btn_skip.setText("Skip this slide (no ROI) → close window")
        else:
            self.btn_skip.setText("Skip this slide (no ROI) → next")

    def _update_header_ui(self):
        if self.viewer is None:
            return
        file_name = os.path.basename(self.vsi_path)
        self.viewer.title = f"VSI ROI Tool — {file_name} — Slice {self.scene_id + 1}/{self.total_scenes}"

        if self.lbl_file is not None:
            self.lbl_file.setText(f"<b>File:</b> {file_name}")
        if self.lbl_slice is not None:
            self.lbl_slice.setText(f"<b>Brain slice:</b> {self.scene_id + 1} / {self.total_scenes}")

        if self.btn_next is not None:
            self.btn_next.setEnabled(not self.is_last_scene)
        if self.btn_close is not None:
            self.btn_close.setEnabled(self.is_last_scene)
        self._sync_skip_button()

    def _set_busy(self, is_busy: bool, text: str = ""):
        if self.progress is not None:
            self.progress.setVisible(is_busy)
            if is_busy:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 1)
                self.progress.setValue(1)

        if self.lbl_status is not None:
            self.lbl_status.setText(text if text else "")
            self.lbl_status.setVisible(bool(text))

        try:
            QApplication.processEvents()
        except Exception:
            pass

    def _add_overview_layers(self):
        for layer in list(self.viewer.layers):
            if layer.name.startswith("overview_"):
                self.viewer.layers.remove(layer)

        ov = self.overview
        debug("Adding overview layers with autoscale contrast...")

        if ov.ndim == 2:
            lo, hi = np.percentile(ov, (1, 99.8))
            self.viewer.add_image(
                ov,
                name=f"overview_{self.vsi_base}_slice{self.scene_id+1:02d}",
                contrast_limits=(float(lo), float(hi)),
                colormap="gray",
            )
            return

        H, W, C = ov.shape
        debug(f"Overview has {C} channels, dtype={ov.dtype}")

        for c in range(C):
            ch = ov[..., c]
            lo, hi = np.percentile(ch, (1, 99.8))
            label = get_channel_label(self.scene, c)
            debug(f"{label} (ch{c}): p1={lo:.1f}, p99.8={hi:.1f}, min={ch.min()}, max={ch.max()}")

            self.viewer.add_image(
                ch,
                name=f"overview_{self.vsi_base}_slice{self.scene_id+1:02d}_{label}_ch{c}",
                contrast_limits=(float(lo), float(hi)),
                colormap="gray",
                blending="additive",
                visible=(c == 0),
            )

        comp = np.max(ov, axis=-1)
        lo, hi = np.percentile(comp, (1, 99.8))
        self.viewer.add_image(
            comp,
            name=f"overview_{self.vsi_base}_slice{self.scene_id+1:02d}_composite_max",
            contrast_limits=(float(lo), float(hi)),
            colormap="gray",
            visible=True,
        )

    # ---------- Slide label / scene overview ----------
    def _build_scene_overview_img(self) -> "np.ndarray | None":
        """
        Render a small (~DOCK_IMG_WIDTH px wide) composite image of all high-res
        scenes placed at their physical positions, with a red rectangle around the
        currently loaded scene.  Uses slideio read_block() at thumbnail resolution.
        Returns an (H, W, 3) uint8 RGB array, or None on failure.
        """
        if not self._scene_coords or self.total_scenes == 0:
            return None
        try:
            from PIL import Image as _PIL, ImageDraw as _Draw

            # Collect physical bounds for every scene that has coord metadata
            scene_bounds = []
            for i in range(self.total_scenes):
                s = self.slide.get_scene(i)
                p = self._scene_coords.get(s.name)
                if p is None:
                    continue
                w_px, h_px = s.size
                x0, y0 = p["ox"], p["oy"]
                x1 = x0 + w_px * p["sx"]
                y1 = y0 + h_px * p["sy"]
                scene_bounds.append(
                    {"i": i, "name": s.name, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
                )

            if not scene_bounds:
                return None

            all_x0 = min(b["x0"] for b in scene_bounds)
            all_y0 = min(b["y0"] for b in scene_bounds)
            all_x1 = max(b["x1"] for b in scene_bounds)
            all_y1 = max(b["y1"] for b in scene_bounds)

            phys_w = all_x1 - all_x0
            phys_h = all_y1 - all_y0
            if phys_w <= 0 or phys_h <= 0:
                return None

            canvas_w = DOCK_IMG_WIDTH
            px_per_um = canvas_w / phys_w
            canvas_h = max(1, int(phys_h * px_per_um))

            canvas = np.full((canvas_h, canvas_w, 3), 25, dtype=np.uint8)

            for b in scene_bounds:
                x0_c = int((b["x0"] - all_x0) * px_per_um)
                y0_c = int((b["y0"] - all_y0) * px_per_um)
                x1_c = max(x0_c + 1, int((b["x1"] - all_x0) * px_per_um))
                y1_c = max(y0_c + 1, int((b["y1"] - all_y0) * px_per_um))
                tw = x1_c - x0_c
                th = y1_c - y0_c

                scene = self.slide.get_scene(b["i"])
                img_data = scene.read_block(size=(tw, th))

                # Use first channel (DAPI / brightest) for grayscale composite
                ch0 = (img_data[:, :, 0] if img_data.ndim == 3 else img_data).astype(np.float32)
                nonzero = ch0[ch0 > 0]
                p99 = float(np.percentile(nonzero, 99)) if nonzero.size > 0 else 1.0
                ch0_u8 = np.clip(ch0 / max(p99, 1.0), 0.0, 1.0)
                ch0_u8 = (ch0_u8 * 210).astype(np.uint8)

                ph = min(ch0_u8.shape[0], canvas_h - y0_c)
                pw = min(ch0_u8.shape[1], canvas_w - x0_c)
                if ph > 0 and pw > 0:
                    canvas[y0_c : y0_c + ph, x0_c : x0_c + pw] = (
                        np.stack([ch0_u8[:ph, :pw]] * 3, axis=-1)
                    )

            # Draw red rectangle for the current scene
            cur_name = self.scene.name if self.scene is not None else ""
            pil_img = _PIL.fromarray(canvas)
            draw = _Draw.Draw(pil_img)
            for b in scene_bounds:
                if b["name"] == cur_name:
                    x0_c = int((b["x0"] - all_x0) * px_per_um)
                    y0_c = int((b["y0"] - all_y0) * px_per_um)
                    x1_c = max(x0_c + 1, int((b["x1"] - all_x0) * px_per_um))
                    y1_c = max(y0_c + 1, int((b["y1"] - all_y0) * px_per_um))
                    for t in range(3):
                        draw.rectangle(
                            [x0_c - t, y0_c - t, x1_c + t, y1_c + t],
                            outline=(255, 50, 50),
                        )
                    break

            return np.array(pil_img)

        except Exception as exc:
            debug(f"Scene overview render failed: {exc}")
            return None

    def _update_overview_display(self):
        """Redraw the scene-overview QLabel with the current scene highlighted."""
        if self._overview_lbl_widget is None:
            return
        try:
            ov_arr = self._build_scene_overview_img()
            if ov_arr is not None:
                pix = numpy_to_qpixmap(ov_arr)
                self._overview_lbl_widget.setPixmap(
                    pix.scaledToWidth(DOCK_IMG_WIDTH, Qt.SmoothTransformation)
                )
        except Exception as exc:
            debug(f"Overview display update failed: {exc}")

    # ---------- Right panel ----------
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ── Slide label image ─────────────────────────────────────────────────
        if self._label_img is not None:
            try:
                label_box = QGroupBox("Slide Label")
                label_box_layout = QVBoxLayout(label_box)
                label_box_layout.setContentsMargins(4, 4, 4, 4)
                lbl_pix = QLabel()
                lbl_pix.setAlignment(Qt.AlignCenter)
                pix = numpy_to_qpixmap(self._label_img)
                lbl_pix.setPixmap(pix.scaledToWidth(DOCK_IMG_WIDTH, Qt.SmoothTransformation))
                label_box_layout.addWidget(lbl_pix)
                layout.addWidget(label_box)
            except Exception as exc:
                debug(f"Label image widget failed: {exc}")

        # ── Scene overview (all slices, current highlighted) ──────────────────
        ov_arr = self._build_scene_overview_img()
        if ov_arr is not None:
            try:
                ov_box = QGroupBox("Slide Overview  (red = current slice)")
                ov_box_layout = QVBoxLayout(ov_box)
                ov_box_layout.setContentsMargins(4, 4, 4, 4)
                self._overview_lbl_widget = QLabel()
                self._overview_lbl_widget.setAlignment(Qt.AlignCenter)
                pix = numpy_to_qpixmap(ov_arr)
                self._overview_lbl_widget.setPixmap(
                    pix.scaledToWidth(DOCK_IMG_WIDTH, Qt.SmoothTransformation)
                )
                ov_box_layout.addWidget(self._overview_lbl_widget)
                layout.addWidget(ov_box)
            except Exception as exc:
                debug(f"Scene overview widget failed: {exc}")

        self.lbl_file = QLabel()
        self.lbl_slice = QLabel()
        layout.addWidget(self.lbl_file)
        layout.addWidget(self.lbl_slice)

        self.lbl_status = QLabel("")
        self.lbl_status.setVisible(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress)

        name_hint = QLabel("⬅  Name EVERY ROI in the bottom-LEFT panel\n     before exporting / advancing.")
        name_hint.setWordWrap(True)
        name_hint.setStyleSheet(
            "color: #ffcc00; font-weight: bold; padding: 4px;"
            "border: 1px solid #ffcc00; border-radius: 4px;"
        )
        layout.addWidget(name_hint)

        proc_box = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_box)

        self.btn_next = QPushButton("Save masks + EXPORT ALL ROIs + NEXT slice")
        self.btn_next.clicked.connect(safe_slot(self.save_export_all_and_next_scene))
        proc_layout.addWidget(self.btn_next)

        self.btn_close = QPushButton("Save masks + EXPORT ALL ROIs + CLOSE window")
        self.btn_close.clicked.connect(safe_slot(self.save_export_all_and_close))
        proc_layout.addWidget(self.btn_close)

        self.btn_skip = QPushButton("Skip this slide (no ROI) → next")
        self.btn_skip.setToolTip("Advance to the next slide without drawing any ROI or exporting.")
        self.btn_skip.clicked.connect(safe_slot(self.skip_and_next_scene))
        proc_layout.addWidget(self.btn_skip)

        layout.addWidget(proc_box)

        if SHOW_DEBUG_PANEL:
            dbg_box = QGroupBox("Debug")
            dbg_layout = QVBoxLayout(dbg_box)

            self.btn_debug = QPushButton("DEBUG: Save masks + export ALL ROIs (print paths)")
            self.btn_debug.clicked.connect(safe_slot(self.debug_save_and_export_all_now))
            dbg_layout.addWidget(self.btn_debug)

            layout.addWidget(dbg_box)

        layout.addStretch(1)

        self._update_header_ui()
        return panel

    def _build_tools_panel(self) -> QWidget:
        """Left-dock panel: Editing buttons + ROI Names scroll area."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        edit_box = QGroupBox("Editing")
        edit_layout = QVBoxLayout(edit_box)

        btn_del_last = QPushButton("Delete last ROI")
        btn_del_last.clicked.connect(safe_slot(self.delete_last_polygon))
        edit_layout.addWidget(btn_del_last)

        btn_del_all = QPushButton("Delete ALL ROIs (reset)")
        btn_del_all.clicked.connect(safe_slot(self.delete_all_polygons))
        edit_layout.addWidget(btn_del_all)

        layout.addWidget(edit_box)

        name_box = QGroupBox("★ ROI Names (REQUIRED — one per polygon) ★")
        name_box.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #ffcc00; "
            "border: 1px solid #ffcc00; border-radius: 4px; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        name_layout = QVBoxLayout(name_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        self.roi_name_layout = QVBoxLayout(container)
        self.roi_name_layout.setContentsMargins(4, 4, 4, 4)
        self.roi_name_layout.setSpacing(6)

        scroll.setWidget(container)
        name_layout.addWidget(scroll)
        layout.addWidget(name_box)

        layout.addStretch(1)

        self._sync_roi_ui()
        return panel

    # ---------- ROI UI syncing ----------
    def _clear_roi_name_rows(self):
        if self.roi_name_layout is None:
            return
        for w in self._roi_rows:
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception:
                pass
        self._roi_rows = []

    def _apply_per_roi_colors_to_shapes(self):
        try:
            n = len(self.shapes.data)
            if n == 0:
                return
            while len(self.roi_colors) < n:
                self.roi_colors.append(self._pick_roi_color(len(self.roi_colors)))
            self.roi_colors = self.roi_colors[:n]

            edge_colors = []
            face_colors = []
            for rgba in self.roi_colors:
                edge_colors.append(rgba)
                face_colors.append((rgba[0], rgba[1], rgba[2], 0.15))
            self.shapes.edge_color = edge_colors
            self.shapes.face_color = face_colors
        except Exception:
            debug("Failed to apply per-ROI colors (non-fatal).")
            traceback.print_exc()

    def _sync_roi_ui(self):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            n = len(self.shapes.data) if self.shapes is not None else 0

            if len(self.roi_names) < n:
                self.roi_names.extend([""] * (n - len(self.roi_names)))
            self.roi_names = self.roi_names[:n]

            if len(self.roi_colors) < n:
                for i in range(len(self.roi_colors), n):
                    self.roi_colors.append(self._pick_roi_color(i))
            self.roi_colors = self.roi_colors[:n]

            self._clear_roi_name_rows()
            if self.roi_name_layout is None:
                return

            for idx in range(n):
                rgba = self.roi_colors[idx]
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                swatch = QLabel("")
                swatch.setFixedSize(18, 18)
                swatch.setStyleSheet(
                    f"background-color: {self._rgba_to_css(rgba)};"
                    "border: 1px solid #444;"
                )
                row_layout.addWidget(swatch)

                lbl = QLabel(f"ROI {idx+1}:")
                lbl.setFixedWidth(50)
                row_layout.addWidget(lbl)

                edit = QLineEdit()
                edit.setPlaceholderText("optional suffix (e.g., CA1_left)")
                edit.setText(self.roi_names[idx])
                edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                def on_text_changed(text, i=idx):
                    self.roi_names[i] = text.strip()
                    debug(f"ROI {i+1} name set to: '{self.roi_names[i]}'")

                edit.textChanged.connect(on_text_changed)
                row_layout.addWidget(edit)

                self.roi_name_layout.addWidget(row)
                self._roi_rows.append(row)

            if self.shapes is not None:
                self._apply_per_roi_colors_to_shapes()

            self._sync_skip_button()

        finally:
            self._syncing_ui = False

    def _on_shapes_data_changed(self, event=None):
        debug(f"Shapes changed. ROI count now = {len(self.shapes.data)}")
        self._sync_roi_ui()
        self._ensure_roi_on_top(force_later=True)
        self._force_polygon_mode()

    # ---------- ROI geometry ----------
    def _get_polygon_by_index(self, idx: int):
        if self.shapes is None or idx < 0 or idx >= len(self.shapes.data):
            return None
        poly = np.array(self.shapes.data[idx], dtype=float)
        if poly.shape[0] < 3:
            return None
        return poly

    def _roi_bbox_full_from_polygon(self, poly_overview_yx):
        y0, x0, y1, x1 = polygon_to_bbox_yx(poly_overview_yx)

        H_ov, W_ov = self.overview.shape[0], self.overview.shape[1]
        y0 = clamp(y0, 0, H_ov - 1)
        y1 = clamp(y1, 0, H_ov - 1)
        x0 = clamp(x0, 0, W_ov - 1)
        x1 = clamp(x1, 0, W_ov - 1)

        s = self.scale_overview_per_full
        fx0 = int(np.floor(x0 / s))
        fy0 = int(np.floor(y0 / s))
        fx1 = int(np.ceil(x1 / s))
        fy1 = int(np.ceil(y1 / s))

        W_full, H_full = self.scene.size
        fx0 = clamp(fx0, 0, W_full - 1)
        fx1 = clamp(fx1, 0, W_full - 1)
        fy0 = clamp(fy0, 0, H_full - 1)
        fy1 = clamp(fy1, 0, H_full - 1)

        fw = max(1, fx1 - fx0 + 1)
        fh = max(1, fy1 - fy0 + 1)

        return fx0, fy0, fw, fh

    # ---------- Delete buttons ----------
    def delete_last_polygon(self):
        debug("Delete last ROI pressed.")
        if len(self.shapes.data) == 0:
            debug("No ROIs to delete.")
            return
        self.shapes.data = self.shapes.data[:-1]
        self._force_polygon_mode()
        self._ensure_roi_on_top(force_later=True)

    def delete_all_polygons(self):
        debug("Delete ALL ROIs pressed.")
        n = len(self.shapes.data)
        if n > 0:
            msg = QMessageBox(self.viewer.window._qt_window)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Confirm: delete ALL ROIs")
            msg.setText(
                f"This will permanently delete all {n} ROI(s) on this slice.\n\n"
                "This cannot be undone. Are you sure?"
            )
            msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
            msg.setDefaultButton(QMessageBox.Cancel)
            if msg.exec_() != QMessageBox.Yes:
                debug("Delete ALL ROIs cancelled by user.")
                return
        self.shapes.data = []
        self.roi_names = []
        self.roi_colors = []
        self._sync_roi_ui()
        self._force_polygon_mode()
        self._ensure_roi_on_top(force_later=True)

    # ---------- Save masks ----------
    def save_all_roi_masks_to_disk(self):
        n = len(self.shapes.data)
        if n == 0:
            debug("No ROIs to save.")
            return []

        H_ov, W_ov = self.overview.shape[0], self.overview.shape[1]
        slice_idx = self.scene_id + 1
        debug(f"Saving ALL ROI masks for slice {slice_idx}... count={n} overview H,W={H_ov},{W_ov}")

        saved = []
        for i in range(n):
            poly = self._get_polygon_by_index(i)
            if poly is None:
                debug(f"ROI {i+1}: invalid. Skipping.")
                continue

            suffix = self.roi_names[i].strip() if i < len(self.roi_names) else ""
            suffix_part = f"_{suffix}" if suffix else ""

            verts_path = os.path.join(
                self.mask_dir,
                f"{self.vsi_base}_slice{slice_idx:02d}_roi{i+1:02d}_vertices_overview_yx.npy"
            )
            np.save(verts_path, poly)

            coords_xy = [[float(x), float(y)] for y, x in poly]
            if coords_xy[0] != coords_xy[-1]:
                coords_xy.append(coords_xy[0])
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "vsi_base": self.vsi_base,
                        "slice_one_indexed": int(slice_idx),
                        "roi_one_indexed": int(i + 1),
                        "suffix": suffix,
                        "space": "overview_pixels"
                    },
                    "geometry": {"type": "Polygon", "coordinates": [coords_xy]}
                }]
            }
            geo_path = os.path.join(
                self.mask_dir,
                f"{self.vsi_base}_slice{slice_idx:02d}_roi{i+1:02d}{suffix_part}_overview.geojson"
            )
            with open(geo_path, "w") as f:
                json.dump(geojson, f, indent=2)

            mask = rasterize_polygon_mask(poly, (H_ov, W_ov))
            mask_vis = (mask * 255).astype(np.uint8)
            mask_path = os.path.join(
                self.mask_dir,
                f"{self.vsi_base}_slice{slice_idx:02d}_roi{i+1:02d}{suffix_part}_mask_overview.tif"
            )
            tiff.imwrite(mask_path, mask_vis, photometric="minisblack", compression=None)

            debug(f"Saved ROI {i+1}: {os.path.basename(mask_path)}")
            saved.append({"roi_idx": i, "mask_path": mask_path})
        return saved

    # ---------- Export full-res TIFFs ----------
    def export_all_rois_fullres(self, debug_tag: str = ""):
        n = len(self.shapes.data)
        if n == 0:
            debug("No ROIs to export.")
            return []

        slice_idx = self.scene_id + 1
        tag = f"{debug_tag}_" if debug_tag else ""

        exported = []
        for i in range(n):
            poly = self._get_polygon_by_index(i)
            if poly is None:
                debug(f"ROI {i+1}: invalid. Skipping export.")
                continue

            fx0, fy0, fw, fh = self._roi_bbox_full_from_polygon(poly)

            suffix = self.roi_names[i].strip() if i < len(self.roi_names) else ""
            suffix_part = f"_{suffix}" if suffix else ""

            out_path = os.path.join(
                self.export_dir,
                f"{tag}{self.vsi_base}_slice{slice_idx:02d}_roi{i+1:02d}_FULLRES{suffix_part}.tif"
            )

            debug(f"Export ROI {i+1}: full XYWH=({fx0},{fy0},{fw},{fh}) -> {out_path}")
            img = self.scene.read_block((fx0, fy0, fw, fh))
            debug(f"ROI {i+1} read OK: dtype={img.dtype}, shape={img.shape}")

            write_multichannel_tiff(out_path, img)
            exported.append({"roi_idx": i, "tiff_path": out_path})
        return exported

    # ---------- ROI naming validation ----------
    def _unnamed_roi_indices(self) -> list:
        """Return 1-based indices of ROIs that have no (non-blank) name."""
        n = len(self.shapes.data)
        unnamed = []
        for i in range(n):
            name = self.roi_names[i].strip() if i < len(self.roi_names) else ""
            if not name:
                unnamed.append(i + 1)
        return unnamed

    def _warn_unnamed_rois(self) -> bool:
        """
        If any ROI is unnamed, pop a warning and return True (caller should abort).
        Returns False when every ROI has a name.
        """
        unnamed = self._unnamed_roi_indices()
        if not unnamed:
            return False
        idx_str = ", ".join(str(i) for i in unnamed)
        QMessageBox.warning(
            self.viewer.window._qt_window,
            "Name every ROI first",
            f"ROI {idx_str} {'has' if len(unnamed) == 1 else 'have'} no name.\n\n"
            "⬅  Enter a name for EVERY ROI in the \"ROI Names\" panel on the "
            "BOTTOM-LEFT of the window before continuing.\n\n"
            "Nothing was saved or exported."
        )
        return True

    # ---------- Buttons ----------
    def debug_save_and_export_all_now(self):
        if len(self.shapes.data) == 0:
            debug("No ROIs drawn. Nothing to debug-export.")
            return

        self._set_busy(True, "DEBUG: saving masks + exporting TIFFs…")
        try:
            self.save_all_roi_masks_to_disk()
            self.export_all_rois_fullres(debug_tag="DEBUG")
            debug("DEBUG done. (Not loading any debug images into napari.)")
        finally:
            self._set_busy(False, "")

    def save_export_all_and_next_scene(self):
        debug("Next slice requested.")
        if self.is_last_scene:
            debug("Already on last slice. NEXT disabled.")
            return
        if len(self.shapes.data) == 0:
            debug("No ROI polygon found. Draw at least one polygon first.")
            return

        if self._warn_unnamed_rois():
            debug("Next slice blocked: one or more ROIs unnamed.")
            return

        msg = QMessageBox(self.viewer.window._qt_window)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Confirm: proceed to next brain slice")
        msg.setText(
            "This will:\n"
            "  1) Save ALL ROI masks\n"
            "  2) Export ALL ROI full-resolution TIFFs\n"
            "  3) Load the NEXT brain slice\n\n"
            "You cannot go back."
        )
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        ret = msg.exec_()
        if ret != QMessageBox.Ok:
            debug("User cancelled next slice.")
            return

        self._set_busy(True, "Saving masks + exporting TIFFs… (please wait)")
        try:
            self.save_all_roi_masks_to_disk()
            self.export_all_rois_fullres(debug_tag="")

            next_id = self.scene_id + 1
            self._set_busy(True, "Loading next brain slice…")
            debug(f"Switching from slice {self.scene_id+1} -> {next_id+1}")

            self.load_scene(next_id)
            self._add_overview_layers()
            self._update_overview_display()

            # reset ROI layer
            self.shapes.data = []
            self.roi_names = []
            self.roi_colors = []
            self._sync_roi_ui()

            self._force_polygon_mode()
            self._ensure_roi_on_top(force_later=True)

        finally:
            self._set_busy(False, "")

    def save_export_all_and_close(self):
        debug("Save & Close requested.")
        if not self.is_last_scene:
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "Not last slice",
                "Save & Close is only enabled on the last brain slice."
            )
            return
        if len(self.shapes.data) == 0:
            QMessageBox.information(
                self.viewer.window._qt_window,
                "No ROIs",
                "No ROI polygons found. Draw at least one polygon before closing."
            )
            return

        if self._warn_unnamed_rois():
            debug("Save & Close blocked: one or more ROIs unnamed.")
            return

        msg = QMessageBox(self.viewer.window._qt_window)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Confirm: Save & Close")
        msg.setText(
            "This will:\n"
            "  1) Save ALL ROI masks\n"
            "  2) Export ALL ROI full-resolution TIFFs\n"
            "  3) Close the napari window\n"
        )
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        ret = msg.exec_()
        if ret != QMessageBox.Ok:
            debug("User cancelled Save & Close.")
            return

        self._set_busy(True, "Saving masks + exporting TIFFs… then closing…")
        try:
            self.save_all_roi_masks_to_disk()
            self.export_all_rois_fullres(debug_tag="")
            debug("All saving/export finished. Closing napari window now...")

            try:
                QApplication.processEvents()
            except Exception:
                pass

            self.viewer.window._qt_window.close()

        finally:
            self._set_busy(False, "")

    def skip_and_next_scene(self):
        debug("Skip slide requested.")
        if len(self.shapes.data) > 0:
            debug("ROI exists — skip button should be disabled.")
            return

        parent_win = self.viewer.window._qt_window

        if self.is_last_scene:
            # Last slide — confirm then close without saving
            msg = QMessageBox(parent_win)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Skip last slide and close?")
            msg.setText(
                f"Skip slice {self.scene_id + 1} of {self.total_scenes} with NO ROI?\n\n"
                "Nothing will be saved for this slide.\n"
                "The window will close."
            )
            msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
            msg.setDefaultButton(QMessageBox.Cancel)
            if msg.exec_() != QMessageBox.Ok:
                debug("Skip+close cancelled.")
                return
            debug(f"Skipping last slice {self.scene_id + 1} — closing window.")
            self.viewer.window._qt_window.close()
            return

        # Not last slide — two-step confirmation before advancing
        msg1 = QMessageBox(parent_win)
        msg1.setIcon(QMessageBox.Warning)
        msg1.setWindowTitle("Skip this slide?")
        msg1.setText(
            "Are you sure you want to skip this slide with NO ROI selection?\n\n"
            "No masks or TIFFs will be exported for this slide.\n"
            "You cannot return back."
        )
        msg1.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
        msg1.setDefaultButton(QMessageBox.Cancel)
        if msg1.exec_() != QMessageBox.Yes:
            debug("Skip cancelled at first dialog.")
            return

        msg2 = QMessageBox(parent_win)
        msg2.setIcon(QMessageBox.Critical)
        msg2.setWindowTitle("Final confirmation — skip slide")
        msg2.setText(
            f"FINAL CONFIRMATION\n\n"
            f"Slice {self.scene_id + 1} of {self.total_scenes} will be skipped entirely.\n"
            f"Nothing will be saved for this slide.\n\n"
            f"Proceed to slice {self.scene_id + 2}?"
        )
        msg2.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg2.setDefaultButton(QMessageBox.Cancel)
        if msg2.exec_() != QMessageBox.Ok:
            debug("Skip cancelled at second dialog.")
            return

        debug(f"Skipping slice {self.scene_id + 1} — no export.")
        self._set_busy(True, f"Skipping slice {self.scene_id + 1}…")
        try:
            next_id = self.scene_id + 1
            self.load_scene(next_id)
            self._add_overview_layers()
            self._update_overview_display()
            self.shapes.data = []
            self.roi_names = []
            self.roi_colors = []
            self._sync_roi_ui()
            self._force_polygon_mode()
            self._ensure_roi_on_top(force_later=True)
        finally:
            self._set_busy(False, "")

    def run_one_file(self):
        self.open_slide()
        self.load_scene(START_SCENE_ID)
        self.setup_viewer()


# ----------------------------
# IMPROVED PICKER UX
# ----------------------------
def choose_vsi_or_folder_dialog(parent=None):
    """
    Ask user: choose a VSI file OR choose a folder.
    Returns ('file', path) or ('folder', path) or ('', '') if cancelled.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Select input")
    box.setText("What would you like to open?")
    btn_file = box.addButton("Open a slide-scanner image (VSI / LIF)", QMessageBox.AcceptRole)
    btn_folder = box.addButton("Set working directory (folder)", QMessageBox.AcceptRole)
    btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(btn_file)
    box.exec_()

    clicked = box.clickedButton()
    if clicked == btn_cancel:
        return "", ""
    if clicked == btn_file:
        return "file", ""
    if clicked == btn_folder:
        return "folder", ""
    return "", ""


def pick_folder(start_dir: str) -> str:
    debug(f"Opening folder picker at: {start_dir}")

    dlg = QFileDialog(None, "Set working directory (folder)", start_dir)
    dlg.setFileMode(QFileDialog.Directory)
    dlg.setOption(QFileDialog.ShowDirsOnly, True)
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)  # <- IMPORTANT on mac
    dlg.setModal(True)

    if dlg.exec_():
        folders = dlg.selectedFiles()
        if folders:
            debug(f"Folder selected: {folders[0]}")
            return folders[0]

    debug("Folder picker cancelled.")
    return ""


def pick_vsi_file(start_dir: str) -> str:
    debug(f"Opening slide-scanner image picker at: {start_dir}")

    dlg = QFileDialog(None, "Select a slide-scanner image (VSI / LIF)", start_dir)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter("Slide-scanner images (*.vsi *.lif);;Olympus VSI (*.vsi);;Leica LIF (*.lif)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)  # <- IMPORTANT on mac
    dlg.setModal(True)

    if dlg.exec_():
        files = dlg.selectedFiles()
        if files:
            debug(f"Slide-scanner image selected: {files[0]}")
            return files[0]

    debug("Slide-scanner image picker cancelled.")
    return ""



# ----------------------------
# MAIN LOOP
# ----------------------------


def _show_error_dialog(title: str, message: str):
    """Show a Qt critical dialog with the error. Safe to call before napari starts."""
    try:
        app = QApplication.instance() or QApplication([])
        from qtpy.QtWidgets import QTextEdit, QDialog, QDialogButtonBox
        dlg = QDialog()
        dlg.setWindowTitle(title)
        dlg.resize(700, 400)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("An error occurred. Please copy the message below and report it."))
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(message)
        te.setFontFamily("Courier")
        layout.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb)
        dlg.exec_()
    except Exception:
        print(message, flush=True)


def main():
    # Optional: suppress only FutureWarnings (keeps real errors visible)
    warnings.filterwarnings("ignore", category=FutureWarning)

    def _excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, flush=True)
        _show_error_dialog("CellMatchR VSI Converter — Unhandled Error", msg)

    sys.excepthook = _excepthook

    output_dir = OUTPUT_DIR_DEFAULT
    ensure_dir(output_dir)

    last_dir = os.path.dirname(output_dir) if os.path.isdir(os.path.dirname(output_dir)) else os.path.expanduser("~")

    def pick_next():
        nonlocal last_dir

        mode, _ = choose_vsi_or_folder_dialog()
        if not mode:
            debug("Selection dialog cancelled. Exiting app.")
            QApplication.quit()
            return

        if mode == "folder":
            folder = pick_folder(start_dir=last_dir)
            if not folder:
                debug("Folder picker cancelled. Returning to selection menu.")
                QTimer.singleShot(0, pick_next)
                return
            last_dir = folder

            vsi_path = pick_vsi_file(start_dir=folder)
            if not vsi_path:
                debug("VSI picker cancelled. Returning to selection menu.")
                QTimer.singleShot(0, pick_next)
                return
            last_dir = os.path.dirname(vsi_path)

        else:  # mode == "file"
            vsi_path = pick_vsi_file(start_dir=last_dir)
            if not vsi_path:
                debug("VSI picker cancelled. Returning to selection menu.")
                QTimer.singleShot(0, pick_next)
                return
            last_dir = os.path.dirname(vsi_path)

        debug(f"Chosen VSI: {vsi_path}")

        # Leica writes both confocal and slide-scanner data as .lif. Warn
        # (non-blocking) if this particular file looks like pure confocal
        # data -- it belongs in Confocal direct-read (Step 2), not here.
        # Advisory only: any classification failure is swallowed and the
        # file opens exactly as before. VSI files are never affected.
        if vsi_path.lower().endswith(".lif"):
            try:
                from cellmatchr.lif_classify import classify_lif_file
                if classify_lif_file(vsi_path).get("file_type") == "confocal":
                    box = QMessageBox()
                    box.setIcon(QMessageBox.Warning)
                    box.setWindowTitle("This looks like confocal data")
                    box.setText(
                        f"'{os.path.basename(vsi_path)}' has no slide-scan / tile-scan "
                        "images — it looks like confocal data, not a slide scan.\n\n"
                        "Continue opening it here anyway, or cancel and use "
                        "Confocal mode (Step 2) instead?"
                    )
                    box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
                    box.setDefaultButton(QMessageBox.Cancel)
                    if box.exec_() != QMessageBox.Yes:
                        debug("User cancelled after confocal-file warning. Returning to picker.")
                        QTimer.singleShot(0, pick_next)
                        return
            except Exception:
                debug("LIF classification check failed (non-fatal) — opening file as usual.")

        try:
            app = VSINapariROIApp(vsi_path=vsi_path, output_dir=output_dir)
            app.run_one_file()

            # When napari closes, come back to picker
            qtwin = app.viewer.window._qt_window
            qtwin.destroyed.connect(lambda: QTimer.singleShot(0, pick_next))

        except Exception:
            debug("Fatal error while processing VSI:")
            msg = traceback.format_exc()
            print(msg, flush=True)
            _show_error_dialog("CellMatchR VSI Converter — Error", msg)
            QTimer.singleShot(0, pick_next)

    # Make sure a Qt app exists (needed in VS Code interactive sometimes)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Keep the process alive after a napari window closes so we can return to
    # the file picker for the NEXT VSI file. Without this Qt quits on the last
    # window closing and the scheduled pick_next() never runs. The app is only
    # explicitly quit when the user cancels the selection dialog (see pick_next).
    app.setQuitOnLastWindowClosed(False)

    # Start picker after Qt loop starts
    QTimer.singleShot(0, pick_next)

    napari.run(force=True)



def _apply_config(config_path: str) -> None:
    """Override OUTPUT_DIR_DEFAULT from a CellMatchr config file."""
    import json
    global OUTPUT_DIR_DEFAULT
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        vsi_out = cfg.get("vsi_output_dir", "").strip()
        if vsi_out:
            OUTPUT_DIR_DEFAULT = vsi_out
            debug(f"[config] OUTPUT_DIR_DEFAULT set to: {OUTPUT_DIR_DEFAULT}")
    except Exception as e:
        debug(f"[config] Could not load {config_path}: {e}")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="CellMatchR VSI Converter")
    _parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.cellmatchr/config.json"),
        help="Path to CellMatchr config JSON",
    )
    _args, _ = _parser.parse_known_args()
    if os.path.exists(_args.config):
        _apply_config(_args.config)
    main()
