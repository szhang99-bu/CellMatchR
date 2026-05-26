# VSI file processing code, manually select a small region of the brain to create TIF
# Shiyu Zhang
# 02/01/2026

import os
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
from qtpy.QtCore import QTimer, QEventLoop


# ----------------------------
# USER SETTINGS
# ----------------------------
OUTPUT_DIR_DEFAULT = "/Volumes/tusm_bygravelab01$/Confocal data archive/Matt/Temp for delete"

OVERVIEW_WIDTH = 2500          # overview width (navigation only)
START_SCENE_ID = 0             # start slice (0-index internally)
MASK_DTYPE = np.uint8          # saved mask dtype (0/255 preview mask)
SHOW_DEBUG_PANEL = False


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
        debug(f"Opening VSI with slideio: {self.vsi_path}")
        self.slide = slideio.open_slide(self.vsi_path, "VSI")
        debug("Opened VSI successfully.")
        self.total_scenes = self.slide.num_scenes
        debug(f"Number of scenes: {self.total_scenes}")
        for i in range(self.total_scenes):
            s = self.slide.get_scene(i)
            debug(f"Scene {i}: size=({s.size[0]} x {s.size[1]}), channels={s.num_channels}")

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

        panel = self._build_right_panel()
        self._dock_panel_ref = panel
        self.viewer.window.add_dock_widget(self._dock_panel_ref, name="ROI Controls", area="right")
        debug("Dock widget added and reference stored.")
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

    # ---------- Right panel ----------
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

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

        edit_box = QGroupBox("Editing")
        edit_layout = QVBoxLayout(edit_box)

        btn_del_last = QPushButton("Delete last ROI")
        btn_del_last.clicked.connect(safe_slot(self.delete_last_polygon))
        edit_layout.addWidget(btn_del_last)

        btn_del_all = QPushButton("Delete ALL ROIs (reset)")
        btn_del_all.clicked.connect(safe_slot(self.delete_all_polygons))
        edit_layout.addWidget(btn_del_all)

        layout.addWidget(edit_box)

        name_box = QGroupBox("ROI Names (one per polygon)")
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
    btn_file = box.addButton("Open a VSI file", QMessageBox.AcceptRole)
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
    debug(f"Opening VSI picker at: {start_dir}")

    dlg = QFileDialog(None, "Select a VSI file", start_dir)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter("VSI files (*.vsi)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)  # <- IMPORTANT on mac
    dlg.setModal(True)

    if dlg.exec_():
        files = dlg.selectedFiles()
        if files:
            debug(f"VSI selected: {files[0]}")
            return files[0]

    debug("VSI picker cancelled.")
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
