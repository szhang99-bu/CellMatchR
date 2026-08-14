"""
parameter_wizard.py
====================
Stage 2 (napari display) of the Parameter Wizard: shows the raw channel
image and its unfiltered segmentation -- produced headlessly by
parameter_wizard_compute.py, which the launcher always runs first -- and
lets you pick example objects to auto-generate CellMatchR's object-
filtering thresholds instead of guessing numbers.

This file deliberately has NO import of cellmatchr.segmentation, cv2,
torch, or micro_sam. Combining the segmentation stack with a live napari
window in one process was observed to crash the interpreter on this
machine, so the heavy computation and the GUI are two separate processes
-- the same split segmentation.py (headless) and roi_editor.py (napari-
only) already use elsewhere in this codebase. This script only ever
reads the raw/label TIFFs and manifest that stage 1 wrote; it never runs
segmentation itself.

What it does
------------
For each channel in the manifest, opens a napari window showing:
  - the raw channel image, unadjusted (no percentile stretch, no gamma)
    -- napari's own contrast slider is there if you want to look closer,
    but every picked statistic always comes from the true raw pixel
    values, never the display adjustment
  - every unfiltered detected object as an OUTLINE only (contour mode),
    so a picked object never hides what's underneath or next to it

A right-hand panel has four "Pick ___" buttons. Click one, then click the
matching object's outline in the image:
  - Smallest valid cell                      -> minimum object size
  - Biggest valid cell                       -> maximum object size
  - Least round (most elongated) valid cell  -> max major/minor ratio
  - Faintest valid cell                      -> minimum mean intensity
Leaving any of the four unset means "no filter" for that criterion --
matches CellMatchR's existing "0 = disabled" convention.

Every threshold gets a safety margin (see MARGIN_FRACTION) so eyeballing
the 2nd- or 3rd-most-extreme object instead of the literal extreme still
keeps the true extreme inside the kept range.

Closing the window (or clicking "Save & Next Channel" / "Save & Finish")
writes the results into seg_per_channel_filters for every channel
actually visited, and enables per-channel filtering. Channels never
reached (if you close early) are left alone -- they fall back to
whatever the plain global filter fields already say.

Launched from the CellMatchR launcher (Step 3 -> "Parameter Wizard"),
after parameter_wizard_compute.py has produced a manifest. Can also be
run standalone once a manifest exists:

    python parameter_wizard.py --config ~/.cellmatchr/config.json --manifest /path/to/parameter_wizard_manifest.json
"""

import argparse
import json
import os
import sys

# This script runs as a standalone subprocess (its own sys.path[0] is
# cellmatchr/, not the project root), so the cellmatchr package isn't
# importable unless the project root is added first -- same fix
# scripts/run_cellmatchr.py uses for the same reason.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import tifffile
from skimage.measure import regionprops_table

from cellmatchr.config import DEFAULT_CONFIG, save_config, save_project_config, CONFIG_PATH

# Every picked threshold is padded by this fraction so a by-eye pick of the
# 2nd/3rd-most-extreme object doesn't accidentally cut off the actual
# most-extreme one the user meant to keep.
MARGIN_FRACTION = 0.15

PICK_ORDER = ["min_size", "max_size", "max_ratio", "min_intensity"]
PICK_LABELS = {
    "min_size":      "Smallest valid cell",
    "max_size":      "Biggest valid cell",
    "max_ratio":     "Least round (most elongated) valid cell",
    "min_intensity": "Faintest valid cell",
}
FILTER_NAMES = {
    "min_size":      "minimum object size",
    "max_size":      "maximum object size",
    "max_ratio":     "maximum major/minor axis ratio",
    "min_intensity": "minimum mean raw intensity",
}
# Each pick gets its own persistent marker color, both on the canvas (a
# rectangle drawn around the picked object) and on its label in the dock
# panel, so it's obvious at a glance which pick is which without needing
# to fill/hide the object itself.
PICK_COLORS = {
    "min_size":      "red",
    "max_size":      "yellow",
    "max_ratio":     "cyan",
    "min_intensity": "magenta",
}
# A little breathing room around the picked object's bounding box so the
# marker rectangle doesn't hug the outline too tightly.
BBOX_PAD = 4
# Highlight style for whichever "Pick ..." button is currently armed.
ARMED_BTN_STYLE = "background-color: #5a4a1a; font-weight: bold;"


def _apply_margin(key, value):
    """Pad a picked statistic into a final filter threshold. Returns 0 (the
    "no filter" sentinel used throughout CellMatchR) only when nothing was
    picked -- a real pick never rounds down to the disabled value."""
    if value is None:
        return 0
    if key == "min_size":
        return max(1, int(value * (1 - MARGIN_FRACTION)))
    if key == "max_size":
        return max(1, int(value * (1 + MARGIN_FRACTION)))
    if key == "max_ratio":
        return max(0.01, round(value * (1 + MARGIN_FRACTION), 2))
    if key == "min_intensity":
        return max(0.1, round(value * (1 - MARGIN_FRACTION), 1))
    raise ValueError(key)


def _load_cfg(path: str) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception as exc:
            print(f"[parameter_wizard] [WARNING] Could not read {path}: {exc}")
    return cfg


def _object_stats(label_image: np.ndarray, raw_image: np.ndarray) -> dict:
    """{label_id: {"area", "mean_intensity", "ratio", "bbox"}} for every detected object."""
    if label_image.max() == 0:
        return {}
    props = regionprops_table(
        label_image, intensity_image=raw_image,
        properties=("label", "area", "mean_intensity",
                    "major_axis_length", "minor_axis_length", "bbox"),
    )
    stats = {}
    for i in range(len(props["label"])):
        major = float(props["major_axis_length"][i])
        minor = max(float(props["minor_axis_length"][i]), 1e-6)
        stats[int(props["label"][i])] = {
            "area": float(props["area"][i]),
            "mean_intensity": float(props["mean_intensity"][i]),
            "ratio": major / minor,
            "bbox": (int(props["bbox-0"][i]), int(props["bbox-1"][i]),
                     int(props["bbox-2"][i]), int(props["bbox-3"][i])),
        }
    return stats


def _rect_from_bbox(bbox, image_shape) -> np.ndarray:
    """Pad a regionprops (min_row, min_col, max_row, max_col) bbox by
    BBOX_PAD and return it as 4 corner points for a napari Shapes rectangle."""
    min_row, min_col, max_row, max_col = bbox
    h, w = image_shape
    r0 = max(0, min_row - BBOX_PAD)
    c0 = max(0, min_col - BBOX_PAD)
    r1 = min(h, max_row + BBOX_PAD)
    c1 = min(w, max_col + BBOX_PAD)
    return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]], dtype=float)


class ParameterWizardApp:
    def __init__(self, cfg: dict, manifest: dict):
        self.cfg = cfg
        self.manifest = manifest
        self.channel_entries = manifest["channels"]
        self.channels = [c["channel"] for c in self.channel_entries]

        self.results: dict = {}          # {channel: {key: raw_picked_value_or_None}}
        self._current_picks: dict = {}   # in-progress picks for the channel on screen (values)
        self._current_pick_ids: dict = {}  # in-progress picks' object label ids (display only)
        self._pick_shape_boxes: dict = {}  # in-progress picks' marker rectangles (display only)
        self._stats: dict = {}
        self._raw_image = None
        self._labels = None
        self._pick_target = None
        self._pick_btns: dict = {}
        self._pick_value_lbls: dict = {}
        self._ch_idx = 0
        self._finished = False

        self.viewer = None
        self._label_lyr = None
        self._pick_shapes_lyr = None
        self._dock_layout = None

    # ── Entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        import napari
        self._process_channel(self._ch_idx)
        napari.run()

    def _process_channel(self, idx: int) -> None:
        entry = self.channel_entries[idx]
        ch = entry["channel"]
        if not entry.get("raw") or not entry.get("labels"):
            print(f"[parameter_wizard] Channel {ch}: not present in this file -- skipping.")
            self.results[ch] = {k: None for k in PICK_ORDER}
            self._advance()
            return

        print(f"[parameter_wizard] Loading channel {ch} "
              f"({entry.get('n_objects', '?')} unfiltered objects)...")
        image = tifffile.imread(entry["raw"]).astype(np.uint16)
        labels = tifffile.imread(entry["labels"]).astype(np.int32)

        self._raw_image = image
        self._labels = labels
        self._stats = _object_stats(labels, image)
        self._show_channel(ch)

    # ── napari display ───────────────────────────────────────────────────

    def _show_channel(self, ch: int) -> None:
        import napari
        self._current_picks = {k: None for k in PICK_ORDER}
        self._current_pick_ids = {k: None for k in PICK_ORDER}
        self._pick_shape_boxes = {k: None for k in PICK_ORDER}
        self._pick_target = None
        title = (f"CellMatchR — Parameter Wizard — Channel {ch}  "
                 f"({len(self._stats)} unfiltered objects)")

        first_time = self.viewer is None
        if first_time:
            self.viewer = napari.Viewer(title=title)
            self.viewer.mouse_drag_callbacks.append(self._on_click)
            self.viewer.window._qt_window.destroyed.connect(self._on_window_closed)
        else:
            self.viewer.title = title
            for lyr in list(self.viewer.layers):
                self.viewer.layers.remove(lyr)
            self._pick_shapes_lyr = None

        # Raw, unadjusted -- no percentile stretch, no gamma. napari auto-
        # ranges contrast_limits to the data's actual min/max, which is
        # just a display convenience; picked statistics always come from
        # this same unmodified array.
        self.viewer.add_image(self._raw_image, name=f"Ch {ch} raw (unadjusted)", colormap="gray")

        self._label_lyr = self.viewer.add_labels(self._labels, name=f"Ch {ch} unfiltered objects")
        self._label_lyr.contour = 2   # outline only, never filled
        self._label_lyr.editable = False
        self._label_lyr.opacity = 1.0

        # Rectangle markers for whichever objects have been picked so far
        # this channel -- one distinct color per pick criterion (never
        # filled, so a picked object stays fully visible for comparison).
        self._refresh_pick_shapes()

        if first_time:
            self._dock_widget = self._build_dock_container()
            self.viewer.window.add_dock_widget(
                self._dock_widget, name="Parameter Wizard", area="right")
        self._populate_dock(ch)

        self.viewer.status = ("Click a 'Pick ...' button below, then click the matching "
                               "object's outline in the image.")

    def _on_click(self, viewer_arg, event) -> None:
        if event.button != 1 or self._pick_target is None or self._label_lyr is None:
            return
        value = self._label_lyr.get_value(event.position, world=True)
        if not value:
            self.viewer.status = "No object there — click directly on an outlined object."
            return
        stats = self._stats.get(int(value))
        if stats is None:
            return
        key = self._pick_target
        raw_value = {
            "min_size": stats["area"], "max_size": stats["area"],
            "max_ratio": stats["ratio"], "min_intensity": stats["mean_intensity"],
        }[key]
        self._current_picks[key] = raw_value
        self._current_pick_ids[key] = int(value)
        self._pick_shape_boxes[key] = _rect_from_bbox(stats["bbox"], self._labels.shape)
        self._refresh_pick_shapes()

        self._pick_target = None
        for k in PICK_ORDER:
            self._pick_btns[k].setStyleSheet("")
            self._refresh_pick_label(k)
        self.viewer.status = f"Picked object {int(value)} as '{PICK_LABELS[key]}'."

    def _refresh_pick_shapes(self) -> None:
        """Redraw the on-canvas marker rectangles from self._pick_shape_boxes.
        Rebuilt (rather than mutated in place) so each shape's edge color
        stays correctly matched to its pick criterion."""
        if self.viewer is None:
            return
        if self._pick_shapes_lyr is not None:
            try:
                self.viewer.layers.remove(self._pick_shapes_lyr)
            except ValueError:
                pass
            self._pick_shapes_lyr = None
        data, colors = [], []
        for key in PICK_ORDER:
            box = self._pick_shape_boxes.get(key)
            if box is not None:
                data.append(box)
                colors.append(PICK_COLORS[key])
        self._pick_shapes_lyr = self.viewer.add_shapes(
            data=data, shape_type="rectangle", ndim=2,
            edge_width=4, face_color="transparent",
            edge_color=colors if colors else "red",
            name="Picked objects",
        )
        self._pick_shapes_lyr.editable = False

    def _refresh_pick_label(self, key: str) -> None:
        """Update one pick's dock label to reflect armed / picked / unset state."""
        lbl = self._pick_value_lbls.get(key)
        if lbl is None:
            return
        if self._pick_target == key:
            lbl.setText("●  Currently picking — click an object outline…")
            lbl.setStyleSheet("color: #ffaa00; font-weight: bold;")
            return
        value = self._current_picks.get(key)
        if value is None:
            lbl.setText("(not set — no filter)")
            lbl.setStyleSheet("")
        else:
            label_id = self._current_pick_ids.get(key)
            lbl.setText(self._format_pick(key, value, label_id))
            lbl.setStyleSheet(f"color: {PICK_COLORS[key]}; font-weight: bold;")

    def _arm_pick(self, key: str) -> None:
        """Arm one pick mode. Clicking the SAME button again is a no-op (stays
        armed); clicking a DIFFERENT button switches to it. Neither ever
        disarms picking on its own -- only a successful pick does that."""
        self._pick_target = key
        for k in PICK_ORDER:
            self._pick_btns[k].setStyleSheet(ARMED_BTN_STYLE if k == key else "")
            self._refresh_pick_label(k)
        if self.viewer is not None:
            self.viewer.status = f"Click the object you want as '{PICK_LABELS[key]}'."

    @staticmethod
    def _format_pick(key: str, raw_value: float, label_id) -> str:
        if key in ("min_size", "max_size"):
            return f"object {label_id}: {raw_value:.0f} px"
        if key == "max_ratio":
            return f"object {label_id}: ratio {raw_value:.2f}"
        return f"object {label_id}: mean {raw_value:.0f}"

    # ── Qt dock panel ────────────────────────────────────────────────────
    # The dock widget is created once and its layout is cleared + rebuilt
    # for each channel, rather than adding a new dock widget every time
    # (which would stack one on top of another).

    def _build_dock_container(self):
        from qtpy.QtWidgets import QWidget, QVBoxLayout
        w = QWidget()
        self._dock_layout = QVBoxLayout(w)
        return w

    def _clear_dock_layout(self) -> None:
        layout = self._dock_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _populate_dock(self, ch: int) -> None:
        from qtpy.QtWidgets import QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget

        self._clear_dock_layout()
        layout = self._dock_layout

        layout.addWidget(QLabel(
            f"<b>Channel {ch}</b><br>"
            "Click a button, then click the matching object's outline in the "
            "image -- a colored box marks it there.<br>"
            "Leave any blank to disable that filter."
        ))

        self._pick_btns = {}
        self._pick_value_lbls = {}
        for key in PICK_ORDER:
            box = QGroupBox(PICK_LABELS[key])
            box_layout = QVBoxLayout(box)

            btn_row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: {PICK_COLORS[key]}; border-radius: 3px;")
            btn = QPushButton(f"Pick {PICK_LABELS[key].lower()}…")
            btn.clicked.connect(lambda _checked=False, k=key: self._arm_pick(k))
            self._pick_btns[key] = btn
            btn_row.addWidget(swatch)
            btn_row.addWidget(btn)
            btn_row_widget = QWidget()
            btn_row_widget.setLayout(btn_row)

            val_lbl = QLabel()
            self._pick_value_lbls[key] = val_lbl

            box_layout.addWidget(btn_row_widget)
            box_layout.addWidget(val_lbl)
            layout.addWidget(box)

        for key in PICK_ORDER:
            self._refresh_pick_label(key)

        clear_btn = QPushButton("Clear all picks for this channel")
        clear_btn.clicked.connect(self._clear_picks)
        layout.addWidget(clear_btn)

        is_last = self._ch_idx >= len(self.channels) - 1
        next_btn = QPushButton("Save & Finish" if is_last else "Save & Next Channel →")
        next_btn.clicked.connect(self._save_and_advance)
        layout.addWidget(next_btn)
        layout.addStretch()

    def _clear_picks(self) -> None:
        self._pick_target = None
        for key in PICK_ORDER:
            self._current_picks[key] = None
            self._current_pick_ids[key] = None
            self._pick_shape_boxes[key] = None
            self._pick_btns[key].setStyleSheet("")
            self._refresh_pick_label(key)
        self._refresh_pick_shapes()

    # ── Advancing / finishing ────────────────────────────────────────────

    def _save_and_advance(self) -> None:
        ch = self.channels[self._ch_idx]
        self.results[ch] = dict(self._current_picks)
        self._advance()

    def _advance(self) -> None:
        self._ch_idx += 1
        if self._ch_idx >= len(self.channels):
            self._finish()
            return
        self._process_channel(self._ch_idx)

    def _on_window_closed(self) -> None:
        """User closed the window instead of clicking Save & Next/Finish --
        commit whatever was picked for the channel on screen and save."""
        if self._finished:
            return
        ch = self.channels[self._ch_idx]
        self.results.setdefault(ch, dict(self._current_picks))
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True

        print("[parameter_wizard] Computing filter thresholds (with safety margin)...")
        try:
            existing = json.loads(self.cfg.get("seg_per_channel_filters", "{}") or "{}")
        except Exception:
            existing = {}

        visited = set(self.results.keys())
        for ch in self.channels:
            if ch not in visited:
                print(f"[parameter_wizard] Channel {ch} was never opened -- left unchanged.")
                continue
            picks = self.results[ch]
            entry = {key: _apply_margin(key, picks.get(key)) for key in PICK_ORDER}
            existing[str(ch)] = entry
            summary = ", ".join(
                f"{FILTER_NAMES[k]}={entry[k]}" if entry[k] else f"{FILTER_NAMES[k]}=disabled"
                for k in PICK_ORDER
            )
            print(f"[parameter_wizard] Channel {ch}: {summary}")

        self.cfg["seg_per_channel_filters"] = json.dumps(existing)
        self.cfg["seg_per_channel_filters_enabled"] = True
        save_config(self.cfg)
        save_project_config(self.cfg)
        print("[parameter_wizard] Saved to config. Done.")

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="CellMatchR Parameter Wizard (display stage)")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to CellMatchr config JSON")
    parser.add_argument("--manifest", required=True,
                         help="Path to the manifest written by parameter_wizard_compute.py")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)
    with open(args.manifest) as f:
        manifest = json.load(f)
    if not manifest.get("channels"):
        print("[ERROR] Manifest has no channels -- nothing to calibrate.")
        sys.exit(1)

    print(f"[parameter_wizard] Sample: '{manifest.get('base_name', '?')}'")
    print(f"[parameter_wizard] Channels: {[c['channel'] for c in manifest['channels']]}")

    app = ParameterWizardApp(cfg, manifest)
    app.run()


if __name__ == "__main__":
    main()
