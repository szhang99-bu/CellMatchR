"""
roi_editor.py
=============
Napari-based pre-validation ROI editor for CellMatchR.

Opens a napari viewer for every sample in a segmentation folder so you can
paint any cells the model missed before running the validator.  Each
disconnected brush region is automatically treated as a separate ROI — you
never need to change the label colour between cells.

Test mode  (--test-mode)
    Does NOT write to the original mask TIF.  Instead:
      • Saves a DEBUG_<mask_name>.tif next to the original
      • Saves a DEBUG_<sample>_review.png with new ROIs highlighted in yellow
      • Prints a before/after ROI count comparison to the terminal
      • Processes only the first image — stops after one test save

Normal mode  (default)
    Writes directly to the prediction mask TIF.  No PNG, no debug files.

Launched from the CellMatchR launcher (Step 4 → "Review Segmented Images").
Can also be run standalone:

    python roi_editor.py --seg-folder /path/to/Segmentation/run --channel 2
    python roi_editor.py --seg-folder /path/to/Segmentation/run --channel 2 --test-mode
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import tifffile
from skimage.measure import label as connected_components

# ── Display resolution cap ───────────────────────────────────────────────────
# Large TIFFs freeze napari on the first brush stroke because it must upload
# the entire array as a GPU texture.  We downsample the *display* layers to
# at most DISPLAY_MAX pixels on the longest side.  The original mask is never
# touched — additions are scaled back up before saving.
DISPLAY_MAX = 2048


def _display_scale(h: int, w: int) -> int:
    """
    Return the integer stride (1 = no downsampling, 2 = half, 4 = quarter…)
    so that max(h/s, w/s) <= DISPLAY_MAX.
    """
    factor = max(h, w) / DISPLAY_MAX
    if factor <= 1.0:
        return 1
    s = 1
    while max(h // s, w // s) > DISPLAY_MAX:
        s *= 2
    return s


def _downsample(arr: np.ndarray, s: int) -> np.ndarray:
    """Stride-based downsample (nearest-neighbour, no blending — safe for labels)."""
    if s == 1:
        return arr
    return arr[::s, ::s]


def _upsample_additions(additions: np.ndarray, s: int,
                        orig_h: int, orig_w: int) -> np.ndarray:
    """
    Scale additions back to full resolution using integer repeat, then trim
    to the exact original size (in case orig dimensions aren't divisible by s).
    """
    if s == 1:
        return additions
    up = np.repeat(np.repeat(additions, s, axis=0), s, axis=1)
    return up[:orig_h, :orig_w]


# ── File-finding helpers ─────────────────────────────────────────────────────

def _find_raw(sample_dir: str, channel: int) -> str | None:
    for pat in (f"*_split_ch{channel}.tif", f"*_split_ch{channel}.TIF"):
        hits = glob(os.path.join(sample_dir, pat))
        if hits:
            return hits[0]
    return None


def _find_mask(sample_dir: str, channel: int) -> str | None:
    skip = ("coverage", "combined", "debug", "unfiltered")
    for pat in (
        f"*_predict_original_ch{channel}*.tif",
        f"*_predict_original_ch{channel}*.TIF",
    ):
        candidates = [
            p for p in glob(os.path.join(sample_dir, pat))
            if not any(k in os.path.basename(p).lower() for k in skip)
        ]
        if candidates:
            return candidates[-1]
    return None


def _find_samples(seg_folder: str, channel: int) -> list[dict]:
    dirs = sorted([
        os.path.join(seg_folder, d)
        for d in os.listdir(seg_folder)
        if os.path.isdir(os.path.join(seg_folder, d))
        and not d.startswith((".", "DEBUG"))
    ])
    out = []
    for d in dirs:
        raw  = _find_raw(d, channel)
        mask = _find_mask(d, channel)
        if raw and mask:
            out.append({"name": os.path.basename(d), "path": d,
                        "raw": raw, "mask": mask})
    return out


# ── Core merge logic ─────────────────────────────────────────────────────────

def _merge_additions(existing: np.ndarray,
                     additions: np.ndarray) -> tuple[np.ndarray, int, set[int]]:
    """
    Separate every disconnected painted blob into its own ROI.

    Crucially, this runs connected-components INDEPENDENTLY for each
    label value (colour).  That means two cells painted in different
    colours that are physically touching will always become separate
    ROIs — which is exactly the point of having multiple colours.

    Two cells painted in the SAME colour that are touching will be
    merged into one ROI (user should use a different colour in that case).

    Only fills pixels where existing mask == 0 (background).
    Returns (new_mask, n_added, new_ids).
    """
    label_values = np.unique(additions)
    label_values = label_values[label_values > 0]   # ignore background
    if len(label_values) == 0:
        return existing, 0, set()

    max_id   = int(existing.max())
    new_mask = existing.copy()
    new_ids  = set()
    added    = 0

    for lv in label_values:
        # Binary mask for just this colour
        single = (additions == lv).astype(np.uint8)
        labeled, n = connected_components(single, return_num=True, connectivity=2)
        for i in range(1, n + 1):
            region = (labeled == i) & (existing == 0)
            if region.sum() == 0:
                continue
            nid = max_id + added + 1
            new_mask[region] = nid
            new_ids.add(nid)
            added += 1

    return new_mask, added, new_ids


# ── Debug PNG ────────────────────────────────────────────────────────────────

def _save_debug_png(raw_path: str,
                    existing: np.ndarray,
                    new_mask: np.ndarray,
                    new_ids: set[int],
                    png_path: str) -> None:
    """
    Save an overlay PNG where:
      • Raw image  → grayscale background
      • Existing ROIs → soft blue-ish tint (semi-transparent)
      • New ROIs   → solid bright yellow (clearly different)
      • Legend in bottom-right corner
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    raw = tifffile.imread(raw_path).astype(np.float32)
    lo, hi = np.percentile(raw, (1, 99))
    raw_norm = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)

    # Build RGB from grayscale
    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

    # Existing ROIs — soft cyan tint
    existing_mask = (existing > 0) & (~np.isin(existing, list(new_ids)))
    rgb[existing_mask] = rgb[existing_mask] * 0.5 + np.array([0.2, 0.6, 0.9]) * 0.5

    # New ROIs — bright yellow
    new_mask_px = np.isin(new_mask, list(new_ids))
    rgb[new_mask_px] = rgb[new_mask_px] * 0.2 + np.array([1.0, 0.95, 0.0]) * 0.8

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)
    ax.axis("off")

    n_existing = len(np.unique(existing)) - 1  # subtract background
    n_new      = len(new_ids)

    patches = [
        mpatches.Patch(color=(0.2, 0.6, 0.9),
                       label=f"Existing ROIs  ({n_existing})"),
        mpatches.Patch(color=(1.0, 0.95, 0.0),
                       label=f"New ROIs added  ({n_new})"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=11,
              framealpha=0.8, facecolor="#111111",
              labelcolor="white", edgecolor="none")

    fig.suptitle(
        f"ROI Editor — test mode preview\n"
        f"Before: {n_existing} ROIs   →   After: {n_existing + n_new} ROIs   "
        f"(+{n_new} added)",
        fontsize=12, color="white",
    )
    fig.patch.set_facecolor("#1e1e2e")
    fig.tight_layout(pad=0.5)
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Brush colour palette ────────────────────────────────────────────────────
# Three visually distinct colours the user can switch between for adjacent cells.
# Each entry:  (label_value, napari_colour_name, hex_for_qt_button, display_name)
BRUSH_COLOURS = [
    (1, "yellow",  "#e6d800", "Yellow"),
    (2, "cyan",    "#00e5e5", "Cyan"),
    (3, "magenta", "#e500e5", "Magenta"),
]


# ── Main application ──────────────────────────────────────────────────────────

class RoiEditorApp:
    """
    Napari viewer with a navigation dock panel.

    Layers
    ------
    Raw Image      — grayscale background (read-only)
    Existing ROIs  — current prediction mask (read-only, for reference)
    Add ROIs here  — empty labels layer the user paints on
    """

    DEFAULT_BRUSH_SIZE = 7

    def __init__(self, seg_folder: str, channel: int, test_mode: bool = False):
        self.seg_folder = seg_folder
        self.channel    = channel
        self.test_mode  = test_mode
        self.samples    = _find_samples(seg_folder, channel)
        self.idx        = 0
        self._test_done    = False     # flips True after first test-mode save

        # Per-sample display state (set in _load_sample)
        self._scale     = 1           # display stride for current sample
        self._orig_h    = 0           # full-res height
        self._orig_w    = 0           # full-res width

        self._first_paint = True    # used to print one-time debug on first stroke

        # napari layer handles
        self.viewer    = None
        self._raw_lyr  = None
        self._mask_lyr = None
        self._add_lyr  = None

        # Qt widgets (set in _build_dock)
        self._prog_lbl   = None
        self._sample_lbl = None
        self._status_lbl = None
        self._save_btn   = None     # stored so we can disable it in test mode
        self._colour_btns: dict = {}  # label_value → QPushButton

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self) -> None:
        import napari
        from qtpy.QtCore import QTimer
        title = "CellMatchR — ROI Editor  [TEST MODE]" if self.test_mode \
                else "CellMatchR — ROI Editor"
        print("[roi_editor] Creating napari viewer...")
        self.viewer = napari.Viewer(title=title)
        print("[roi_editor] Loading first sample...")
        self._load_sample(self.idx)
        print("[roi_editor] Building dock panel...")
        self.viewer.window.add_dock_widget(
            self._build_dock(), name="Navigation", area="right"
        )
        # Pre-warm the Metal/OpenGL renderer so the first brush stroke
        # doesn't trigger a ~30s shader compilation freeze.
        # We paint one pixel, force a render, then clear it — this
        # runs 800 ms after the window opens so it's fully visible first.
        print("[roi_editor] Window ready — pre-warming renderer in background...")
        QTimer.singleShot(800, self._prewarm_renderer)
        napari.run()

    def _prewarm_renderer(self) -> None:
        """
        Paint a single invisible pixel and force a Qt render cycle so that
        macOS/Metal compiles the Labels-layer shader NOW (at load time) rather
        than on the first real brush stroke, which would freeze the UI for ~30s.
        """
        from qtpy.QtWidgets import QApplication
        if self._add_lyr is None:
            return
        print("[roi_editor] Pre-warming renderer (Metal shader compile)...")
        self.viewer.status = "Initializing renderer — please wait a moment…"
        h, w = self._add_lyr.data.shape[:2]
        # Paint one pixel at the centre
        tmp = np.zeros((h, w), dtype=np.int32)
        tmp[h // 2, w // 2] = 1
        self._add_lyr.data = tmp
        # Process all pending Qt events — this triggers the actual GPU draw call
        QApplication.processEvents()
        # Clear immediately
        self._add_lyr.data = np.zeros((h, w), dtype=np.int32)
        QApplication.processEvents()
        self.viewer.status = "Ready — select 'Add ROIs here' layer and press P to paint"
        print("[roi_editor] Renderer ready. You can now paint without delay.")

    def _on_first_paint(self, event=None) -> None:
        """Fires on the first data change to the additions layer (i.e. first stroke)."""
        if self._first_paint:
            self._first_paint = False
            print("[roi_editor] First brush stroke received — rendering...")

    # ── Dock widget ──────────────────────────────────────────────────────────

    def _build_dock(self):
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
        )
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QFont

        root = QWidget()
        lay  = QVBoxLayout()
        lay.setSpacing(6)
        root.setLayout(lay)

        # ── Test mode banner ──
        if self.test_mode:
            banner = QLabel("🧪  TEST MODE")
            banner.setAlignment(Qt.AlignCenter)
            banner.setStyleSheet(
                "background:#7c3a00; color:#fab387; font-weight:bold;"
                " font-size:12px; padding:6px; border-radius:4px;"
            )
            lay.addWidget(banner)
            sub = QLabel(
                "Saves to DEBUG_ file only.\n"
                "Processes one image then stops.\n"
                "No changes to original mask."
            )
            sub.setAlignment(Qt.AlignCenter)
            sub.setStyleSheet("color:#fab387; font-size:10px;")
            lay.addWidget(sub)
            lay.addWidget(self._hr())

        # ── Title ──
        title = QLabel("ROI Editor")
        title.setFont(QFont("", 13, QFont.Bold))
        lay.addWidget(title)

        # ── First-stroke warning ──
        freeze_warn = QLabel(
            "⏳  FIRST STROKE MAY FREEZE\n"
            "      FOR UP TO 30 SECONDS\n"
            "This is normal — macOS is compiling\n"
            "the GPU renderer on first use.\n"
            "It only happens once per session."
        )
        freeze_warn.setAlignment(Qt.AlignCenter)
        freeze_warn.setWordWrap(True)
        freeze_warn.setStyleSheet(
            "background:#3a3000; color:#f9e2af; font-size:11px;"
            " font-weight:bold; padding:8px; border-radius:5px;"
            " border: 1px solid #f9e2af;"
        )
        lay.addWidget(freeze_warn)

        lay.addWidget(self._hr())

        # ── Instructions ──
        instr = QLabel(
            "<b>How to add missed cells:</b><br><br>"
            "1. Make sure <i>Add ROIs here</i> layer is selected<br>"
            "2. Press <b>P</b> for the brush<br>"
            "3. Paint over missed cells<br>"
            "   • Same colour + not touching → separate ROIs ✓<br>"
            "   • <b>Touching cells → use different colours</b><br>"
            "4. Press Save when done<br><br>"
            "<i>[ and ] keys adjust brush size</i>"
        )
        instr.setWordWrap(True)
        instr.setTextFormat(Qt.RichText)
        instr.setStyleSheet("font-size: 11px;")
        lay.addWidget(instr)

        lay.addWidget(self._hr())

        # ── Progress / status ──
        self._prog_lbl = QLabel()
        self._prog_lbl.setFont(QFont("", 10, QFont.Bold))
        lay.addWidget(self._prog_lbl)

        self._sample_lbl = QLabel()
        self._sample_lbl.setWordWrap(True)
        self._sample_lbl.setStyleSheet("color:#aaaaaa; font-size:10px;")
        lay.addWidget(self._sample_lbl)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("font-size:11px;")
        lay.addWidget(self._status_lbl)

        lay.addWidget(self._hr())

        # ── Colour picker ──
        colour_lbl = QLabel("Brush colour  (use different colours for touching cells):")
        colour_lbl.setWordWrap(True)
        colour_lbl.setStyleSheet("font-size:10px; color:#aaaaaa;")
        lay.addWidget(colour_lbl)

        colour_row = QWidget()
        colour_row_lay = QHBoxLayout()
        colour_row_lay.setContentsMargins(0, 0, 0, 0)
        colour_row_lay.setSpacing(4)
        colour_row.setLayout(colour_row_lay)

        self._colour_btns = {}
        for lv, _napari_name, hex_col, display_name in BRUSH_COLOURS:
            btn = QPushButton(display_name)
            btn.setCheckable(True)
            btn.setChecked(lv == 1)   # yellow selected by default
            btn.setStyleSheet(
                f"QPushButton {{ background:{hex_col}; color:#111; font-weight:bold;"
                f" padding:4px 6px; border-radius:4px; border:2px solid transparent; }}"
                f"QPushButton:checked {{ border:2px solid white; }}"
                f"QPushButton:hover {{ opacity:0.85; }}"
            )
            # Capture lv in default arg to avoid closure issue
            btn.clicked.connect(lambda _, label=lv: self._set_brush_colour(label))
            colour_row_lay.addWidget(btn)
            self._colour_btns[lv] = btn

        lay.addWidget(colour_row)
        lay.addWidget(self._hr())

        # ── Buttons ──
        save_label = "🧪  Test Save" if self.test_mode else "💾  Save & Next  →"
        self._save_btn = QPushButton(save_label)
        self._save_btn.setStyleSheet(
            "QPushButton { background:#2a7f4f; color:white; font-weight:bold;"
            " padding:6px; border-radius:4px; }"
            "QPushButton:hover { background:#35a264; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._save_btn.clicked.connect(self._save_and_next)
        lay.addWidget(self._save_btn)

        if not self.test_mode:
            skip_btn = QPushButton("Skip  →")
            skip_btn.setStyleSheet("QPushButton { padding:6px; border-radius:4px; }")
            skip_btn.clicked.connect(self._skip)
            lay.addWidget(skip_btn)

            prev_btn = QPushButton("←  Previous")
            prev_btn.setStyleSheet("QPushButton { padding:6px; border-radius:4px; }")
            prev_btn.clicked.connect(self._prev)
            lay.addWidget(prev_btn)

        lay.addStretch()

        finish_btn = QPushButton("Finish  ✓")
        finish_btn.setStyleSheet(
            "QPushButton { background:#444; color:white; padding:6px;"
            " border-radius:4px; }"
            "QPushButton:hover { background:#666; }"
        )
        finish_btn.clicked.connect(self.viewer.close)
        lay.addWidget(finish_btn)

        self._refresh_labels()
        return root

    @staticmethod
    def _hr():
        from qtpy.QtWidgets import QFrame
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color:#555;")
        return f

    # ── Sample loading ───────────────────────────────────────────────────────

    def _load_sample(self, idx: int) -> None:
        s   = self.samples[idx]
        print(f"[roi_editor] Loading sample {idx + 1}/{len(self.samples)}: {s['name']}")

        print(f"[roi_editor]   Reading raw image...")
        raw = tifffile.imread(s["raw"]).astype(np.float32)
        print(f"[roi_editor]   Reading mask...")
        msk = tifffile.imread(s["mask"]).astype(np.int32)
        n_rois = len(np.unique(msk)) - 1
        print(f"[roi_editor]   Mask has {n_rois} existing ROIs")

        # ── Compute display scale for this sample ────────────────────
        h, w = raw.shape[:2]
        self._orig_h = h
        self._orig_w = w
        self._scale  = _display_scale(h, w)

        raw_d = _downsample(raw, self._scale)
        msk_d = _downsample(msk, self._scale)

        if self._scale > 1:
            print(f"[roi_editor]   {w}×{h} → display at "
                  f"{raw_d.shape[1]}×{raw_d.shape[0]} (1/{self._scale})")
        else:
            print(f"[roi_editor]   Full resolution ({w}×{h}), no downsampling needed")

        lo, hi = np.percentile(raw_d, (1, 99))

        if self._raw_lyr is None:
            self._raw_lyr = self.viewer.add_image(
                raw_d, name="Raw Image", colormap="gray",
                contrast_limits=[lo, hi],
            )
            self._mask_lyr = self.viewer.add_labels(
                msk_d, name="Existing ROIs", opacity=0.45,
            )
            self._mask_lyr.editable = False
            self._add_lyr = self.viewer.add_labels(
                np.zeros_like(msk_d, dtype=np.int32),
                name="Add ROIs here", opacity=0.75,
            )
            # Pre-assign all three brush colours so napari never needs to
            # recompute the LUT on a stroke (the main cause of freezes).
            # Must be set as a property after creation, not as a kwarg.
            self._add_lyr.color = {lv: name for lv, name, _, _ in BRUSH_COLOURS}
            self._add_lyr.brush_size     = self.DEFAULT_BRUSH_SIZE
            self._add_lyr.selected_label = 1
            self._add_lyr.mode           = "paint"
            self.viewer.layers.selection.active = self._add_lyr
            # Hook for first-stroke debug output
            self._add_lyr.events.data.connect(self._on_first_paint)
            print("[roi_editor] Layers created — waiting for renderer pre-warm...")
        else:
            self._raw_lyr.data            = raw_d
            self._raw_lyr.contrast_limits = [lo, hi]
            self._mask_lyr.data           = msk_d
            self._add_lyr.data            = np.zeros_like(msk_d, dtype=np.int32)
            self._add_lyr.selected_label  = 1
            self._add_lyr.mode            = "paint"
            self.viewer.layers.selection.active = self._add_lyr

        self.viewer.reset_view()

    # ── Button callbacks ─────────────────────────────────────────────────────

    def _set_brush_colour(self, label: int) -> None:
        """Switch the active brush label and update button checked states."""
        if self._add_lyr is not None:
            self._add_lyr.selected_label = label
            self._add_lyr.mode = "paint"
            self.viewer.layers.selection.active = self._add_lyr
        for lv, btn in self._colour_btns.items():
            btn.setChecked(lv == label)

    def _save_and_next(self) -> None:
        self._save_current()
        if not self.test_mode:
            self._advance()

    def _skip(self) -> None:
        self._set_status("Skipped — no changes saved.", colour="#aaaaaa")
        self._advance()

    def _prev(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self._load_sample(self.idx)
            self._set_status("")
            self._refresh_labels()

    def _advance(self) -> None:
        if self.idx < len(self.samples) - 1:
            self.idx += 1
            self._load_sample(self.idx)
            self._refresh_labels()
        else:
            self._set_status("✓  All images reviewed!", colour="#a6e3a1")

    # ── Save logic ───────────────────────────────────────────────────────────

    def _save_current(self) -> None:
        additions_display = self._add_lyr.data
        if additions_display.max() == 0:
            self._set_status("Nothing drawn — skipped.", colour="#aaaaaa")
            return

        s = self.samples[self.idx]

        # Scale additions back to full resolution before merging
        additions = _upsample_additions(
            additions_display, self._scale, self._orig_h, self._orig_w
        )

        # Always merge into the original full-res mask from disk
        existing = tifffile.imread(s["mask"]).astype(np.int32)
        new_mask, n_added, new_ids = _merge_additions(existing, additions)

        if n_added == 0:
            self._set_status(
                "Painted pixels overlapped existing ROIs only — nothing added.",
                colour="#fab387",
            )
            return

        if self.test_mode:
            self._test_save(s, existing, new_mask, new_ids, n_added)
        else:
            self._normal_save(s, new_mask, n_added)

    def _normal_save(self, s: dict, new_mask: np.ndarray, n_added: int) -> None:
        """Write directly to the original mask TIF. No PNG, no debug files."""
        tifffile.imwrite(s["mask"], new_mask.astype(np.uint32), compression="lzw")
        # Refresh display layers at display resolution
        self._mask_lyr.data = _downsample(new_mask, self._scale)
        self._add_lyr.data  = np.zeros_like(self._mask_lyr.data, dtype=np.int32)
        self._set_status(f"✓  {n_added} new ROI(s) saved.", colour="#a6e3a1")

    def _test_save(self, s: dict, existing: np.ndarray,
                   new_mask: np.ndarray, new_ids: set[int], n_added: int) -> None:
        """
        Test-mode save:
          1. Write DEBUG_ TIF (never touches the original)
          2. Write DEBUG_ PNG with new ROIs in yellow
          3. Print before/after comparison to terminal
          4. Disable the save button (one image only)
        """
        mask_dir  = os.path.dirname(s["mask"])
        mask_base = os.path.basename(s["mask"])
        stem      = os.path.splitext(mask_base)[0]

        debug_tif = os.path.join(mask_dir, f"DEBUG_{mask_base}")
        debug_png = os.path.join(mask_dir, f"DEBUG_{stem}_review.png")

        # 1. Save DEBUG TIF
        tifffile.imwrite(debug_tif, new_mask.astype(np.uint32), compression="lzw")

        # 2. Save debug PNG
        try:
            _save_debug_png(s["raw"], existing, new_mask, new_ids, debug_png)
        except Exception as exc:
            print(f"[roi_editor] WARNING: could not save debug PNG: {exc}")
            debug_png = "(failed)"

        # 3. Print comparison
        n_before = len(np.unique(existing)) - 1
        n_after  = len(np.unique(new_mask))  - 1
        print()
        print("=" * 55)
        print("  ROI Editor — TEST MODE result")
        print("=" * 55)
        print(f"  Sample        : {s['name']}")
        print(f"  Original ROIs : {n_before}")
        print(f"  After adding  : {n_after}  (+{n_added})")
        print(f"  DEBUG TIF     : {debug_tif}")
        print(f"  DEBUG PNG     : {debug_png}")
        print("=" * 55)
        print()

        # 4. Update viewer for visual confirmation (display res), disable save
        self._mask_lyr.data = _downsample(new_mask, self._scale)
        self._add_lyr.data  = np.zeros_like(self._mask_lyr.data, dtype=np.int32)

        self._set_status(
            f"🧪 Test done: +{n_added} ROI(s)\n"
            f"DEBUG files saved.\nUncheck Test Mode to save for real.",
            colour="#fab387",
        )

        self._test_done = True
        if self._save_btn is not None:
            self._save_btn.setEnabled(False)
            self._save_btn.setText("🧪 Test complete")

    # ── Label helpers ────────────────────────────────────────────────────────

    def _refresh_labels(self) -> None:
        if self._prog_lbl is None:
            return
        n = len(self.samples)
        if self.test_mode:
            self._prog_lbl.setText(f"Image 1 of {n}  (test: 1st only)")
        else:
            self._prog_lbl.setText(f"Image {self.idx + 1} of {n}")
        self._sample_lbl.setText(self.samples[self.idx]["name"])

    def _set_status(self, text: str, colour: str = "#cdd6f4") -> None:
        if self._status_lbl is None:
            return
        self._status_lbl.setStyleSheet(f"font-size:11px; color:{colour};")
        self._status_lbl.setText(text)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CellMatchR ROI Editor — add missed cells before validation."
    )
    parser.add_argument("--seg-folder", required=True,
                        help="Segmentation run folder")
    parser.add_argument("--channel", type=int, required=True,
                        help="Channel number (e.g. 2)")
    parser.add_argument("--test-mode", action="store_true", default=False,
                        help="Test mode: save to DEBUG_ files only, "
                             "process one image, print comparison")
    args = parser.parse_args()

    if not os.path.isdir(args.seg_folder):
        sys.exit(f"ERROR: folder not found: {args.seg_folder}")

    samples = _find_samples(args.seg_folder, args.channel)
    if not samples:
        sys.exit(
            f"ERROR: no samples with raw + mask found in:\n"
            f"  {args.seg_folder}  (channel {args.channel})"
        )

    mode_str = " [TEST MODE]" if args.test_mode else ""
    print(f"[roi_editor{mode_str}] {len(samples)} samples — channel {args.channel}")

    RoiEditorApp(args.seg_folder, args.channel, test_mode=args.test_mode).run()


if __name__ == "__main__":
    main()
