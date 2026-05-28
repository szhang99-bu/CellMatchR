"""
roi_editor.py
=============
Napari-based pre-validation ROI editor for CellMatchR.

Opens a napari viewer for every sample in a segmentation folder so you can
paint any cells the model missed before running the validator.  Each
disconnected brush region is automatically treated as a separate ROI — you
never need to change the label colour between cells.

Launched from the CellMatchR launcher (Step 4 — Validation →
"Review Segmented Images" button).  Can also be run standalone:

    python roi_editor.py --seg-folder /path/to/Segmentation/run --channel 2
"""

import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import tifffile
from skimage.measure import label as connected_components


# ── Helpers to locate per-sample files ──────────────────────────────────────

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
    """Return a list of dicts {name, path, raw, mask} for every usable sample."""
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

def _merge_additions(existing: np.ndarray, additions: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Take the user's painted additions layer (all drawn pixels share label=1
    or any single value) and:
      1. Run connected-components to give each disconnected blob its own ID.
      2. Only assign pixels where the existing mask is background (0).
      3. Assign new IDs starting from max(existing) + 1.

    Returns (new_mask, n_added).
    """
    drawn_binary = (additions > 0).astype(np.uint8)
    labeled, n   = connected_components(drawn_binary, return_num=True, connectivity=2)
    if n == 0:
        return existing, 0

    max_id   = int(existing.max())
    new_mask = existing.copy()
    added    = 0
    for i in range(1, n + 1):
        region = (labeled == i) & (existing == 0)   # don't overwrite existing ROIs
        if region.sum() == 0:
            continue
        new_mask[region] = max_id + added + 1
        added += 1

    return new_mask, added


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

    # Default brush radius in pixels — user can change it in the napari toolbar
    DEFAULT_BRUSH_SIZE = 18

    def __init__(self, seg_folder: str, channel: int):
        self.seg_folder = seg_folder
        self.channel    = channel
        self.samples    = _find_samples(seg_folder, channel)
        self.idx        = 0

        # Napari layer references (set in run())
        self.viewer     = None
        self._raw_lyr   = None
        self._mask_lyr  = None
        self._add_lyr   = None

        # Qt widget labels (set in _build_dock())
        self._prog_lbl   = None
        self._sample_lbl = None
        self._status_lbl = None

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        import napari
        self.viewer = napari.Viewer(title="CellMatchR — ROI Editor")
        self._load_sample(self.idx)
        dock_widget = self._build_dock()
        self.viewer.window.add_dock_widget(dock_widget, name="Navigation", area="right")
        napari.run()

    # ── Dock widget ──────────────────────────────────────────────────────────

    def _build_dock(self):
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QPushButton, QLabel, QFrame, QSizePolicy
        )
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QFont

        root = QWidget()
        lay  = QVBoxLayout()
        lay.setSpacing(6)
        root.setLayout(lay)

        # ── Title ──
        title = QLabel("ROI Editor")
        title.setFont(QFont("", 13, QFont.Bold))
        lay.addWidget(title)

        lay.addWidget(self._hr())

        # ── Instructions ──
        instr = QLabel(
            "<b>How to add missed cells:</b><br><br>"
            "1. Make sure <i>Add ROIs here</i> layer is selected (click it in the layer list)<br>"
            "2. Press <b>P</b> to activate the brush<br>"
            "3. Paint over every missed cell — <b>same colour is fine</b>, "
               "each disconnected blob will automatically become its own ROI<br>"
            "4. Press <b>Save &amp; Next</b> when done with this image<br><br>"
            "<i>Tip: adjust brush size with [ and ] keys</i>"
        )
        instr.setWordWrap(True)
        instr.setTextFormat(Qt.RichText)
        instr.setStyleSheet("font-size: 11px;")
        lay.addWidget(instr)

        lay.addWidget(self._hr())

        # ── Progress ──
        self._prog_lbl = QLabel()
        self._prog_lbl.setFont(QFont("", 10, QFont.Bold))
        lay.addWidget(self._prog_lbl)

        self._sample_lbl = QLabel()
        self._sample_lbl.setWordWrap(True)
        self._sample_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        lay.addWidget(self._sample_lbl)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("font-size: 11px;")
        lay.addWidget(self._status_lbl)

        lay.addWidget(self._hr())

        # ── Buttons ──
        save_btn = QPushButton("💾  Save & Next  →")
        save_btn.setStyleSheet(
            "QPushButton { background:#2a7f4f; color:white; font-weight:bold;"
            " padding:6px; border-radius:4px; }"
            "QPushButton:hover { background:#35a264; }"
        )
        save_btn.clicked.connect(self._save_and_next)
        lay.addWidget(save_btn)

        skip_btn = QPushButton("Skip  →")
        skip_btn.setStyleSheet(
            "QPushButton { padding:6px; border-radius:4px; }"
        )
        skip_btn.clicked.connect(self._skip)
        lay.addWidget(skip_btn)

        prev_btn = QPushButton("←  Previous")
        prev_btn.setStyleSheet(
            "QPushButton { padding:6px; border-radius:4px; }"
        )
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
        f.setStyleSheet("color: #555;")
        return f

    # ── Sample loading ───────────────────────────────────────────────────────

    def _load_sample(self, idx: int) -> None:
        s   = self.samples[idx]
        raw = tifffile.imread(s["raw"]).astype(np.float32)
        msk = tifffile.imread(s["mask"]).astype(np.int32)

        lo, hi = np.percentile(raw, (1, 99))

        if self._raw_lyr is None:
            # First load — create layers
            self._raw_lyr = self.viewer.add_image(
                raw, name="Raw Image", colormap="gray",
                contrast_limits=[lo, hi],
            )
            self._mask_lyr = self.viewer.add_labels(
                msk, name="Existing ROIs", opacity=0.45,
            )
            self._mask_lyr.editable = False

            add_data = np.zeros_like(msk, dtype=np.int32)
            self._add_lyr = self.viewer.add_labels(
                add_data, name="Add ROIs here", opacity=0.75,
            )
            self._add_lyr.brush_size    = self.DEFAULT_BRUSH_SIZE
            self._add_lyr.selected_label = 1
            self._add_lyr.mode          = "paint"
            # Make it the active layer so the user can start painting immediately
            self.viewer.layers.selection.active = self._add_lyr
        else:
            # Subsequent loads — swap data
            self._raw_lyr.data           = raw
            self._raw_lyr.contrast_limits = [lo, hi]
            self._mask_lyr.data          = msk
            self._add_lyr.data           = np.zeros_like(msk, dtype=np.int32)
            self._add_lyr.selected_label  = 1
            self._add_lyr.mode            = "paint"
            self.viewer.layers.selection.active = self._add_lyr

        self.viewer.reset_view()

    # ── Button callbacks ─────────────────────────────────────────────────────

    def _save_and_next(self) -> None:
        self._save_current()
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
        additions = self._add_lyr.data
        if additions.max() == 0:
            self._set_status("Nothing drawn — skipped.", colour="#aaaaaa")
            return

        # Re-read the mask from disk in case a previous save already modified it
        s        = self.samples[self.idx]
        existing = tifffile.imread(s["mask"]).astype(np.int32)

        new_mask, n_added = _merge_additions(existing, additions)

        if n_added == 0:
            self._set_status(
                "Painted pixels overlapped existing ROIs only — nothing new saved.",
                colour="#fab387",
            )
            return

        tifffile.imwrite(s["mask"], new_mask.astype(np.uint32), compression="lzw")

        # Refresh the reference layer so the user can see the additions
        self._mask_lyr.data = new_mask
        self._add_lyr.data  = np.zeros_like(new_mask, dtype=np.int32)

        self._set_status(f"✓  {n_added} new ROI(s) saved.", colour="#a6e3a1")

    # ── Label helpers ────────────────────────────────────────────────────────

    def _refresh_labels(self) -> None:
        if self._prog_lbl is None:
            return
        self._prog_lbl.setText(f"Image {self.idx + 1} of {len(self.samples)}")
        self._sample_lbl.setText(self.samples[self.idx]["name"])

    def _set_status(self, text: str, colour: str = "#cdd6f4") -> None:
        if self._status_lbl is None:
            return
        self._status_lbl.setStyleSheet(f"font-size: 11px; color: {colour};")
        self._status_lbl.setText(text)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CellMatchR ROI Editor — add missed cells before validation."
    )
    parser.add_argument("--seg-folder", required=True,
                        help="Path to the segmentation run folder "
                             "(e.g. /project/Segmentation/run_1)")
    parser.add_argument("--channel", type=int, required=True,
                        help="Channel number to edit (e.g. 2)")
    args = parser.parse_args()

    if not os.path.isdir(args.seg_folder):
        sys.exit(f"ERROR: folder not found: {args.seg_folder}")

    samples = _find_samples(args.seg_folder, args.channel)
    if not samples:
        sys.exit(
            f"ERROR: no samples with raw + mask found in:\n"
            f"  {args.seg_folder}\n"
            f"  (channel {args.channel})"
        )

    print(f"[roi_editor] {len(samples)} samples found for channel {args.channel}")
    RoiEditorApp(args.seg_folder, args.channel).run()


if __name__ == "__main__":
    main()
