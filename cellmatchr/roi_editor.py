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
    Connected-components on the drawn layer → unique IDs for each blob.
    Only fills background pixels (existing == 0).

    Returns (new_mask, n_added, new_ids).
    """
    drawn_binary = (additions > 0).astype(np.uint8)
    labeled, n   = connected_components(drawn_binary, return_num=True, connectivity=2)
    if n == 0:
        return existing, 0, set()

    max_id   = int(existing.max())
    new_mask = existing.copy()
    new_ids  = set()
    added    = 0
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

    DEFAULT_BRUSH_SIZE = 18

    def __init__(self, seg_folder: str, channel: int, test_mode: bool = False):
        self.seg_folder = seg_folder
        self.channel    = channel
        self.test_mode  = test_mode
        self.samples    = _find_samples(seg_folder, channel)
        self.idx        = 0
        self._test_done = False     # flips True after first test-mode save

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

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self) -> None:
        import napari
        title = "CellMatchR — ROI Editor  [TEST MODE]" if self.test_mode \
                else "CellMatchR — ROI Editor"
        self.viewer = napari.Viewer(title=title)
        self._load_sample(self.idx)
        self.viewer.window.add_dock_widget(
            self._build_dock(), name="Navigation", area="right"
        )
        napari.run()

    # ── Dock widget ──────────────────────────────────────────────────────────

    def _build_dock(self):
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QPushButton, QLabel, QFrame,
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

        lay.addWidget(self._hr())

        # ── Instructions ──
        instr = QLabel(
            "<b>How to add missed cells:</b><br><br>"
            "1. Make sure <i>Add ROIs here</i> layer is selected<br>"
            "2. Press <b>P</b> for the brush<br>"
            "3. Paint over missed cells — <b>same colour is fine</b>, "
               "each disconnected blob becomes its own ROI<br>"
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
        raw = tifffile.imread(s["raw"]).astype(np.float32)
        msk = tifffile.imread(s["mask"]).astype(np.int32)
        lo, hi = np.percentile(raw, (1, 99))

        if self._raw_lyr is None:
            self._raw_lyr = self.viewer.add_image(
                raw, name="Raw Image", colormap="gray",
                contrast_limits=[lo, hi],
            )
            self._mask_lyr = self.viewer.add_labels(
                msk, name="Existing ROIs", opacity=0.45,
            )
            self._mask_lyr.editable = False
            self._add_lyr = self.viewer.add_labels(
                np.zeros_like(msk, dtype=np.int32),
                name="Add ROIs here", opacity=0.75,
            )
            self._add_lyr.brush_size     = self.DEFAULT_BRUSH_SIZE
            self._add_lyr.selected_label = 1
            self._add_lyr.mode           = "paint"
            self.viewer.layers.selection.active = self._add_lyr
        else:
            self._raw_lyr.data            = raw
            self._raw_lyr.contrast_limits = [lo, hi]
            self._mask_lyr.data           = msk
            self._add_lyr.data            = np.zeros_like(msk, dtype=np.int32)
            self._add_lyr.selected_label  = 1
            self._add_lyr.mode            = "paint"
            self.viewer.layers.selection.active = self._add_lyr

        self.viewer.reset_view()

    # ── Button callbacks ─────────────────────────────────────────────────────

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
        additions = self._add_lyr.data
        if additions.max() == 0:
            self._set_status("Nothing drawn — skipped.", colour="#aaaaaa")
            return

        s        = self.samples[self.idx]
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
        self._mask_lyr.data = new_mask
        self._add_lyr.data  = np.zeros_like(new_mask, dtype=np.int32)
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

        # 4. Update viewer for visual confirmation, disable save
        self._mask_lyr.data = new_mask
        self._add_lyr.data  = np.zeros_like(new_mask, dtype=np.int32)

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
