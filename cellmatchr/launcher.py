"""
CellMatchr Launcher — step-by-step wizard UI.
"""

import glob
import json
import os
import re
import sys
import subprocess
import threading
from datetime import datetime
from uuid import uuid4
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

from .config import (load_config, save_config, save_project_config,
                     load_project_config, CONFIG_PATH, DEFAULT_CONFIG)

# Optional PIL for logo
try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG       = "#1e1e2e"
SIDEBAR  = "#181825"
CARD     = "#313244"
FG       = "#cdd6f4"
ACCENT   = "#89b4fa"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
MUTED    = "#6c7086"
ACTIVE   = "#45475a"
ORANGE   = "#fab387"   # warm peach — "not started" state (less alarming than RED)

FONT       = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_STEP  = ("Segoe UI", 11)
FONT_MONO  = ("Courier New", 9)

STEPS = [
    ("1", "Create Project"),
    ("2", "Input & Paths"),
    ("3", "Segmentation"),
    ("4", "Validation"),
    ("5", "Analysis"),
    ("6", "Run Pipeline"),
]

# Spinner frames
_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Subfolder names written inside project_output_folder
SUBFOLDER = {
    "vsi":  "VSI_Converted_TIFF",
    "seg":  "Segmentation",
    "val":  "Validation",
    "ana":  "Analysis",
}


def _scanned_input_files(folder: str, extensions: tuple[str, ...],
                         skip_subdirs: bool = False) -> list[str]:
    """List the files segmentation scans: root and, optionally, one level down."""
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        if os.path.isfile(full) and entry.lower().endswith(extensions):
            paths.append(full)
        elif os.path.isdir(full) and not skip_subdirs:
            for name in sorted(os.listdir(full)):
                child = os.path.join(full, name)
                if os.path.isfile(child) and name.lower().endswith(extensions):
                    paths.append(child)
    return paths


def _first_confocal_preview_input(folder: str, extensions: tuple[str, ...],
                                  skip_subdirs: bool = False
                                  ) -> tuple[str, int | None] | None:
    """Pick a direct-read input and its confocal LIF scene, if applicable."""
    from .lif_classify import classify_lif_file

    for path in _scanned_input_files(folder, extensions, skip_subdirs):
        if path.lower().endswith(".lif"):
            try:
                result = classify_lif_file(path)
            except Exception:
                continue
            scene = next((index for index, _name, kind in result["scenes"]
                          if kind == "confocal"), None)
            if scene is not None:
                return path, scene
            continue
        return path, None
    return None


def _has_converter_roi_masks(folder: str) -> bool:
    """Check mask names without treating project-path characters as globs."""
    try:
        with os.scandir(folder) as files:
            return any(item.is_file() and item.name.endswith(
                "_mask_overview.tif") for item in files)
    except OSError:
        return False


def _has_analysis_roi_masks(folder: str) -> bool:
    """Accept the .tif and .roi files that analysis can read from a custom folder."""
    try:
        with os.scandir(folder) as files:
            return any(item.is_file() and item.name.endswith(
                (".tif", ".roi")) for item in files)
    except OSError:
        return False


def _segmentation_run_details(project_folder: str, run_name: str) -> str:
    """Summarize optional run metadata without excluding older projects."""
    progress_path = os.path.join(project_folder, run_name, "_run_progress.json")
    try:
        with open(progress_path) as source:
            identity = json.load(source).get("run_identity")
        if not isinstance(identity, dict):
            raise ValueError("run identity unavailable")
        backend = str(identity["backend"]).strip().lower()
        method = {"microsam": "Micro-SAM", "cellpose": "Cellpose-SAM"}.get(
            backend, backend)
        model = str(identity["model"])
        preprocess = str(identity["preprocess"])
        channels = ", ".join(str(channel)
                              for channel in identity["channels"])
        return (f"{method}  ·  model {model}  ·  {preprocess}  ·  "
                f"channels {channels or 'unspecified'}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return "Earlier run — method details unavailable"


# ---------------------------------------------------------------------------
# Conflict dialog — shown when an output folder already exists
# ---------------------------------------------------------------------------

class _ConflictDialog(tk.Toplevel):
    """Ask user: replace existing folder, or save with a custom suffix."""

    def __init__(self, parent: tk.Tk, folder_path: str,
                 suggested_suffix: str = "",
                 allow_replace: bool = True) -> None:
        super().__init__(parent)
        self.result = None          # None=cancel | "replace" | "<suffix string>"
        self._folder_path = folder_path
        self.transient(parent)
        self.grab_set()
        self.configure(bg=BG)
        self.title("Output already exists" if allow_replace else
                   "Save new segmentation run")
        self.resizable(False, False)

        folder_name = os.path.basename(folder_path)

        tk.Label(self, text=f"'{folder_name}' already exists.",
                 bg=BG, fg=FG, font=FONT_BOLD).pack(pady=(20, 4), padx=24)
        tk.Label(self, text=("Choose what to do:" if allow_replace else
                             "The existing masks and validation will be kept."),
                 bg=BG, fg=MUTED, font=FONT).pack(pady=(0, 14), padx=24)

        if allow_replace:
            tk.Button(
                self, text="Replace  (overwrite existing output)",
                command=self._replace,
                bg=RED, fg="white", font=FONT_BOLD, relief="flat",
                padx=14, pady=8,
            ).pack(fill="x", padx=24, pady=4)

            sep = tk.Frame(self, bg=MUTED, height=1)
            sep.pack(fill="x", padx=24, pady=10)

            tk.Label(self, text="— or save as a new folder —",
                     bg=BG, fg=MUTED, font=FONT).pack(pady=(0, 6))
        else:
            tk.Label(self, text="New folder name:",
                     bg=BG, fg=FG, font=FONT).pack(pady=(0, 6))

        row = tk.Frame(self, bg=BG)
        row.pack(padx=24, pady=4)
        tk.Label(row, text=f"{folder_name}_", bg=BG, fg=MUTED, font=FONT).pack(side="left")
        self._suffix_var = tk.StringVar(value=suggested_suffix)
        self._entry = tk.Entry(row, textvariable=self._suffix_var,
                               bg=CARD, fg=FG, insertbackground=FG,
                               relief="flat", font=FONT,
                               width=34 if not allow_replace else 16)
        self._entry.pack(side="left", padx=4)
        self._entry.focus_set()
        self._entry.bind("<Return>", lambda _: self._new_suffix())

        tk.Button(
            self, text=("Save with suffix" if allow_replace else
                        "Save as new run"),
            command=self._new_suffix,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
            padx=14, pady=8,
        ).pack(fill="x", padx=24, pady=4)

        tk.Button(
            self, text="Cancel", command=self._cancel,
            bg=CARD, fg=FG, font=FONT, relief="flat", padx=14, pady=6,
        ).pack(pady=(4, 20), padx=24)

        self._centre(parent)
        self.wait_window(self)

    def _replace(self):
        self.result = "replace"
        self.destroy()

    def _new_suffix(self):
        s = self._suffix_var.get().strip()
        if not s or s in (".", "..") or "/" in s or "\\" in s:
            self._entry.config(bg="#5a1e2e")
            return
        if os.path.exists(f"{self._folder_path}_{s}"):
            self._entry.config(bg="#5a1e2e")
            return
        self.result = s
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    def _centre(self, parent):
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h   = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + pw//2 - w//2}+{py + ph//2 - h//2}")


class _SegmentationRunDialog(tk.Toplevel):
    """Let the user choose the precise run before analysis starts."""

    def __init__(self, parent: tk.Tk, choices: list[str], initial: str,
                 project_folder: str) -> None:
        super().__init__(parent)
        self.result = None
        self.transient(parent)
        self.grab_set()
        self.configure(bg=BG)
        self.title("Choose segmentation run")
        self.resizable(False, False)

        tk.Label(self, text="Which segmentation run should be analyzed?",
                 bg=BG, fg=FG, font=FONT_BOLD).pack(padx=24, pady=(20, 10))
        self._choice = tk.StringVar(value=initial if initial in choices else choices[0])
        picker = ttk.Combobox(self, textvariable=self._choice, values=choices,
                              state="readonly", width=48)
        picker.pack(fill="x", padx=24, pady=(0, 6))
        picker.focus_set()
        details = tk.Label(
            self,
            text=_segmentation_run_details(project_folder, self._choice.get()),
            bg=BG, fg=MUTED, font=FONT, wraplength=440, justify="left",
        )
        details.pack(anchor="w", padx=24, pady=(0, 14))
        self._choice.trace_add(
            "write", lambda *_: details.config(text=_segmentation_run_details(
                project_folder, self._choice.get())))

        buttons = tk.Frame(self, bg=BG)
        buttons.pack(padx=24, pady=(0, 20))
        tk.Button(buttons, text="Use selected run", command=self._accept,
                  bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
                  padx=12, pady=7).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Cancel", command=self.destroy,
                  bg=CARD, fg=FG, font=FONT, relief="flat",
                  padx=12, pady=7).pack(side="left")
        self.bind("<Return>", lambda _event: self._accept())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.update_idletasks()
        self.geometry(
            f"+{parent.winfo_x() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)}"
            f"+{parent.winfo_y() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)}"
        )
        self.wait_window(self)

    def _accept(self) -> None:
        self.result = self._choice.get()
        self.destroy()


class _ExistingSegmentationChoiceDialog(tk.Toplevel):
    """Choose what Run Pipeline should do when masks already exist."""

    def __init__(self, parent: tk.Tk, input_folder: str,
                 analysis_enabled: bool) -> None:
        super().__init__(parent)
        self.result = None
        self.transient(parent)
        self.grab_set()
        self.configure(bg=BG)
        self.title("Segmentation already exists")
        self.resizable(False, False)

        tk.Label(self, text="Segmentation masks already exist in this project.",
                 bg=BG, fg=FG, font=FONT_BOLD).pack(
            anchor="w", padx=24, pady=(20, 8))
        tk.Label(self, text=f"Current input: {input_folder or 'not selected'}",
                 bg=BG, fg=MUTED, font=FONT, wraplength=480,
                 justify="left").pack(anchor="w", padx=24, pady=(0, 12))

        tk.Button(
            self, text="Segment current input as a new run",
            command=lambda: self._choose("new"), bg=ACCENT, fg="#1e1e2e",
            font=FONT_BOLD, relief="flat", padx=14, pady=9,
        ).pack(fill="x", padx=24, pady=4)
        tk.Label(self, text="Keeps existing masks and saves this input separately.",
                 bg=BG, fg=MUTED, font=FONT).pack(
            anchor="w", padx=24, pady=(0, 10))

        tk.Button(
            self, text="Analyze an existing segmentation run",
            command=lambda: self._choose("existing"),
            state="normal" if analysis_enabled else "disabled",
            bg=CARD, fg=FG, font=FONT_BOLD, relief="flat", padx=14, pady=9,
        ).pack(fill="x", padx=24, pady=4)
        if not analysis_enabled:
            tk.Label(self, text="Enable Analysis in Step 5 to use this option.",
                     bg=BG, fg=MUTED, font=FONT).pack(
                anchor="w", padx=24, pady=(0, 8))

        cancel = tk.Button(self, text="Cancel", command=self.destroy,
                           bg=BG, fg=FG, font=FONT, relief="flat",
                           padx=14, pady=7)
        cancel.pack(pady=(8, 20))
        cancel.focus_set()
        self.bind("<Escape>", lambda _event: self.destroy())
        self.update_idletasks()
        self.geometry(
            f"+{parent.winfo_x() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)}"
            f"+{parent.winfo_y() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)}"
        )
        self.wait_window(self)

    def _choose(self, choice: str) -> None:
        self.result = choice
        self.destroy()


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------

class CellMatchrLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CellMatchR")
        self._set_window_icon()
        self.root.configure(bg=BG)
        self.root.geometry("1150x800")
        self.root.minsize(900, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.config = DEFAULT_CONFIG.copy()  # always start clean
        self.current_step = 0
        self._spinning = False
        self._spin_idx  = 0
        # Live background processes (segmentation/analysis runs, validation
        # windows). Used to warn before navigating away mid-run.
        self._active_procs: list = []

        # per-step input-variant selections (populated on Run step)
        self._val_input_var: tk.StringVar | None = None
        self._ana_input_var: tk.StringVar | None = None
        # validation channel selection (populated in Step 4)
        self._val_channel_var: tk.StringVar | None = None
        # validation channel name entries, colors, and preview (populated in Step 4)
        self._val_ch_name_vars: list = []
        self._val_ch_color_vars: list = []
        self._val_ch_names_frame: tk.Frame | None = None
        self._val_ch_images: list = []
        self._val_ch_photo_refs: list = []
        self._val_ch_index: int = 0
        self._val_lbl: tk.Label | None = None
        self._val_preview_img: tk.Label | None = None
        self._val_prev_btn: tk.Button | None = None
        self._val_next_btn: tk.Button | None = None
        self._val_run_btn: tk.Button | None = None
        self._val_spin_job: str | None = None
        self._review_btn: tk.Button | None = None
        self._review_status: tk.Label | None = None
        self._review_test_mode: tk.BooleanVar | None = None
        # analysis channel selections (populated in Step 5)
        self._ana_coi_chan_var: tk.StringVar | None = None
        self._ana_pm_chan_var: tk.StringVar | None = None
        # open-existing project JSON path (populated in Step 1 open pane)
        self._open_json_var: tk.StringVar = tk.StringVar()

        self._vars: dict = {}
        self._build_vars()
        self._build_layout()
        self._show_step(0)

    # ------------------------------------------------------------------
    # Variable binding
    # ------------------------------------------------------------------

    def _build_vars(self) -> None:
        str_fields = [
            "project_name", "project_output_folder",
            "input_type", "input_root_folder",
            "seg_channels", "seg_backend", "seg_model_type", "seg_preprocess_mode",
            "val_mask_pattern", "val_channel_names", "val_channel_colors",
            "ana_red_channel", "ana_pv_channel", "ana_iou_threshold",
            "ana_roi_cell_rule", "ana_roi_mask_folder", "ana_subfolder_name",
        ]
        num_fields = [
            "seg_min_object_size", "seg_max_object_size",
            "seg_tile_h", "seg_tile_w", "seg_halo_h", "seg_halo_w",
            "seg_qc_downsample", "seg_max_ratio", "seg_raw_mean_min",
            "seg_cellpose_diameter", "seg_cellpose_flow_threshold",
            "seg_cellpose_cellprob_threshold",
        ]
        bool_fields = [
            "seg_test_run", "seg_dry_run", "seg_skip_subdirs", "seg_resume",
            "val_enabled", "ana_enabled", "ana_coloc_enabled", "ana_save_qc",
            "apply_partial_validation",
        ]
        for k in str_fields:
            self._vars[k] = tk.StringVar(value=str(self.config.get(k, "")))
        for k in num_fields:
            self._vars[k] = tk.StringVar(value=str(self.config.get(k, "")))
        for k in bool_fields:
            self._vars[k] = tk.BooleanVar(value=bool(self.config.get(k, False)))

    def _sync_vars_to_config(self) -> None:
        for k, v in self._vars.items():
            raw = v.get()
            orig = self.config.get(k)
            if isinstance(orig, bool):
                self.config[k] = bool(raw)
            elif isinstance(orig, int):
                try:
                    self.config[k] = int(raw)
                except (ValueError, TypeError):
                    pass
            elif isinstance(orig, float):
                try:
                    self.config[k] = float(raw)
                except (ValueError, TypeError):
                    pass
            else:
                self.config[k] = raw

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _set_window_icon(self) -> None:
        """Use CellMatchr_Logo.png as the window / Dock / taskbar icon."""
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "CellMatchr_Logo.png",
        )
        if not (_PIL and os.path.exists(logo_path)):
            return
        try:
            img = Image.open(logo_path)
            # Keep a reference so Tk doesn't garbage-collect the image.
            self._icon_img = ImageTk.PhotoImage(img)
            self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

        # ── Sidebar ──────────────────────────────────────────────────
        self.sidebar = tk.Frame(self.root, bg=SIDEBAR, width=220)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsw")
        self.sidebar.grid_propagate(False)

        # Logo
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "CellMatchr_Logo.png",
        )
        if _PIL and os.path.exists(logo_path):
            img = Image.open(logo_path).resize((120, 120), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(img)
            tk.Label(self.sidebar, image=self._logo_img,
                     bg=SIDEBAR).pack(pady=(18, 4))
        else:
            tk.Frame(self.sidebar, bg=SIDEBAR, height=12).pack()

        # Title — two-tone seamless text via Canvas
        _fnt_title = ("Segoe UI", 18, "bold")
        _title_cv = tk.Canvas(self.sidebar, bg=SIDEBAR,
                               highlightthickness=0, bd=0,
                               width=160, height=28)
        _t1 = _title_cv.create_text(0, 0, text="Cell",   fill=ACCENT, font=_fnt_title, anchor="nw")
        _b1 = _title_cv.bbox(_t1)
        _t2 = _title_cv.create_text(_b1[2], 0, text="MatchR", fill=GREEN,  font=_fnt_title, anchor="nw")
        _b2 = _title_cv.bbox(_t2)
        _title_cv.config(width=_b2[2], height=max(_b1[3], _b2[3]))
        _title_cv.pack(pady=(0, 2))
        tk.Label(self.sidebar, text="confocal image pipeline",
                 bg=SIDEBAR, fg=MUTED, font=("Segoe UI", 8)).pack(pady=(0, 16))

        ttk.Separator(self.sidebar).pack(fill="x", padx=16, pady=4)

        # Step buttons
        self._step_btns: list[tk.Button] = []
        for i, (num, name) in enumerate(STEPS):
            btn = tk.Button(
                self.sidebar,
                text=f"  {num}   {name}",
                bg=SIDEBAR, fg=MUTED, font=FONT_STEP,
                anchor="w", relief="flat", bd=0,
                activebackground=ACTIVE, activeforeground=FG,
                command=lambda idx=i: self._show_step(idx),
            )
            btn.pack(fill="x", padx=8, ipady=7, pady=2)
            self._step_btns.append(btn)

        # ── Project info panel (bottom of sidebar) ───────────────────
        ttk.Separator(self.sidebar).pack(side="bottom", fill="x", padx=16, pady=4)
        self._proj_folder_lbl = tk.Label(
            self.sidebar, text="—", bg=SIDEBAR, fg=MUTED,
            font=("Segoe UI", 8), wraplength=190, justify="left",
        )
        self._proj_folder_lbl.pack(side="bottom", anchor="w", padx=14, pady=(0, 4))
        self._proj_name_lbl = tk.Label(
            self.sidebar, text="No project set", bg=SIDEBAR, fg=FG,
            font=("Segoe UI", 9, "bold"), wraplength=190, justify="left",
        )
        self._proj_name_lbl.pack(side="bottom", anchor="w", padx=14, pady=(0, 2))
        tk.Label(self.sidebar, text="PROJECT", bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 7)).pack(side="bottom", anchor="w", padx=14, pady=(6, 0))
        tk.Label(self.sidebar, text="v0.1.0", bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 7)).pack(side="bottom", pady=4)

        # ── Scrollable content area ───────────────────────────────────
        content_outer = tk.Frame(self.root, bg=BG)
        content_outer.grid(row=0, column=1, sticky="nsew")
        content_outer.grid_rowconfigure(0, weight=1)
        content_outer.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(content_outer, bg=BG, highlightthickness=0,
                                  yscrollincrement=20)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(content_outer, orient="vertical",
                             command=self._canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=vsb.set)

        self.content_frame = tk.Frame(self._canvas, bg=BG)
        self._cwin = self._canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self.content_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")
            ),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._cwin, width=e.width),
        )
        self.root.bind_all("<MouseWheel>", self._on_scroll)
        self.root.bind_all("<Button-4>",   self._on_scroll)
        self.root.bind_all("<Button-5>",   self._on_scroll)

        # ── Bottom nav bar ────────────────────────────────────────────
        nav = tk.Frame(self.root, bg=SIDEBAR, height=58)
        nav.grid(row=1, column=1, sticky="ew")
        nav.grid_propagate(False)

        self.btn_back = tk.Button(
            nav, text="◀  Back", command=self._back,
            bg=CARD, fg=FG, font=FONT_BOLD, relief="flat",
            padx=18, pady=8,
            activebackground=ACTIVE, activeforeground=FG,
        )
        self.btn_back.pack(side="left", padx=14, pady=10)

        self.btn_next = tk.Button(
            nav, text="Next  ▶", command=self._next,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
            padx=18, pady=8,
        )
        self.btn_next.pack(side="right", padx=14, pady=10)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        if event.num == 4:                            # Linux scroll-up
            self._canvas.yview_scroll(-2, "units")
        elif event.num == 5:                          # Linux scroll-down
            self._canvas.yview_scroll(2, "units")
        elif event.delta:
            if abs(event.delta) >= 120:
                # Windows mouse wheel: ±120 per detent → 3 units (60 px)
                self._canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
            else:
                # macOS trackpad: delta ±1-5, yscrollincrement=20 → 20-100 px/event
                self._canvas.yview_scroll(-event.delta, "units")

    def _bind_combobox_scroll(self, cb: "ttk.Combobox") -> None:
        """Prevent the combobox from cycling its value on mousewheel.
        Instead, forward the event to the canvas scroll and swallow it."""
        def _cb_scroll(event):
            self._on_scroll(event)
            return "break"
        cb.bind("<MouseWheel>", _cb_scroll)
        cb.bind("<Button-4>",   _cb_scroll)
        cb.bind("<Button-5>",   _cb_scroll)

    # ------------------------------------------------------------------
    # Background-process tracking + leave-while-running guard
    # ------------------------------------------------------------------

    def _track_proc(self, proc, label: str) -> None:
        """Register a launched subprocess so we can warn before leaving it."""
        try:
            self._active_procs.append((proc, label))
        except Exception:
            pass

    def _running_procs(self) -> list:
        """Return [(proc, label), …] for processes still alive; prune the rest."""
        live = []
        for proc, label in list(self._active_procs):
            try:
                if proc.poll() is None:
                    live.append((proc, label))
            except Exception:
                pass
        self._active_procs = live
        return live

    def _is_running(self, label: str) -> bool:
        """True if a tracked process with this label is still alive."""
        return any(lbl == label for _, lbl in self._running_procs())

    def _terminate_running(self) -> None:
        """Terminate (then kill) every tracked process that is still alive."""
        for proc, _ in self._running_procs():
            try:
                proc.terminate()
            except Exception:
                pass
        for proc, _ in list(self._active_procs):
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._active_procs = []

    def _warn_yesno(self, title: str, message: str,
                    yes_text: str = "Yes", no_text: str = "No") -> bool:
        """Modal warning dialog. 'No' is the highlighted default and sits on
        the RIGHT (so Enter / the obvious button is the safe choice); 'Yes'
        (the destructive action) is de-emphasised on the left.  Returns True
        only if the user explicitly chooses Yes."""
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=CARD)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        result = {"v": False}

        tk.Label(dlg, text="⚠", bg=CARD, fg=RED,
                 font=("Segoe UI", 34, "bold")).pack(pady=(18, 2))
        tk.Label(dlg, text=title, bg=CARD, fg=RED,
                 font=("Segoe UI", 12, "bold")).pack(padx=28)
        tk.Label(dlg, text=message, bg=CARD, fg=FG, font=FONT,
                 justify="left", wraplength=440).pack(padx=28, pady=(10, 18))

        def _yes():
            result["v"] = True
            dlg.destroy()

        def _no():
            result["v"] = False
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=CARD)
        btn_row.pack(pady=(0, 20))
        # Yes (destructive) on the LEFT, muted; No (safe default) on the RIGHT.
        tk.Button(btn_row, text=yes_text, command=_yes,
                  bg=ACTIVE, fg=MUTED, font=FONT, relief="flat",
                  activebackground=ACTIVE, activeforeground=FG,
                  padx=16, pady=7).pack(side="left", padx=(0, 12))
        no_btn = tk.Button(btn_row, text=no_text, command=_no,
                           bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
                           activebackground=ACCENT, padx=20, pady=7)
        no_btn.pack(side="left")

        no_btn.focus_set()
        dlg.bind("<Return>", lambda _e: _no())
        dlg.bind("<Escape>", lambda _e: _no())
        dlg.protocol("WM_DELETE_WINDOW", _no)

        dlg.update_idletasks()
        x = self.root.winfo_rootx() + max(
            (self.root.winfo_width() - dlg.winfo_width()) // 2, 0)
        y = self.root.winfo_rooty() + max(
            (self.root.winfo_height() - dlg.winfo_height()) // 3, 0)
        dlg.geometry(f"+{x}+{y}")
        dlg.lift()
        dlg.attributes("-topmost", True)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return result["v"]

    def _confirm_leave_running(self) -> bool:
        """If a process is running, require TWO confirmations before leaving.
        Returns True if it is safe to navigate (nothing running, or the user
        confirmed twice — in which case the process is terminated)."""
        running = self._running_procs()
        if not running:
            return True
        labels = "\n      • ".join(sorted({lbl for _, lbl in running}))
        first = self._warn_yesno(
            "PROCESS IN PROGRESS",
            "A process is currently running:\n\n"
            f"      • {labels}\n\n"
            "Leaving this page will CANCEL the running process and you may lose "
            "the results of the current run.\n\n"
            "Do you want to cancel it and move to another page?",
            yes_text="Yes, cancel it", no_text="No, stay here",
        )
        if not first:
            return False
        second = self._warn_yesno(
            "ARE YOU ABSOLUTELY SURE?",
            "❗  This will TERMINATE the running process immediately.\n\n"
            "Any unfinished work from this run will be lost and cannot be "
            "recovered.\n\n"
            "Terminate the running process and switch pages?",
            yes_text="Yes, terminate", no_text="No, go back",
        )
        if not second:
            return False
        self._terminate_running()
        return True

    def _on_close(self) -> None:
        """Window-close (X) handler — single warning if a process is running."""
        running = self._running_procs()
        if running:
            labels = "\n      • ".join(sorted({lbl for _, lbl in running}))
            if not self._warn_yesno(
                "PROCESS IN PROGRESS",
                "A process is currently running:\n\n"
                f"      • {labels}\n\n"
                "Closing CellMatchR will terminate it and you may lose the "
                "results of the current run.\n\n"
                "Close anyway?",
                yes_text="Yes, close", no_text="No, stay open",
            ):
                return
            self._terminate_running()
        self.root.quit()

    def _show_step(self, idx: int) -> None:
        # Guard against navigating away while a run/window is in progress.
        if idx != self.current_step and not self._confirm_leave_running():
            return
        self._sync_vars_to_config()
        save_config(self.config)
        save_project_config(self.config)
        self.current_step = idx

        for i, btn in enumerate(self._step_btns):
            btn.config(bg=ACTIVE if i == idx else SIDEBAR,
                       fg=FG    if i == idx else MUTED)

        for w in self.content_frame.winfo_children():
            w.destroy()

        builders = [
            self._build_project_step,
            self._build_input_step,
            self._build_segmentation_step,
            self._build_validation_step,
            self._build_analysis_step,
            self._build_run_step,
        ]
        builders[idx](self.content_frame)

        self.btn_back.config(state="normal" if idx > 0 else "disabled")
        if idx == len(STEPS) - 1:
            self.btn_next.config(text="Save & Close", command=self._save_and_close)
        else:
            self.btn_next.config(text="Next  ▶", command=self._next)

        self._canvas.yview_moveto(0)
        self._update_project_display()

    def _next(self):
        if self.current_step < len(STEPS) - 1:
            self._show_step(self.current_step + 1)

    def _back(self):
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _save_and_close(self):
        if not self._confirm_leave_running():
            return
        self._sync_vars_to_config()
        save_config(self.config)
        self.root.quit()

    def _update_project_display(self) -> None:
        name   = self.config.get("project_name", "").strip() or "No project set"
        folder = self.config.get("project_output_folder", "").strip()
        short  = os.path.basename(folder) if folder else "—"
        self._proj_name_lbl.config(text=name)
        self._proj_folder_lbl.config(text=short)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _section_header(self, parent, title: str, subtitle: str = "") -> None:
        tk.Label(parent, text=title, bg=BG, fg=FG,
                 font=FONT_TITLE, anchor="w").pack(
            fill="x", padx=32, pady=(32, 4)
        )
        if subtitle:
            tk.Label(parent, text=subtitle, bg=BG, fg=MUTED,
                     font=FONT, anchor="w", wraplength=700).pack(
                fill="x", padx=32, pady=(0, 12)
            )
        ttk.Separator(parent).pack(fill="x", padx=32, pady=(0, 20))

    def _card(self, parent, title: str = "") -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD, padx=20, pady=14)
        outer.pack(fill="x", padx=32, pady=6)
        if title:
            tk.Label(outer, text=title, bg=CARD, fg=ACCENT,
                     font=FONT_BOLD).pack(anchor="w", pady=(0, 8))
        return outer

    def _form_grid(self, card: tk.Frame, rows: list,
                   col0_width: int = 240) -> tk.Frame:
        g = tk.Frame(card, bg=CARD)
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=col0_width)
        g.grid_columnconfigure(1, weight=1)
        for i, (label, key) in enumerate(rows):
            tk.Label(g, text=label, bg=CARD, fg=FG, font=FONT).grid(
                row=i, column=0, sticky="w", pady=3
            )
            tk.Entry(g, textvariable=self._vars[key], bg=BG, fg=FG,
                     insertbackground=FG, relief="flat", font=FONT, width=22).grid(
                row=i, column=1, sticky="w", pady=3, padx=8
            )
        return g

    def _path_row(self, card: tk.Frame, label: str, key: str) -> None:
        tk.Label(card, text=label, bg=CARD, fg=FG, font=FONT).pack(
            anchor="w", pady=(6, 2)
        )
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x")
        tk.Entry(row, textvariable=self._vars[key], bg=BG, fg=FG,
                 insertbackground=FG, relief="flat", font=FONT, width=56).pack(
            side="left"
        )
        tk.Button(
            row, text="Browse",
            command=lambda k=key: self._browse_dir(k),
            bg=CARD, fg=MUTED, relief="flat", font=FONT, padx=6,
        ).pack(side="left", padx=4)

    def _format_status_row(self, parent, ext: str, brand: str, status: str,
                            note: str | None = None) -> None:
        """One row in the Step 2 'Supported File Types' list, e.g.
        '✔  .czi   Zeiss   tested'. status is 'tested' or 'beta'."""
        icon, color = {"tested": ("✔", GREEN), "beta": ("🧪", YELLOW)}.get(status, ("?", MUTED))
        row = tk.Frame(parent, bg=CARD)
        row.pack(anchor="w", fill="x", pady=1)
        tk.Label(row, text=icon, bg=CARD, fg=color, font=FONT_BOLD, width=2).pack(side="left")
        tk.Label(row, text=ext, bg=CARD, fg=FG, font=FONT_MONO, width=13, anchor="w").pack(side="left")
        tk.Label(row, text=brand, bg=CARD, fg=FG, font=FONT, anchor="w", width=20).pack(side="left")
        tk.Label(row, text=("tested" if status == "tested" else "beta — not yet validated on real files"),
                 bg=CARD, fg=color, font=FONT, anchor="w").pack(side="left")
        if note:
            tk.Label(parent, text=f"      {note}", bg=CARD, fg=MUTED, font=(FONT[0], 8),
                     justify="left", wraplength=650).pack(anchor="w", pady=(0, 4))

    def _refresh_input_file_preview(self) -> None:
        """List files found in the confocal input folder, flagging any .lif
        that will be skipped because it's actually slide-scanner data (Leica
        writes both to the same extension — see cellmatchr/lif_classify.py)."""
        frame = getattr(self, "_input_preview_frame", None)
        if frame is None:
            return
        try:
            if not frame.winfo_exists():
                return
            for w in frame.winfo_children():
                w.destroy()

            folder = self._vars["input_root_folder"].get().strip()
            if not folder or not os.path.isdir(folder):
                return

            skip_subdirs = bool(self._vars["seg_skip_subdirs"].get())
            files = _scanned_input_files(
                folder, self._INPUT_EXTS, skip_subdirs)
            if not files:
                return

            ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(2, 8))
            scope = "this folder" if skip_subdirs else "this folder and one level down"
            tk.Label(frame, text=f"Files found in {scope} ({len(files)}):",
                     bg=CARD, fg=MUTED, font=FONT).pack(anchor="w", pady=(0, 4))

            from .lif_classify import classify_lif_file

            max_shown = 40
            for full in files[:max_shown]:
                display_name = os.path.relpath(full, folder)
                if full.lower().endswith(".lif"):
                    try:
                        ftype = classify_lif_file(full).get("file_type", "unknown")
                    except Exception:
                        ftype = "unknown"
                    if ftype == "slide_scanner":
                        text, color = f"⊘  {display_name}   —  slide-scanner LIF, skipped here; use Slide scanner mode", MUTED
                    elif ftype == "mixed":
                        text, color = f"⚠  {display_name}   —  mixed LIF; only confocal images are used here", YELLOW
                    elif ftype == "confocal":
                        text, color = f"✔  {display_name}   —  confocal LIF", FG
                    else:
                        text, color = f"?  {display_name}   —  could not read LIF metadata; preview unavailable", MUTED
                else:
                    text, color = f"✔  {display_name}", FG
                tk.Label(frame, text=text, bg=CARD, fg=color, font=FONT_MONO,
                         anchor="w", justify="left", wraplength=680).pack(anchor="w")

            if len(files) > max_shown:
                tk.Label(frame, text=f"…and {len(files) - max_shown} more",
                         bg=CARD, fg=MUTED, font=FONT).pack(anchor="w", pady=(2, 0))
        except (tk.TclError, AttributeError):
            pass

    def _project_has_outputs(self) -> bool:
        """True if the current project folder already contains segmentation,
        validation, or analysis output folders (i.e. work that would be
        orphaned by switching the input folder)."""
        return bool(
            self._find_variants(SUBFOLDER["seg"])
            or self._find_variants(SUBFOLDER["val"])
            or self._find_variants(SUBFOLDER["ana"])
        )

    def _confirm_input_folder_change(self) -> bool:
        """Two-stage warning shown before the input folder of an existing
        project is changed.  Returns True only if the user confirms BOTH."""
        first = messagebox.askyesno(
            "Switching the input folder?",
            "This project already has segmentation / validation / analysis "
            "output saved in its working directory.\n\n"
            "Changing the input folder will break the link to those existing "
            "results — many project settings (segmentation channels, validation "
            "paths, etc.) will no longer match what is on disk and the project "
            "file can lose track of them.\n\n"
            "If you want to work on a different dataset, it is strongly "
            "recommended to START A NEW, SEPARATE PROJECT instead.\n\n"
            "Do you still want to change the input folder?",
            icon="warning", default="no", parent=self.root,
        )
        if not first:
            return False
        second = messagebox.askyesno(
            "Are you absolutely sure?",
            "There is NO GOING BACK.\n\n"
            "Once you pick a new input folder and save, the project file will "
            "point to the new data and the previous configuration may be "
            "overwritten. The old segmentation/validation outputs will remain "
            "on disk but this project will no longer reference them.\n\n"
            "Proceed with changing the input folder?",
            icon="warning", default="no", parent=self.root,
        )
        return bool(second)

    def _browse_dir(self, key: str) -> None:
        # Guard the input folder of an existing project behind two warnings,
        # so it can't be reset by accident (which would orphan saved outputs).
        if (key == "input_root_folder"
                and self.config.get("input_root_folder", "").strip()
                and self._project_has_outputs()):
            if not self._confirm_input_folder_change():
                return
        self.root.lift()
        self.root.attributes("-topmost", True)
        p = filedialog.askdirectory(parent=self.root)
        self.root.attributes("-topmost", False)
        if p:
            self._vars[key].set(p)

    # ------------------------------------------------------------------
    # Spinner helpers
    # ------------------------------------------------------------------

    def _spinner_start(self, label: tk.Label) -> None:
        self._spinning = True
        self._spin_idx = 0

        def _tick():
            if not self._spinning:
                label.config(text="")
                return
            label.config(text=_SPIN[self._spin_idx % len(_SPIN)])
            self._spin_idx += 1
            label.after(80, _tick)

        _tick()

    def _spinner_stop(self, label: tk.Label) -> None:
        self._spinning = False
        label.config(text="")

    # ------------------------------------------------------------------
    # Variant helpers (detect existing output subfolders)
    # ------------------------------------------------------------------

    def _find_variants(self, base: str) -> list:
        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            return []
        results = []
        for name in sorted(os.listdir(proj)):
            full = os.path.join(proj, name)
            if os.path.isdir(full) and (name == base or name.startswith(base + "_")):
                results.append(name)
        return results

    def _resolve_output_dir(self, base: str,
                            suggested_suffix: str = "") -> str | None:
        """
        Check if base subfolder exists inside project_output_folder.
        If it does, show conflict dialog.
        Returns the resolved folder path, or None if user cancelled.
        """
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set. Go to Step 1 — Create Project.")
            return None
        candidate = os.path.join(proj, base)
        if os.path.isdir(candidate):
            dlg = _ConflictDialog(self.root, candidate, suggested_suffix)
            if dlg.result is None:
                return None
            if dlg.result == "replace":
                return candidate
            return os.path.join(proj, f"{base}_{dlg.result}")
        return candidate

    def _suggest_segmentation_suffix(self) -> str:
        """Give a new run a recognizable, collision-free name by default."""
        backend = str(self.config.get("seg_backend", "microsam")).strip().lower()
        if backend not in ("microsam", "cellpose"):
            backend = "microsam"
        stem = f"{backend}_{datetime.now():%Y%m%d_%H%M%S}"
        proj = self.config.get("project_output_folder", "").strip()
        suffix = stem
        number = 2
        while os.path.exists(os.path.join(proj, f"{SUBFOLDER['seg']}_{suffix}")):
            suffix = f"{stem}_{number}"
            number += 1
        return suffix

    def _resolve_segmentation_output_dir(self, auto_new: bool = False
                                         ) -> str | None:
        """Create a separate new run whenever an existing run has content."""
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set. Go to Step 1.")
            return None
        base = os.path.join(proj, SUBFOLDER["seg"])
        if not os.path.lexists(base):
            return base
        if os.path.isdir(base) and not os.path.islink(base) and not os.listdir(base):
            return base
        if auto_new:
            return f"{base}_{self._suggest_segmentation_suffix()}"
        dialog = _ConflictDialog(
            self.root, base, self._suggest_segmentation_suffix(),
            allow_replace=False,
        )
        if dialog.result is None or dialog.result == "replace":
            return None
        return f"{base}_{dialog.result}"

    def _segmentation_identity_from_config(self) -> dict:
        """Match the identity stored in a new segmentation progress file."""
        backend = str(self.config.get("seg_backend") or "microsam").strip().lower()
        model = (str(self.config.get("seg_model_type") or "vit_b_lm")
                 if backend == "microsam" else "cpsam")
        raw_channels = str(self.config.get("seg_channels") or "")
        channels = [int(part.strip()) for part in raw_channels.split(",")
                    if part.strip()]
        return {
            "backend": backend,
            "model": model,
            "preprocess": str(self.config.get("seg_preprocess_mode") or
                              "local_bg_subtract"),
            "channels": channels,
        }

    def _choose_analysis_run(self, *, ask_if_multiple: bool = False) -> str | None:
        """Choose an existing segmentation folder without guessing its method."""
        choices = self._find_variants(SUBFOLDER["seg"])
        if not choices:
            return None
        from_card = (self._ana_input_var.get()
                     if self._ana_input_var is not None else "")
        saved = str(self.config.get("seg_input_subfolder", "")).strip()
        selected = next((name for name in (from_card, saved) if name in choices),
                        None)
        if len(choices) > 1 and (ask_if_multiple or selected is None):
            dialog = _SegmentationRunDialog(
                self.root, choices, selected or choices[0],
                self.config.get("project_output_folder", "").strip())
            if dialog.result is None:
                return None
            selected = dialog.result
        if selected is None:
            selected = choices[0]
        if self._ana_input_var is not None:
            self._ana_input_var.set(selected)
        self.config["seg_input_subfolder"] = selected
        return selected

    def _analysis_output_dir(self, segmentation_run: str) -> str:
        """Keep legacy paths while pairing new run variants by suffix."""
        proj = self.config.get("project_output_folder", "").strip()
        seg_base = SUBFOLDER["seg"]
        if segmentation_run == seg_base:
            name = SUBFOLDER["ana"]
        elif segmentation_run.startswith(seg_base + "_"):
            name = SUBFOLDER["ana"] + segmentation_run[len(seg_base):]
        else:
            raise ValueError(f"Unknown segmentation run: {segmentation_run}")
        return os.path.join(proj, name)

    def _segmentation_source(self, segmentation_run: str) -> tuple[str, str]:
        """Return the selected run's input kind and converter variant, if known."""
        proj = self.config.get("project_output_folder", "").strip()
        progress = os.path.join(proj, segmentation_run, "_run_progress.json")
        try:
            with open(progress) as source:
                inputs = json.load(source).get("input_sources")
            if not isinstance(inputs, dict) or "converter_run" not in inputs:
                return "unknown", ""
            variant = inputs["converter_run"]
            if variant == "":
                return "direct", ""
            if (isinstance(variant, str)
                    and (variant == SUBFOLDER["vsi"]
                         or variant.startswith(SUBFOLDER["vsi"] + "_"))
                    and os.path.basename(variant) == variant):
                return "converted", variant
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return "unknown", ""

    def _converter_roi_paths(self) -> set[str]:
        proj = self.config.get("project_output_folder", "").strip()
        outputs = [os.path.join(proj, name)
                   for name in self._find_variants(SUBFOLDER["vsi"])]
        outputs.append(os.path.join(proj, SUBFOLDER["vsi"]))
        saved = str(self.config.get("vsi_output_dir") or "").strip()
        if saved:
            outputs.append(saved)
        return {os.path.normpath(os.path.join(
            output, "DONOTDELETE-MASK_FOR_VSI")) for output in outputs}

    def _set_roi_mask_folder(self, folder: str) -> None:
        self.config["ana_roi_mask_folder"] = folder
        if "ana_roi_mask_folder" in self._vars:
            self._vars["ana_roi_mask_folder"].set(folder)

    def _pair_roi_masks_with_run(self, segmentation_run: str,
                                 confirm_unknown: bool = False) -> bool:
        """Choose ROI masks from the analyzed run, never a different input."""
        kind, converter_run = self._segmentation_source(segmentation_run)
        current = str(self.config.get("ana_roi_mask_folder") or "").strip()
        is_auto = bool(current and os.path.normpath(current)
                       in self._converter_roi_paths())
        if kind == "direct":
            if is_auto:
                self._set_roi_mask_folder("")
            return True
        if kind == "converted":
            proj = self.config.get("project_output_folder", "").strip()
            matched = os.path.join(proj, converter_run,
                                   "DONOTDELETE-MASK_FOR_VSI")
            if (current and not is_auto
                    and os.path.normpath(current) != os.path.normpath(matched)
                    and confirm_unknown):
                choice = messagebox.askyesnocancel(
                    "ROI folder differs from this run",
                    f"The selected run '{segmentation_run}' came from "
                    f"{converter_run}. Its matching ROI masks are in:\n"
                    f"{matched}\n\n"
                    f"The current custom folder is:\n{current}\n\n"
                    "Yes: use the custom folder. No: use the matching folder. "
                    "Cancel: stop analysis.",
                    parent=self.root, default=messagebox.CANCEL,
                    icon="warning",
                )
                if choice is None:
                    return False
                if choice is False:
                    current = ""
                elif not _has_analysis_roi_masks(current):
                    if not messagebox.askyesno(
                            "ROI masks are missing",
                            f"No .tif or .roi masks were found in the custom "
                            f"folder:\n{current}\n\n"
                            "Continue with whole-image analysis? Yes clears "
                            "the ROI folder; No cancels so you can choose "
                            "the correct masks in Step 5.",
                            parent=self.root, default=messagebox.NO,
                            icon="warning"):
                        return False
                    self._set_roi_mask_folder("")
                    return True
            if not current or is_auto:
                if confirm_unknown and not _has_converter_roi_masks(matched):
                    if not messagebox.askyesno(
                            "ROI masks are missing",
                            f"The selected run '{segmentation_run}' came from "
                            f"{converter_run}, but no ROI masks were found in:\n"
                            f"{matched}\n\n"
                            "Continue with whole-image analysis? Yes continues "
                            "without ROI-specific counts. No cancels so you "
                            "can restore or select the masks in Step 5.",
                            parent=self.root, default=messagebox.NO,
                            icon="warning"):
                        return False
                    self._set_roi_mask_folder("")
                    return True
                self._set_roi_mask_folder(matched)
            return True
        if not current:
            variants = self._find_variants(SUBFOLDER["vsi"])
            if self.config.get("input_type") == "vsi" and len(variants) == 1:
                current = os.path.join(
                    self.config.get("project_output_folder", "").strip(),
                    variants[0], "DONOTDELETE-MASK_FOR_VSI")
                self._set_roi_mask_folder(current)
                is_auto = True
            elif confirm_unknown and variants:
                return messagebox.askyesno(
                    "No ROI masks selected",
                    f"The older run '{segmentation_run}' does not record "
                    "whether it came from a slide scanner. No ROI mask "
                    "folder is selected.\n\n"
                    "Continue with whole-image analysis? Yes continues "
                    "without ROI-specific counts. No cancels so you can "
                    "choose the correct masks in Step 5.",
                    parent=self.root, default=messagebox.NO, icon="warning",
                )
        if confirm_unknown and is_auto:
            if not _has_converter_roi_masks(current):
                if not messagebox.askyesno(
                        "ROI masks are missing",
                        f"No ROI masks were found for the older run "
                        f"'{segmentation_run}' in:\n{current}\n\n"
                        "Continue with whole-image analysis? Yes continues "
                        "without ROI-specific counts. No cancels so you can "
                        "choose or restore the masks in Step 5.",
                        parent=self.root, default=messagebox.NO,
                        icon="warning"):
                    return False
                self._set_roi_mask_folder("")
                return True
            return messagebox.askyesno(
                "Confirm ROI masks for this run",
                f"The older segmentation run '{segmentation_run}' does not "
                "record which converter output made it.\n\n"
                f"Use ROI masks from:\n{current}\n\n"
                "Yes continues with this folder. No cancels so you can "
                "choose the correct folder in Step 5.",
                parent=self.root, default=messagebox.NO, icon="warning",
            )
        return True

    def _roi_help_for_run(self, segmentation_run: str) -> str:
        if not segmentation_run:
            return "Choose a segmentation run first; ROI masks are optional."
        kind, converter_run = self._segmentation_source(segmentation_run)
        if kind == "converted":
            return (f"This run came from {converter_run}. Its ROI masks are "
                    "selected automatically. Check the folder below before analysis.")
        if kind == "direct":
            return ("Direct-read run: ROI masks are optional. Leave this blank "
                    "for whole-image analysis, or choose matching .tif/.roi files.")
        return ("Earlier run: its source was not recorded. Check that the ROI "
                "mask folder below matches this run before analysis.")

    def _analysis_generated_files(self, segmentation_run: str,
                                  output_dir: str) -> list[str]:
        """List only established analysis files for the selected run."""
        files = []
        for filename in (
            "combined_analysis_summary.xlsx", "combined_analysis_summary.csv",
            "analysis_run_summary.json", "analysis_errors.log",
        ):
            path = os.path.join(output_dir, filename)
            if os.path.isfile(path):
                files.append(path)

        proj = self.config.get("project_output_folder", "").strip()
        seg_root = os.path.join(proj, segmentation_run)
        configured = str(self.config.get("ana_subfolder_name") or
                         "analysis_results").strip()
        subfolders = {configured, "analysis_results", "colocalization_analysis"}
        if not os.path.isdir(seg_root):
            return files
        for sample in os.listdir(seg_root):
            if sample.startswith((".", "DEBUG")):
                continue
            sample_dir = os.path.join(seg_root, sample)
            if not os.path.isdir(sample_dir):
                continue
            output_pattern = re.compile(
                rf"^{re.escape(sample)}_(?:summary\.csv|"
                r"(?:channel_\d+|cell_of_interest|protein_marker)_object_table\.csv|"
                r"channel_\d+_to_channel_\d+_(?:matches|iou_candidates)\.csv|"
                r"roi_mask_resized\.tif|QC_(?:overlay|roi_preview)\.(?:tif|png))$"
            )
            for subfolder in subfolders:
                results_dir = os.path.join(sample_dir, subfolder)
                if not os.path.isdir(results_dir):
                    continue
                for filename in os.listdir(results_dir):
                    path = os.path.join(results_dir, filename)
                    if output_pattern.fullmatch(filename) and os.path.isfile(path):
                        files.append(path)
        return sorted(set(files))

    @staticmethod
    def _write_analysis_backup_manifest(backup_dir: str, manifest: dict) -> None:
        """Persist recovery instructions before moving any result file."""
        path = os.path.join(backup_dir, "backup_manifest.json")
        pending = path + ".tmp"
        with open(pending, "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending, path)

    def _set_analysis_backup_state(self, backup_dir: str, state: str) -> None:
        with open(os.path.join(backup_dir, "backup_manifest.json")) as f:
            manifest = json.load(f)
        manifest["state"] = state
        self._write_analysis_backup_manifest(backup_dir, manifest)

    def _recover_analysis_backups(self, segmentation_run: str) -> int:
        """Finish or roll back an interrupted replacement for this run."""
        proj = os.path.abspath(
            self.config.get("project_output_folder", "").strip())
        backup_root = os.path.join(proj, ".analysis_backups")
        if not os.path.isdir(backup_root):
            return 0
        if os.path.commonpath((
                os.path.realpath(proj), os.path.realpath(backup_root))) != os.path.realpath(proj):
            raise ValueError(f"Backup folder points outside the project: {backup_root}")
        recovered = 0
        for name in sorted(os.listdir(backup_root)):
            backup_dir = os.path.join(backup_root, name)
            manifest_path = os.path.join(backup_dir, "backup_manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("segmentation_run") != segmentation_run:
                continue
            if manifest.get("state") not in ("preparing", "ready", "restoring"):
                continue
            output_dir = self._analysis_output_dir(segmentation_run)
            if (manifest["state"] == "ready" and
                    self._analysis_workbook_written(output_dir)):
                self._set_analysis_backup_state(backup_dir, "committed")
                self._log(f"Finished recording the earlier analysis run: {backup_dir}")
            else:
                self._restore_analysis_backup(backup_dir)
                self._log(f"Recovered interrupted analysis results from: {backup_dir}")
            recovered += 1
        return recovered

    def _recover_all_analysis_backups(self) -> int:
        """Recover unfinished replacements when an existing project is opened."""
        proj = self.config.get("project_output_folder", "").strip()
        backup_root = os.path.join(proj, ".analysis_backups")
        if not os.path.isdir(backup_root):
            return 0
        if os.path.commonpath((
                os.path.realpath(proj), os.path.realpath(backup_root))) != os.path.realpath(proj):
            raise ValueError(f"Backup folder points outside the project: {backup_root}")
        runs = set()
        for name in os.listdir(backup_root):
            manifest_path = os.path.join(backup_root, name,
                                         "backup_manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("state") in ("preparing", "ready", "restoring"):
                runs.add(manifest["segmentation_run"])
        return sum(self._recover_analysis_backups(run) for run in sorted(runs))

    def _prepare_analysis_output(self, segmentation_run: str
                                 ) -> tuple[str, str | None] | None:
        """Confirm replacement once, then save only this run's old results."""
        output_dir = self._analysis_output_dir(segmentation_run)
        try:
            self._recover_analysis_backups(segmentation_run)
        except Exception as exc:
            messagebox.showerror(
                "Could not recover previous analysis",
                f"The previous analysis backup needs attention:\n{exc}\n"
                "Analysis was not started.", parent=self.root,
            )
            return None
        existing = self._analysis_generated_files(segmentation_run, output_dir)
        if not existing:
            return output_dir, None

        replace = messagebox.askyesno(
            "Replace analysis results?",
            f"Segmentation run: {segmentation_run}\n"
            f"Analysis folder: {os.path.basename(output_dir)}\n\n"
            "Results already exist. Replace this run's Excel summary and "
            "per-sample analysis files?\n"
            "Yes keeps a backup of the previous results. No cancels.\n"
            "Other runs will not be changed.",
            parent=self.root, default=messagebox.NO, icon="warning",
        )
        if not replace:
            self._log("Analysis canceled; existing results were kept.")
            if hasattr(self, "_ana_status") and self._ana_status.winfo_exists():
                self._ana_status.config(
                    text="Analysis canceled; existing results were kept.", fg=MUTED)
            return None

        proj = os.path.abspath(
            self.config.get("project_output_folder", "").strip())
        proj_real = os.path.realpath(proj)
        backup_dir = os.path.join(
            proj, ".analysis_backups",
            f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}",
        )
        moved = []
        try:
            originals = [os.path.abspath(path) for path in existing]
            for original in originals:
                if os.path.commonpath((proj_real, os.path.realpath(original))) != proj_real:
                    raise ValueError(
                        f"Analysis output is outside the project: {original}")
            if os.path.commonpath((
                    proj_real, os.path.realpath(os.path.dirname(backup_dir)))) != proj_real:
                raise ValueError("Analysis backup folder points outside the project")
            os.makedirs(backup_dir, exist_ok=False)
            manifest = {
                "state": "preparing",
                "segmentation_run": segmentation_run,
                "analysis_output": os.path.relpath(output_dir, proj),
                "original_files": [os.path.relpath(p, proj) for p in originals],
            }
            self._write_analysis_backup_manifest(backup_dir, manifest)
            for original in originals:
                relative = os.path.relpath(original, proj)
                saved = os.path.join(backup_dir, relative)
                os.makedirs(os.path.dirname(saved), exist_ok=True)
                os.replace(original, saved)
                moved.append((original, saved))
            manifest["state"] = "ready"
            self._write_analysis_backup_manifest(backup_dir, manifest)
        except Exception as exc:
            for original, saved in reversed(moved):
                try:
                    os.makedirs(os.path.dirname(original), exist_ok=True)
                    os.replace(saved, original)
                except OSError:
                    pass
            messagebox.showerror(
                "Could not prepare analysis",
                f"The previous results could not be backed up.\n{exc}\n"
                f"Backup folder: {backup_dir}\n"
                "Analysis was not started.", parent=self.root,
            )
            if hasattr(self, "_ana_status") and self._ana_status.winfo_exists():
                self._ana_status.config(
                    text="Could not back up previous results.", fg=RED)
            return None
        return output_dir, backup_dir

    def _restore_analysis_backup(self, backup_dir: str) -> str:
        """Restore previous results if the replacement run fails fatally."""
        proj = os.path.abspath(
            self.config.get("project_output_folder", "").strip())
        proj_real = os.path.realpath(proj)
        with open(os.path.join(backup_dir, "backup_manifest.json")) as f:
            manifest = json.load(f)
        state = manifest.get("state")
        if state not in ("preparing", "ready", "restoring"):
            raise ValueError(f"Backup cannot be restored in state {state}: {backup_dir}")
        segmentation_run = manifest["segmentation_run"]
        output_dir = self._analysis_output_dir(segmentation_run)
        if state != "restoring":
            manifest["state"] = "restoring"
            self._write_analysis_backup_manifest(backup_dir, manifest)
        original_paths = {
            os.path.join(proj, relative) for relative in manifest["original_files"]
        }
        for generated in self._analysis_generated_files(segmentation_run, output_dir):
            if os.path.commonpath((proj_real, os.path.realpath(generated))) != proj_real:
                raise ValueError(f"Analysis output is outside the project: {generated}")
            if generated not in original_paths:
                os.remove(generated)
        for relative in manifest["original_files"]:
            original = os.path.join(proj, relative)
            saved = os.path.join(backup_dir, relative)
            if os.path.commonpath((
                    proj_real, os.path.realpath(os.path.dirname(original)))) != proj_real:
                raise ValueError(f"Cannot restore outside the project: {original}")
            if os.path.isfile(saved):
                os.makedirs(os.path.dirname(original), exist_ok=True)
                os.replace(saved, original)
            elif not os.path.isfile(original):
                raise FileNotFoundError(
                    f"Neither original nor backup is available: {original}")
        manifest["state"] = "restored"
        self._write_analysis_backup_manifest(backup_dir, manifest)
        return backup_dir

    @staticmethod
    def _analysis_workbook_written(output_dir: str) -> bool:
        """A zero exit alone does not mean the new workbook was saved."""
        try:
            with open(os.path.join(output_dir, "analysis_run_summary.json")) as f:
                written = bool(json.load(f)["workbook_written"])
            return written and os.path.isfile(os.path.join(
                output_dir, "combined_analysis_summary.xlsx"))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        # Guard against the log widget having been destroyed by page navigation
        # while a background thread is still streaming output.
        try:
            if hasattr(self, "_log_box") and self._log_box.winfo_exists():
                self._log_box.insert("end", msg + "\n")
                self._log_box.see("end")
        except tk.TclError:
            pass

    def _vsi_log_append(self, msg: str) -> None:
        """Append a line to the VSI converter output log box."""
        try:
            if not (hasattr(self, "_vsi_log") and self._vsi_log.winfo_exists()):
                return
            self._vsi_log.config(state="normal")
            self._vsi_log.insert("end", msg + "\n")
            self._vsi_log.see("end")
            self._vsi_log.config(state="disabled")
        except tk.TclError:
            pass

    def _set_run_status(self, msg: str, color: str = MUTED) -> None:
        try:
            if hasattr(self, "_run_status") and self._run_status.winfo_exists():
                self._run_status.config(text=msg, fg=color)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Script path helper
    # ------------------------------------------------------------------

    def _script(self, name: str) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, name)

    # ------------------------------------------------------------------
    # Subprocess runner (streams stdout to log)
    # ------------------------------------------------------------------

    def _run_subprocess_in_log(self, script: str, label: str,
                                on_done=None, success_check=None) -> None:
        self._log(f"\n{'─'*60}")
        self._log(f"▶  {label}")
        self._log(f"{'─'*60}")
        self._set_run_status(f"{label} running…", ACCENT)

        def _thread():
            try:
                proc = subprocess.Popen(
                    [sys.executable, script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._track_proc(proc, label)
                for line in proc.stdout:
                    self.root.after(0, self._log, line.rstrip())
                proc.wait()
                process_ok = proc.returncode == 0
                ok = process_ok
                if process_ok and success_check is not None:
                    ok = bool(success_check())
                if ok:
                    msg = f"\n✓  {label} complete."
                elif process_ok:
                    msg = f"\n✗  {label} did not save the Excel summary."
                else:
                    msg = f"\n✗  {label} failed (code {proc.returncode})."
                self.root.after(0, self._log, msg)
                self.root.after(
                    0, self._set_run_status,
                    f"{label} done." if ok else f"{label} failed.",
                    GREEN if ok else RED,
                )
                if on_done:
                    self.root.after(0, on_done, ok)
            except Exception as exc:
                self.root.after(0, self._log, f"\n[ERROR] {exc}")
                self.root.after(0, self._set_run_status, f"{label} error.", RED)
                if on_done:
                    self.root.after(0, on_done, False)

        threading.Thread(target=_thread, daemon=True).start()

    # ==================================================================
    # STEP 1 — Create Project
    # ==================================================================

    def _build_project_step(self, parent) -> None:
        self._section_header(
            parent,
            "Step 1 — Create Project",
            "Start a new project or reload an existing one.",
        )

        # ── Mode toggle ───────────────────────────────────────────────
        mode_card = self._card(parent)
        mode_row = tk.Frame(mode_card, bg=CARD)
        mode_row.pack(anchor="w", pady=(0, 16))

        self._btn_new = tk.Button(
            mode_row, text="  ✦  New Project  ",
            command=lambda: self._set_proj_mode("new"),
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat", padx=16, pady=9,
        )
        self._btn_new.pack(side="left", padx=(0, 6))

        self._btn_open = tk.Button(
            mode_row, text="  📂  Open Existing Project  ",
            command=lambda: self._set_proj_mode("open"),
            bg=ACTIVE, fg=MUTED, font=FONT_BOLD, relief="flat", padx=16, pady=9,
        )
        self._btn_open.pack(side="left")

        # ── New project pane ──────────────────────────────────────────
        self._pane_new = tk.Frame(mode_card, bg=CARD)

        g = tk.Frame(self._pane_new, bg=CARD)
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=160)
        g.grid_columnconfigure(1, weight=1)
        tk.Label(g, text="Project name:", bg=CARD, fg=FG, font=FONT).grid(
            row=0, column=0, sticky="w", pady=6)
        tk.Entry(g, textvariable=self._vars["project_name"], bg=BG, fg=FG,
                 insertbackground=FG, relief="flat", font=FONT, width=44).grid(
            row=0, column=1, sticky="w", pady=6, padx=8)

        self._path_row(self._pane_new,
                       "Project output folder  (all results go into subfolders here):",
                       "project_output_folder")

        tk.Label(
            self._pane_new,
            text="CellMatchR creates:  VSI_Converted_TIFF/  •  Segmentation/  •  Validation/  •  Analysis/",
            bg=CARD, fg=MUTED, font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(6, 0))

        self._proj_status = tk.Label(self._pane_new, text="", bg=CARD, fg=MUTED, font=FONT)
        self._proj_status.pack(anchor="w", pady=(10, 4))

        tk.Button(
            self._pane_new, text="✔  Create Project Folder",
            command=self._create_project,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat", padx=18, pady=10,
        ).pack(anchor="w")

        # ── Open existing pane ────────────────────────────────────────
        self._pane_open = tk.Frame(mode_card, bg=CARD)

        tk.Label(self._pane_open,
                 text="Select the  cellmatchr_project.json  file saved inside a previous project folder.",
                 bg=CARD, fg=MUTED, font=FONT, wraplength=660).pack(anchor="w", pady=(0, 10))

        # File-picker row for the JSON file
        _json_row = tk.Frame(self._pane_open, bg=CARD)
        _json_row.pack(fill="x", pady=(0, 4))
        tk.Label(_json_row, text="Project file (.json):", bg=CARD, fg=FG,
                 font=FONT, width=26, anchor="w").pack(side="left")
        tk.Entry(_json_row, textvariable=self._open_json_var,
                 bg="#3c3c4e", fg=FG, insertbackground=FG,
                 relief="flat", font=FONT).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(_json_row, text="Browse…",
                  command=self._browse_project_json,
                  bg=ACTIVE, fg=FG, font=FONT, relief="flat", padx=10).pack(side="left")

        self._open_status = tk.Label(self._pane_open, text="", bg=CARD, fg=MUTED, font=FONT)
        self._open_status.pack(anchor="w", pady=(10, 4))

        tk.Button(
            self._pane_open, text="📂  Load Project Settings",
            command=self._open_project,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat", padx=18, pady=10,
        ).pack(anchor="w")

        # Show new-project pane by default
        self._pane_new.pack(fill="x")

    def _set_proj_mode(self, mode: str) -> None:
        if mode == "new":
            self._pane_open.pack_forget()
            self._pane_new.pack(fill="x")
            self._btn_new.config(bg=ACCENT, fg="#1e1e2e")
            self._btn_open.config(bg=ACTIVE, fg=MUTED)
        else:
            self._pane_new.pack_forget()
            self._pane_open.pack(fill="x")
            self._btn_new.config(bg=ACTIVE, fg=MUTED)
            self._btn_open.config(bg=ACCENT, fg="#1e1e2e")

    def _create_project(self) -> None:
        self._sync_vars_to_config()
        folder = self.config.get("project_output_folder", "").strip()
        if not folder:
            self._proj_status.config(text="⚠  Set a project output folder first.", fg=RED)
            return
        try:
            os.makedirs(folder, exist_ok=True)
            save_config(self.config)
            save_project_config(self.config)
            self._update_project_display()
            self._proj_status.config(text=f"✔  Project folder ready: {folder}", fg=GREEN)
        except Exception as exc:
            self._proj_status.config(text=f"✗  Could not create folder: {exc}", fg=RED)

    def _browse_project_json(self) -> None:
        self.root.lift()
        self.root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select cellmatchr_project.json",
            filetypes=[("JSON files", "*.json")],
        )
        self.root.attributes("-topmost", False)
        if path:
            self._open_json_var.set(path)

    def _open_project(self) -> None:
        json_path = self._open_json_var.get().strip()
        if not json_path or not os.path.isfile(json_path):
            self._open_status.config(
                text="⚠  Select a cellmatchr_project.json file first.", fg=RED)
            return
        folder = os.path.dirname(os.path.abspath(json_path))
        cfg = load_project_config(folder)
        if cfg is None:
            self._open_status.config(
                text=f"✗  Could not read project file:\n   {json_path}",
                fg=RED)
            return
        # Load config into memory and re-sync all vars
        self.config = cfg
        # Always repoint the project to where the .json actually lives — the
        # stored project_output_folder may be a stale path from another machine
        # or a moved/renamed folder. Using the json's own directory guarantees
        # the segmentation/validation/analysis folders are found on disk.
        self.config["project_output_folder"] = folder
        self._val_input_var = None
        self._ana_input_var = None
        recovered = 0
        recovery_error = ""
        try:
            recovered = self._recover_all_analysis_backups()
        except Exception as exc:
            recovery_error = str(exc)
        # A stored seg-folder selection from another machine may not exist
        # here; clear it so the dropdowns default to a folder actually on disk.
        saved_seg = self.config.get("seg_input_subfolder", "").strip()
        if saved_seg and not os.path.isdir(os.path.join(folder, saved_seg)):
            self.config["seg_input_subfolder"] = ""
        # Repair a stale input folder (e.g. the converter output folder was
        # renamed after saving) so the channel preview and segmentation still
        # find the images.
        if self._folder_image_count(self.config.get("input_root_folder", "").strip()) == 0:
            healed = self._resolve_existing_input_folder()
            if self._folder_image_count(healed) > 0:
                self.config["input_root_folder"] = healed
        for k, v in self._vars.items():
            val = self.config.get(k)
            if val is not None:
                try:
                    v.set(val)
                except Exception:
                    pass
        save_config(self.config)
        self._update_project_display()
        name = self.config.get("project_name") or os.path.basename(folder)
        if recovery_error:
            self._open_status.config(
                text=f"⚠  Loaded project: {name}. Analysis backup needs attention.",
                fg=YELLOW)
            messagebox.showwarning(
                "Analysis backup needs attention",
                f"The project opened, but an interrupted analysis could not "
                f"be recovered automatically:\n{recovery_error}\n\n"
                f"Backups: {os.path.join(folder, '.analysis_backups')}",
                parent=self.root,
            )
        elif recovered:
            self._open_status.config(
                text=f"✔  Loaded project: {name}. Recovered {recovered} "
                     "interrupted analysis run(s).", fg=GREEN)
        else:
            self._open_status.config(text=f"✔  Loaded project: {name}", fg=GREEN)

    # ==================================================================
    # STEP 2 — Input & Paths
    # ==================================================================

    def _build_input_step(self, parent) -> None:
        self._section_header(
            parent,
            "Step 2 — Input & Paths",
            "Select your data source and (if needed) convert VSI files to TIF.",
        )

        # ── Input type radio ─────────────────────────────────────────
        type_card = self._card(parent, "Input Data Format")
        tk.Label(
            type_card, text="How was this data acquired?",
            bg=CARD, fg=FG, font=FONT_BOLD,
        ).pack(anchor="w", pady=(0, 6))
        for val, lbl in [
            ("czi", "1.  Confocal   —  read files directly, one sample per acquisition"),
            ("vsi", "2.  Slide scanner   —  draw ROIs on a large scan, export full-res TIFFs"),
        ]:
            tk.Radiobutton(
                type_card, text=lbl,
                variable=self._vars["input_type"], value=val,
                bg=CARD, fg=FG, selectcolor=CARD,
                activebackground=CARD, font=FONT_BOLD,
            ).pack(anchor="w", pady=2)

        # ── Supported file types, shown per mode ─────────────────────
        fmt_card = self._card(parent, "Supported File Types")

        self._fmt_confocal_frame = tk.Frame(fmt_card, bg=CARD)
        for ext, brand, status, note in [
            (".czi",      "Zeiss",             "tested", None),
            (".lif",      "Leica (confocal)",  "tested",
             "a slide-scanner .lif placed in this folder is detected automatically and skipped — see the file list below"),
            (".tif / .tiff", "any",            "tested", None),
            (".nd2",      "Nikon",             "beta",   None),
            (".oib / .oif", "Olympus FluoView", "beta",  None),
        ]:
            self._format_status_row(self._fmt_confocal_frame, ext, brand, status, note)
        tk.Label(
            self._fmt_confocal_frame,
            text="Not supported:  Olympus .oir (newer FV3000) — convert it to TIF in Fiji / ImageJ first.",
            bg=CARD, fg=MUTED, font=FONT, justify="left", wraplength=700,
        ).pack(anchor="w", pady=(6, 0))

        self._fmt_slide_frame = tk.Frame(fmt_card, bg=CARD)
        for ext, brand, status, note in [
            (".vsi", "Olympus slide scanner", "tested", None),
            (".lif", "Leica (slide scanner)", "tested",
             "a confocal .lif placed in this folder is detected automatically and skipped"),
        ]:
            self._format_status_row(self._fmt_slide_frame, ext, brand, status, note)

        # ── VSI section (shown when input_type == "vsi") ─────────────
        self._vsi_section = tk.Frame(parent, bg=BG)

        vsi_card = self._card(self._vsi_section, "Slide Scanner Converter  (Olympus VSI · Leica LIF)")
        tk.Label(vsi_card,
                 text="Draw one ROI per region on the slide scan and export each as a "
                      "full-resolution TIFF.\nOutput auto-saves to: Project folder / VSI_Converted_TIFF/",
                 bg=CARD, fg=MUTED, font=FONT, justify="left").pack(anchor="w", pady=(0, 6))

        # Collapsible output override
        self._vsi_override_frame = tk.Frame(vsi_card, bg=CARD)
        self._path_row(self._vsi_override_frame, "Override output folder:", "input_root_folder")
        toggle_lbl = tk.Label(vsi_card,
                              text="(Optional) Override VSI conversion output directory",
                              bg=CARD, fg=MUTED, font=(FONT[0], FONT[1], "underline"),
                              cursor="hand2")
        toggle_lbl.pack(anchor="w", pady=(0, 6))
        toggle_lbl.bind("<Button-1>", lambda _: (
            self._vsi_override_frame.pack_forget()
            if self._vsi_override_frame.winfo_ismapped()
            else self._vsi_override_frame.pack(fill="x", pady=(0, 6))
        ))

        btn_row = tk.Frame(vsi_card, bg=CARD)
        btn_row.pack(anchor="w", pady=(2, 0))
        tk.Button(btn_row, text="Launch Slide Scanner Converter →",
                  command=self._launch_vsi,
                  bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
                  padx=14, pady=7).pack(side="left")
        self._vsi_spin = tk.Label(btn_row, text="", bg=CARD, fg=ACCENT,
                                  font=("Segoe UI", 13), width=2)
        self._vsi_spin.pack(side="left", padx=6)
        self._vsi_warn = tk.Label(btn_row, text="", bg=CARD, fg=RED, font=FONT)
        self._vsi_warn.pack(side="left", padx=4)

        # Output log — hidden until the converter runs; shows full output on error
        self._vsi_log_frame = tk.Frame(vsi_card, bg=CARD)
        tk.Label(self._vsi_log_frame, text="Output log:", bg=CARD, fg=MUTED,
                 font=FONT).pack(anchor="w", pady=(8, 2))
        self._vsi_log = scrolledtext.ScrolledText(
            self._vsi_log_frame, height=8, bg=BG, fg=GREEN,
            font=FONT_MONO, relief="flat", wrap="word", state="disabled",
        )
        self._vsi_log.pack(fill="x")
        # (frame stays hidden until first run)

        # Auto-detected input folder display (set after converter closes)
        auto_card = self._card(self._vsi_section, "Input Folder  (auto-detected from converter)")
        self._vsi_auto_label = tk.Label(
            auto_card,
            text=self._vsi_auto_text(),
            bg=CARD, fg=FG if self.config.get("input_root_folder") else MUTED,
            font=FONT, wraplength=700, anchor="w",
        )
        self._vsi_auto_label.pack(anchor="w", pady=(0, 6))

        # Optional override of auto-detected folder
        self._vsi_input_override_frame = tk.Frame(auto_card, bg=CARD)
        self._path_row(self._vsi_input_override_frame,
                       "Override input folder:", "input_root_folder")
        ovr2_lbl = tk.Label(auto_card,
                            text="(Optional) Use a different TIF input folder",
                            bg=CARD, fg=MUTED, font=(FONT[0], FONT[1], "underline"),
                            cursor="hand2")
        ovr2_lbl.pack(anchor="w")
        ovr2_lbl.bind("<Button-1>", lambda _: (
            self._vsi_input_override_frame.pack_forget()
            if self._vsi_input_override_frame.winfo_ismapped()
            else self._vsi_input_override_frame.pack(fill="x", pady=(4, 0))
        ))

        # ── CZI/LIF/ND2/OIB-OIF/TIF section (shown when input_type == "czi") ─
        self._czi_section = tk.Frame(parent, bg=BG)
        czi_card = self._card(self._czi_section, "CZI / LIF / ND2 / OIB-OIF / TIF Input Folder")
        tk.Label(czi_card,
                 text="Select the root folder. Each subfolder should contain one "
                      "CZI, LIF, ND2, OIB/OIF, or TIF file.",
                 bg=CARD, fg=MUTED, font=FONT).pack(anchor="w", pady=(0, 8))
        self._path_row(czi_card,
                       "Input root folder:",
                       "input_root_folder")

        confirm_row = tk.Frame(czi_card, bg=CARD)
        confirm_row.pack(anchor="w", pady=(10, 0))
        tk.Button(confirm_row, text="✔  Confirm Input Folder",
                  command=self._confirm_input_folder,
                  bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
                  padx=14, pady=7).pack(side="left")
        self._czi_confirm_lbl = tk.Label(confirm_row, text="", bg=CARD, fg=GREEN, font=FONT)
        self._czi_confirm_lbl.pack(side="left", padx=8)

        # Live preview: lists files in the folder, flags any .lif that will
        # be skipped because it's actually slide-scanner data (or the
        # reverse, in the Slide scanner mode's own picker).
        self._input_preview_frame = tk.Frame(czi_card, bg=CARD)
        self._input_preview_frame.pack(fill="x", pady=(4, 0))

        # ── Wire radio toggle ─────────────────────────────────────────
        self._vars["input_type"].trace_add("write", lambda *_: self._on_input_type_changed())
        self._vars["input_root_folder"].trace_add("write", lambda *_: self._refresh_input_file_preview())
        self._vars["seg_skip_subdirs"].trace_add("write", lambda *_: self._refresh_input_file_preview())
        self._on_input_type_changed()
        self._refresh_input_file_preview()

    def _vsi_auto_text(self) -> str:
        folder = self.config.get("input_root_folder", "").strip()
        if folder:
            return f"TIF exports will be read from:  {folder}"
        return "TIF input folder will be set automatically after the converter closes."

    def _on_input_type_changed(self) -> None:
        # Guard: frames only exist while Step 2 is displayed
        try:
            t = self._vars["input_type"].get()
            if t == "vsi":
                self._czi_section.pack_forget()
                self._vsi_section.pack(fill="x")
                self._fmt_confocal_frame.pack_forget()
                self._fmt_slide_frame.pack(fill="x")
            else:
                self._vsi_section.pack_forget()
                self._czi_section.pack(fill="x")
                self._fmt_slide_frame.pack_forget()
                self._fmt_confocal_frame.pack(fill="x")
                self._refresh_input_file_preview()
        except (tk.TclError, AttributeError):
            pass

    def _confirm_input_folder(self) -> None:
        self._sync_vars_to_config()
        folder = self.config.get("input_root_folder", "").strip()
        if not folder or not os.path.isdir(folder):
            self._czi_confirm_lbl.config(text="⚠  Folder not found.", fg=RED)
            return
        save_config(self.config)
        save_project_config(self.config)
        self._czi_confirm_lbl.config(text=f"✔  Set: {os.path.basename(folder)}", fg=GREEN)
        self._refresh_input_file_preview()

    # ==================================================================
    # STEP 3 — Segmentation
    # ==================================================================

    def _build_segmentation_step(self, parent) -> None:
        self._section_header(
            parent,
            "Step 3 — Segmentation",
            "Configure and run MicroSAM tiled automatic segmentation.",
        )

        # ── Input folder status ──────────────────────────────────────
        input_card = self._card(parent, "Input Folder")

        # Converter outputs available to segment (VSI_Converted_TIFF[_*]).
        self._seg_input_variants = self._find_input_variants()
        _path_by_name = {n: p for n, p in self._seg_input_variants}

        folder = self.config.get("input_root_folder", "").strip()
        # If the stored folder has no images (e.g. it was renamed), default to a
        # converter output that does, so the preview and run work out of the box.
        if self._folder_image_count(folder) == 0 and self._seg_input_variants:
            folder = next((p for _, p in self._seg_input_variants
                           if self._direct_image_count(p) > 0),
                          self._seg_input_variants[0][1])
            self.config["input_root_folder"] = folder
        input_ok = bool(folder and os.path.isdir(folder))

        self._seg_input_warn = tk.Label(
            input_card,
            text="" if input_ok else "⚠  No input folder set — go to Step 2 to set your data source.",
            bg=CARD, fg=RED, font=FONT, wraplength=700,
        )
        self._seg_input_warn.pack(anchor="w")

        self._seg_input_display = tk.Label(
            input_card,
            text=folder if input_ok else "—",
            bg=CARD, fg=FG if input_ok else MUTED, font=FONT_MONO, wraplength=700,
        )
        self._seg_input_display.pack(anchor="w", pady=(2, 0))

        # Dropdown to choose among multiple converter outputs (like Val/Analysis)
        if len(self._seg_input_variants) > 1:
            _names = [n for n, _ in self._seg_input_variants]
            _cur = next((n for n, p in self._seg_input_variants
                         if os.path.normpath(p) == os.path.normpath(folder)), _names[0])
            self._seg_src_var = tk.StringVar(value=_cur)

            ttk.Separator(input_card, orient="horizontal").pack(fill="x", pady=(8, 6))
            tk.Label(input_card, text="Converted output to use:", bg=CARD, fg=MUTED,
                     font=FONT).pack(anchor="w")
            self._seg_src_cur_lbl = tk.Label(input_card, text=f"Using:  {_cur}",
                                             bg=CARD, fg=FG, font=FONT_MONO, wraplength=660)
            self._seg_src_cur_lbl.pack(anchor="w", pady=(0, 4))

            _ovr_frame = tk.Frame(input_card, bg=CARD)
            _ovr_lbl = tk.Label(input_card, text="▶  Use a different converted output",
                                bg=CARD, fg=MUTED, font=(FONT[0], FONT[1], "underline"),
                                cursor="hand2")
            _ovr_lbl.pack(anchor="w")
            _cb_src = ttk.Combobox(_ovr_frame, textvariable=self._seg_src_var,
                                   values=_names, state="readonly", width=46)
            _cb_src.pack(side="left", pady=(4, 0))
            self._bind_combobox_scroll(_cb_src)

            def _seg_src_changed(*_):
                nm = self._seg_src_var.get()
                p = _path_by_name.get(nm, "")
                self.config["input_root_folder"] = p
                # Also update the bound StringVar -- input_root_folder is one
                # of self._vars, and _sync_vars_to_config() (called on every
                # step change and before every "Run") would otherwise
                # silently overwrite self.config with this var's stale value.
                self._vars["input_root_folder"].set(p)
                self._align_converter_roi_folder(p)
                self._seg_src_cur_lbl.config(text=f"Using:  {nm}")
                has = self._folder_image_count(p) > 0
                self._seg_input_display.config(text=p if p else "—", fg=FG if has else MUTED)
                self._seg_input_warn.config(
                    text="" if has else f"⚠  No images found in:  {p}")
                # reset the channel preview for the newly chosen source
                self._ch_images = []
                self._ch_photo_refs = []
                self._ch_index = 0
                try:
                    self._ch_preview_img.config(image="")
                    self._ch_lbl.config(
                        text="Press 'Load Preview' to load channels." if has
                        else "No images in this output",
                        fg=ACCENT if has else MUTED)
                    self._ch_load_btn.config(state="normal" if (has and _PIL) else "disabled")
                    self._ch_prev_btn.config(state="disabled")
                    self._ch_next_btn.config(state="disabled")
                except (tk.TclError, AttributeError):
                    pass
            self._seg_src_var.trace_add("write", _seg_src_changed)

            def _toggle_seg_src(lbl=_ovr_lbl):
                if _ovr_frame.winfo_ismapped():
                    _ovr_frame.pack_forget()
                    lbl.config(text="▶  Use a different converted output")
                else:
                    _ovr_frame.pack(fill="x", pady=(0, 4))
                    lbl.config(text="▼  Use a different converted output")
            _ovr_lbl.bind("<Button-1>", lambda _: _toggle_seg_src())

        # ── Channel preview + picker ─────────────────────────────────
        ch_card = self._card(parent, "Channel Preview  (click to toggle prediction)")
        self._ch_images: list = []
        self._ch_photo_refs: list = []
        self._ch_index: int = 0
        self._ch_selected: set = set()

        nav_row = tk.Frame(ch_card, bg=CARD)
        nav_row.pack(anchor="w", pady=(0, 6))

        self._ch_prev_btn = tk.Button(
            nav_row, text="◀", command=lambda: self._ch_navigate(-1),
            bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat", padx=8, pady=4,
            state="disabled",
        )
        self._ch_prev_btn.pack(side="left")

        self._ch_lbl = tk.Label(nav_row, text="No preview", bg=CARD, fg=MUTED, font=FONT, width=28)
        self._ch_lbl.pack(side="left", padx=10)

        self._ch_next_btn = tk.Button(
            nav_row, text="▶", command=lambda: self._ch_navigate(1),
            bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat", padx=8, pady=4,
            state="disabled",
        )
        self._ch_next_btn.pack(side="left")

        self._ch_load_btn = tk.Button(
            nav_row, text="⟳  Load Preview",
            command=self._trigger_ch_preview_load,
            bg=ACTIVE, fg=ACCENT, font=FONT_BOLD, relief="flat", padx=10, pady=4,
            activeforeground=FG,
            state="normal" if (input_ok and _PIL) else "disabled",
        )
        self._ch_load_btn.pack(side="left", padx=(12, 0))

        self._ch_preview_img = tk.Label(ch_card, bg=CARD, cursor="hand2")
        self._ch_preview_img.pack(anchor="w", pady=(0, 4))
        self._ch_preview_img.bind("<Button-1>", lambda _: self._ch_toggle_current())

        self._ch_load_progress = ttk.Progressbar(ch_card, mode="indeterminate", length=380)
        # Hidden by default — shown only while loading

        self._ch_field_lbl = tk.Label(
            ch_card,
            text="Channels to predict (click image above, or type manually):",
            bg=CARD, fg=MUTED, font=FONT,
        )
        self._ch_field_lbl.pack(anchor="w")

        if not _PIL:
            tk.Label(ch_card, text="Install Pillow (pip install Pillow) to enable channel preview.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w", pady=4)

        # ── Channels field ───────────────────────────────────────────
        model_card = self._card(parent, "Segmentation Settings")

        # ── Backend selector (Micro-SAM | Cellpose-SAM) ──────────────
        _BK_DISPLAY = {"microsam": "Micro-SAM", "cellpose": "Cellpose-SAM"}
        _BK_REVERSE = {v: k for k, v in _BK_DISPLAY.items()}
        _bk_frame = tk.Frame(model_card, bg=CARD)
        _bk_frame.pack(fill="x", pady=(0, 8))
        tk.Label(_bk_frame, text="Segmentation backend:", bg=CARD, fg=FG,
                 font=FONT, width=24, anchor="w").pack(side="left")
        self._seg_backend_disp = tk.StringVar(
            value=_BK_DISPLAY.get(self.config.get("seg_backend", "microsam"), "Micro-SAM"))
        _cb_bk = ttk.Combobox(_bk_frame, textvariable=self._seg_backend_disp,
                              values=list(_BK_DISPLAY.values()), state="readonly", width=18)
        _cb_bk.pack(side="left", padx=8)
        self._bind_combobox_scroll(_cb_bk)

        g = tk.Frame(model_card, bg=CARD)
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=260)
        g.grid_columnconfigure(1, weight=1)

        # Channel row with red asterisk
        ch_lbl_frame = tk.Frame(g, bg=CARD)
        ch_lbl_frame.grid(row=0, column=0, sticky="w", pady=3)
        tk.Label(ch_lbl_frame, text="Channels to predict:", bg=CARD, fg=FG, font=FONT).pack(side="left")
        tk.Label(ch_lbl_frame, text=" *", bg=CARD, fg=RED, font=FONT_BOLD).pack(side="left")

        ch_entry_frame = tk.Frame(g, bg=CARD)
        ch_entry_frame.grid(row=0, column=1, sticky="w", pady=3, padx=8)
        tk.Entry(ch_entry_frame, textvariable=self._vars["seg_channels"],
                 bg=BG, fg=FG, insertbackground=FG, relief="flat", font=FONT, width=16).pack(side="left")

        tk.Label(g, text="0-based IDs, comma-separated  (e.g.  0  or  0,2)",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))

        tk.Label(g, text="Model type:", bg=CARD, fg=FG, font=FONT).grid(
            row=2, column=0, sticky="w", pady=3)
        _cb_model = ttk.Combobox(
            g, textvariable=self._vars["seg_model_type"],
            values=["vit_b_lm", "vit_l_lm", "vit_h_lm", "vit_b", "vit_l", "vit_h"],
            state="readonly", width=18,
        )
        _cb_model.grid(row=2, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_model)

        tk.Label(g, text="Preprocessing mode:", bg=CARD, fg=FG, font=FONT).grid(
            row=3, column=0, sticky="w", pady=3)
        _cb_prep = ttk.Combobox(
            g, textvariable=self._vars["seg_preprocess_mode"],
            values=["local_bg_subtract", "local_bg_subtract_mild",
                    "raw_percentile", "tile_clahe_mild", "tile_clahe"],
            state="readonly", width=24,
        )
        _cb_prep.grid(row=3, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_prep)

        # ── Cellpose-SAM parameters (shown only when that backend is chosen) ──
        self._cellpose_frame = tk.Frame(model_card, bg=CARD)
        tk.Label(self._cellpose_frame,
                 text="Cellpose-SAM parameters  (defaults work for most images — tune on a test image if needed):",
                 bg=CARD, fg=FG, font=FONT_BOLD, wraplength=700, justify="left").pack(anchor="w", pady=(0, 4))
        _cg = tk.Frame(self._cellpose_frame, bg=CARD)
        _cg.pack(fill="x")
        _cg.grid_columnconfigure(0, minsize=240)
        _cg.grid_columnconfigure(1, weight=1)
        # (label, config key, short unit hint, full explanation line)
        _cp_fields = [
            ("Cell diameter (px)", "seg_cellpose_diameter", "0 = auto",
             "Expected cell size in pixels. 0 = let the model decide (recommended — "
             "Cellpose-SAM handles scale itself). Set your cells' typical diameter only "
             "if detections look systematically too coarse or too fine."),
            ("Flow threshold", "seg_cellpose_flow_threshold", "~0.0–3.0, default 0.4",
             "Shape-quality filter: discards masks whose recomputed flow error exceeds this. "
             "Higher = keep more cells (incl. irregular ones); lower = stricter, drops "
             "poorly-formed blobs. Raise if real odd-shaped cells are missed; lower if you get junk."),
            ("Cell-probability threshold", "seg_cellpose_cellprob_threshold", "−6 to +6, default 0.0",
             "Detection sensitivity. Lower (toward −6) = more and larger masks, catches dim/small "
             "cells (more false positives); higher (toward +6) = fewer, only-confident masks "
             "(cleaner but misses faint cells). Lower if cells are missed; raise if background is segmented."),
        ]
        _r = 0
        for _lbl, _key, _unit, _expl in _cp_fields:
            _lf = tk.Frame(_cg, bg=CARD)
            _lf.grid(row=_r, column=0, sticky="w", pady=(4, 0))
            tk.Label(_lf, text=_lbl, bg=CARD, fg=FG, font=FONT).pack(side="left")
            tk.Label(_lf, text=f"  ({_unit})", bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
            tk.Entry(_cg, textvariable=self._vars[_key], bg=BG, fg=FG, insertbackground=FG,
                     relief="flat", font=FONT, width=10).grid(row=_r, column=1, sticky="w", pady=(4, 0), padx=8)
            _r += 1
            tk.Label(_cg, text=_expl, bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                     wraplength=700, justify="left").grid(row=_r, column=0, columnspan=2,
                                                          sticky="w", pady=(0, 2))
            _r += 1
        tk.Label(self._cellpose_frame,
                 text="Cellpose-SAM uses one unified model; the first run downloads ~1 GB of weights to ~/.cellpose. "
                      "The preprocessing, tiling and object-size/shape filters above still apply to its output.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), wraplength=700, justify="left").pack(anchor="w", pady=(6, 0))

        def _on_seg_backend_changed(*_):
            bk = _BK_REVERSE.get(self._seg_backend_disp.get(), "microsam")
            self._vars["seg_backend"].set(bk)
            if bk == "cellpose":
                self._cellpose_frame.pack(fill="x", pady=(6, 0))
                try: _cb_model.configure(state="disabled")
                except tk.TclError: pass
            else:
                self._cellpose_frame.pack_forget()
                try: _cb_model.configure(state="readonly")
                except tk.TclError: pass
        self._seg_backend_disp.trace_add("write", _on_seg_backend_changed)
        _on_seg_backend_changed()   # apply initial state

        filt_card = self._card(parent, "Object Filtering")
        tk.Label(
            filt_card,
            text="Set to 0 to disable a filter.  Values depend on imaging conditions —\n"
                 "open an example image and measure a typical cell to choose appropriate bounds.",
            bg=CARD, fg=MUTED, font=FONT, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        wizard_row = tk.Frame(filt_card, bg=CARD)
        wizard_row.pack(anchor="w", fill="x", pady=(0, 10))
        tk.Label(
            wizard_row,
            text="Not sure what numbers to use? Segments the first sample unfiltered, "
                 "then lets you pick example objects by eye instead:",
            bg=CARD, fg=MUTED, font=FONT, justify="left", wraplength=650,
        ).pack(anchor="w")
        _wiz_btn_row = tk.Frame(wizard_row, bg=CARD)
        _wiz_btn_row.pack(anchor="w", pady=(4, 0))
        self._wizard_btn = tk.Button(
            _wiz_btn_row, text="🧙  Parameter Wizard",
            command=self._launch_parameter_wizard,
            bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat",
            padx=12, pady=6,
            state="normal" if input_ok else "disabled",
        )
        self._wizard_btn.pack(side="left")
        self._wizard_status = tk.Label(_wiz_btn_row, text="", bg=CARD, fg=FG, font=FONT)
        self._wizard_status.pack(side="left", padx=10)
        self._wizard_progress = ttk.Progressbar(wizard_row, mode="indeterminate", length=400)
        self._wizard_progress.pack(anchor="w", pady=(6, 0))

        filt_grid = tk.Frame(filt_card, bg=CARD)
        filt_grid.pack(fill="x")
        filt_grid.grid_columnconfigure(0, minsize=240)
        filt_grid.grid_columnconfigure(1, minsize=80)
        filt_grid.grid_columnconfigure(2, weight=1)
        _filt_rows = [
            ("Min object size (pixels):",     "seg_min_object_size", "e.g. 1200"),
            ("Max object size (pixels):",     "seg_max_object_size", "e.g. 30000"),
            ("Max major / minor axis ratio:", "seg_max_ratio",       "e.g. 15"),
            ("Min mean raw intensity:",       "seg_raw_mean_min",    "e.g. 5000"),
        ]
        self._global_filt_entries = []
        self._global_filt_labels  = []
        for row_i, (lbl, key, hint) in enumerate(_filt_rows):
            lbl_w = tk.Label(filt_grid, text=lbl, bg=CARD, fg=FG, font=FONT)
            lbl_w.grid(row=row_i, column=0, sticky="w", pady=3)
            self._global_filt_labels.append(lbl_w)
            ent = tk.Entry(filt_grid, textvariable=self._vars[key], bg=BG, fg=FG,
                           insertbackground=FG, relief="flat", font=FONT, width=12)
            ent.grid(row=row_i, column=1, sticky="w", pady=3, padx=8)
            self._global_filt_entries.append(ent)
            tk.Label(filt_grid, text=hint, bg=CARD, fg=MUTED, font=FONT).grid(
                row=row_i, column=2, sticky="w", pady=3)

        # ── Per-channel filter overrides ──────────────────────────────
        self._per_ch_filt_enabled = tk.BooleanVar(
            value=bool(self.config.get("seg_per_channel_filters_enabled", False))
        )
        tk.Checkbutton(
            filt_card,
            text="Set individual filter thresholds per channel",
            variable=self._per_ch_filt_enabled,
            command=self._toggle_per_ch_filters,
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        ).pack(anchor="w", pady=(12, 2))
        self._per_ch_filt_frame = tk.Frame(filt_card, bg=CARD)
        self._per_ch_filter_vars: dict = {}
        # Rebuild rows whenever the channel list changes
        self._vars["seg_channels"].trace_add(
            "write", lambda *_: self._rebuild_per_ch_filter_rows()
        )
        if self._per_ch_filt_enabled.get():
            self._per_ch_filt_frame.pack(fill="x", padx=(20, 0), pady=(0, 4))
            self._rebuild_per_ch_filter_rows()

        self._form_grid(self._card(parent, "Tiling"), [
            ("Tile height (px):",             "seg_tile_h"),
            ("Tile width (px):",              "seg_tile_w"),
            ("Halo height (px):",             "seg_halo_h"),
            ("Halo width (px):",              "seg_halo_w"),
            ("QC overlay downsample factor:", "seg_qc_downsample"),
        ])

        flags_card = self._card(parent, "Run Flags")
        tk.Checkbutton(
            flags_card, text="Test run — first file only, saves per-tile debug PNGs",
            variable=self._vars["seg_test_run"],
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        ).pack(anchor="w", pady=2)
        tk.Checkbutton(
            flags_card,
            text="Skip subfolders — only scan files directly in the input folder"
                 "  (if unchecked, also scans one level of subfolders)",
            variable=self._vars["seg_skip_subdirs"],
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
            wraplength=660, justify="left",
        ).pack(anchor="w", pady=2)

        # ── Run button + full log ────────────────────────────────────
        run_card = self._card(parent, "Run Segmentation")
        btn_row = tk.Frame(run_card, bg=CARD)
        btn_row.pack(anchor="w")
        self._seg_run_btn = tk.Button(
            btn_row, text="▶  Run Segmentation",
            command=self._run_segmentation_step,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
            padx=18, pady=10,
            state="normal" if input_ok else "disabled",
        )
        self._seg_run_btn.pack(side="left")

        # "Continue a previous run" appears ONLY when an unfinished run is
        # detected for this project (progress recorded but not finished).
        _prog = self._seg_run_progress()
        if input_ok and _prog and not _prog.get("finished", True):
            self._seg_resume_btn = tk.Button(
                btn_row, text="⟲  Continue a previous run",
                command=lambda: self._run_segmentation_step(resume=True),
                bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat",
                padx=14, pady=10,
            )
            self._seg_resume_btn.pack(side="left", padx=(8, 0))

        self._seg_status = tk.Label(btn_row, text="", bg=CARD, fg=MUTED, font=FONT)
        self._seg_status.pack(side="left", padx=12)

        if _prog and not _prog.get("finished", True):
            _n = int(_prog.get("n_completed", 0))
            _tot = int(_prog.get("total", 0))
            tk.Label(
                run_card,
                text=f"⚠  Previous run did not finish — {_n} of {_tot} file(s) segmented in "
                     f"“{os.path.basename(_prog.get('folder',''))}”.  "
                     "Click “Continue a previous run” to segment the rest (finished files are skipped).",
                bg=CARD, fg=YELLOW, font=FONT, wraplength=700, justify="left",
            ).pack(anchor="w", pady=(2, 0))

        self._seg_progress = ttk.Progressbar(run_card, mode="indeterminate", length=500)
        self._seg_progress.pack(anchor="w", pady=(8, 4))

        self._seg_log = scrolledtext.ScrolledText(
            run_card, bg="#0d0d15", fg="#a6e3a1",
            font=FONT_MONO, height=14, relief="flat", wrap="word",
            state="disabled",
        )
        self._seg_log.pack(fill="x", pady=(4, 0))

    # ------------------------------------------------------------------
    # Per-channel filter helpers
    # ------------------------------------------------------------------

    def _toggle_per_ch_filters(self) -> None:
        enabled = self._per_ch_filt_enabled.get()
        for ent in getattr(self, "_global_filt_entries", []):
            ent.config(state="disabled" if enabled else "normal",
                       fg=MUTED if enabled else FG,
                       disabledforeground=MUTED)
        for lbl in getattr(self, "_global_filt_labels", []):
            lbl.config(fg=MUTED if enabled else FG)
        if enabled:
            self._per_ch_filt_frame.pack(fill="x", padx=(20, 0), pady=(0, 4))
            self._rebuild_per_ch_filter_rows()
        else:
            self._per_ch_filt_frame.pack_forget()
        self._save_per_ch_filter_config()

    def _rebuild_per_ch_filter_rows(self) -> None:
        if not hasattr(self, "_per_ch_filt_frame") or not self._per_ch_filt_enabled.get():
            return
        for w in self._per_ch_filt_frame.winfo_children():
            w.destroy()
        self._per_ch_filter_vars = {}

        channels = [
            int(x.strip())
            for x in self._vars["seg_channels"].get().split(",")
            if x.strip().isdigit()
        ]
        if not channels:
            tk.Label(self._per_ch_filt_frame, text="No channels configured yet.",
                     bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        import json as _json
        try:
            existing = _json.loads(self.config.get("seg_per_channel_filters", "{}") or "{}")
        except Exception:
            existing = {}

        # Header
        hdr = tk.Frame(self._per_ch_filt_frame, bg=CARD)
        hdr.pack(fill="x", pady=(2, 4))
        for col, text in enumerate(["", "Min size", "Max size", "Max ratio", "Min intensity"]):
            tk.Label(hdr, text=text, bg=CARD, fg=MUTED, font=FONT,
                     width=8 if col > 0 else 7, anchor="w").grid(row=0, column=col, padx=4)

        defaults = {
            "min_size":     self._vars["seg_min_object_size"].get(),
            "max_size":     self._vars["seg_max_object_size"].get(),
            "max_ratio":    self._vars["seg_max_ratio"].get(),
            "min_intensity":self._vars["seg_raw_mean_min"].get(),
        }
        for ch in channels:
            ch_cfg = existing.get(str(ch), {})
            row = tk.Frame(self._per_ch_filt_frame, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"Ch {ch}:", bg=CARD, fg=FG, font=FONT_BOLD,
                     width=7, anchor="w").pack(side="left", padx=(0, 4))
            ch_vars = {}
            for key in ("min_size", "max_size", "max_ratio", "min_intensity"):
                var = tk.StringVar(value=str(ch_cfg.get(key, defaults[key])))
                var.trace_add("write", lambda *_, c=ch: self._save_per_ch_filter_config())
                tk.Entry(row, textvariable=var, bg=BG, fg=FG,
                         insertbackground=FG, relief="flat", font=FONT, width=9).pack(side="left", padx=4)
                ch_vars[key] = var
            self._per_ch_filter_vars[ch] = ch_vars

    def _save_per_ch_filter_config(self) -> None:
        import json as _json
        data = {
            str(ch): {k: v.get() for k, v in ch_vars.items()}
            for ch, ch_vars in self._per_ch_filter_vars.items()
        }
        self.config["seg_per_channel_filters"] = _json.dumps(data)
        self.config["seg_per_channel_filters_enabled"] = self._per_ch_filt_enabled.get()
        save_config(self.config)

    # ------------------------------------------------------------------
    # Segmentation channel preview helpers
    # ------------------------------------------------------------------

    _INPUT_EXTS = (".czi", ".lif", ".nd2", ".oib", ".oif", ".tif", ".tiff")

    def _folder_image_count(self, folder: str) -> int:
        """Number of readable input images directly in `folder` or one level
        down (matches how the input folder is scanned)."""
        if not folder or not os.path.isdir(folder):
            return 0
        n = 0
        try:
            for item in os.listdir(folder):
                full = os.path.join(folder, item)
                if os.path.isfile(full) and item.lower().endswith(self._INPUT_EXTS):
                    n += 1
                elif os.path.isdir(full):
                    try:
                        for f in os.listdir(full):
                            if f.lower().endswith(self._INPUT_EXTS):
                                n += 1
                    except OSError:
                        pass
        except OSError:
            pass
        return n

    def _direct_image_count(self, folder: str) -> int:
        """Number of input images located DIRECTLY in `folder` (not in subdirs)."""
        if not folder or not os.path.isdir(folder):
            return 0
        n = 0
        try:
            for item in os.listdir(folder):
                if item.lower().endswith(self._INPUT_EXTS) \
                        and os.path.isfile(os.path.join(folder, item)):
                    n += 1
        except OSError:
            pass
        return n

    def _seg_run_progress(self) -> "dict | None":
        """Find the most recently updated unfinished Segmentation run.

        Returns the parsed dict (with an added 'folder' key) or None. Segmentation
        writes/updates <Segmentation*>/_run_progress.json after every image, so an
        unfinished run is detectable here (finished == False)."""
        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            return None
        variants = self._find_variants(SUBFOLDER["seg"])
        if not variants:
            return None
        unfinished = []
        for variant in variants:
            folder = os.path.join(proj, variant)
            ppath = os.path.join(folder, "_run_progress.json")
            if not os.path.isfile(ppath):
                continue
            try:
                with open(ppath) as f:
                    data = json.load(f)
                if not data.get("finished", True):
                    data["folder"] = folder
                    unfinished.append((os.path.getmtime(ppath), data))
            except (OSError, ValueError, TypeError):
                continue
        return max(unfinished, key=lambda pair: pair[0])[1] if unfinished else None

    def _find_input_variants(self) -> list:
        """[(display_name, images_folder)] for every converter output
        (VSI_Converted_TIFF[_*]) under the project — so the user can pick which
        one to segment, the same way Validation/Analysis pick a segmentation
        output. The images folder is the variant's `exports` subfolder when it
        exists, else the variant folder itself."""
        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            return []
        out = []
        for name in self._find_variants(SUBFOLDER["vsi"]):
            base = os.path.join(proj, name)
            exp = os.path.join(base, "exports")
            out.append((name, exp if os.path.isdir(exp) else base))
        return out

    def _input_converter_variant(self, images_folder: str) -> str | None:
        """Identify in-project exports, excluding links to outside folders."""
        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not images_folder:
            return None
        for name, path in self._find_input_variants():
            if os.path.normpath(path) != os.path.normpath(images_folder):
                continue
            try:
                relative = os.path.relpath(
                    os.path.realpath(images_folder), os.path.realpath(proj))
            except ValueError:  # Different drives on Windows.
                continue
            parts = relative.split(os.sep)
            if parts[0] == name and (len(parts) == 1 or parts[1:] == ["exports"]):
                return name
        return None

    def _align_converter_roi_folder(self, images_folder: str,
                                    previous_output: str | None = None) -> None:
        """Keep automatic ROI masks paired with the chosen converter output.

        A manually selected mask folder outside converter outputs remains a
        manual choice. This also works after a project has moved, because the
        destination is rebuilt from the current project directory.
        """
        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not images_folder:
            return
        selected = self._input_converter_variant(images_folder)
        if selected is None:
            return
        prior_output = (self.config.get("vsi_output_dir", "").strip()
                        if previous_output is None else previous_output)
        current = str(self.config.get("ana_roi_mask_folder") or "").strip()
        auto_paths = self._converter_roi_paths()
        if prior_output:
            auto_paths.add(os.path.normpath(os.path.join(
                prior_output, "DONOTDELETE-MASK_FOR_VSI")))
        new_output = os.path.join(proj, selected)
        self.config["vsi_output_dir"] = new_output
        if current and os.path.normpath(current) not in auto_paths:
            return
        new_masks = os.path.join(new_output, "DONOTDELETE-MASK_FOR_VSI")
        self.config["ana_roi_mask_folder"] = new_masks
        if "ana_roi_mask_folder" in self._vars:
            self._vars["ana_roi_mask_folder"].set(new_masks)

    def _clear_converter_roi_folder_for_direct_input(self) -> None:
        """Avoid applying an automatic slide ROI mask to direct-read images."""
        if self.config.get("input_type") == "vsi":
            return
        current = str(self.config.get("ana_roi_mask_folder") or "").strip()
        if not current:
            return
        if os.path.normpath(current) in self._converter_roi_paths():
            self.config["ana_roi_mask_folder"] = ""
            if "ana_roi_mask_folder" in self._vars:
                self._vars["ana_roi_mask_folder"].set("")

    def _sync_roi_mask_to_input(self) -> None:
        if self.config.get("input_type") == "vsi":
            self._align_converter_roi_folder(
                self.config.get("input_root_folder", "").strip())
        else:
            self._clear_converter_roi_folder_for_direct_input()

    def _resolve_existing_input_folder(self) -> str:
        """Return a folder that actually contains input images.

        If the stored `input_root_folder` is missing or empty (e.g. the
        converter output folder was renamed/moved after the project was saved),
        look under the project for the converter `exports` folder — so the
        preview and segmentation keep working instead of silently finding
        nothing. Output folders (Segmentation/Validation/Analysis) and the
        converter's intermediate/mask folders are never chosen as input.
        """
        cfg_folder = self.config.get("input_root_folder", "").strip()
        if self._folder_image_count(cfg_folder) > 0:
            return cfg_folder

        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            return cfg_folder

        exclude = {SUBFOLDER["seg"], SUBFOLDER["val"], SUBFOLDER["ana"]}
        # Candidates are *leaf* folders that hold images directly; prefer the
        # converter's `exports` subfolder.
        candidates = [proj]
        try:
            for name in sorted(os.listdir(proj)):
                if any(name == e or name.startswith(e + "_") for e in exclude):
                    continue
                full = os.path.join(proj, name)
                if os.path.isdir(full):
                    candidates.append(os.path.join(full, "exports"))  # converter output
                    candidates.append(full)
        except OSError:
            pass
        best, best_n = cfg_folder, 0
        for c in candidates:
            n = self._direct_image_count(c)  # leaf folder = images directly inside
            if n > best_n:
                best, best_n = c, n
        return best

    def _find_first_image(self) -> tuple[str, int | None] | None:
        folder = self._resolve_existing_input_folder()
        # Self-heal: if we relocated to a valid folder, remember it so the
        # segmentation run uses the same images.
        cfg = self.config.get("input_root_folder", "").strip()
        if folder and folder != cfg and self._folder_image_count(folder) > 0:
            self.config["input_root_folder"] = folder
        return _first_confocal_preview_input(
            folder, self._INPUT_EXTS,
            bool(self._vars["seg_skip_subdirs"].get()),
        )

    def _load_channel_previews(self, callback=None) -> None:
        """Run in daemon thread. Posts result back to main thread via callback."""
        if callback is None:
            callback = self._set_channel_previews
        if not _PIL:
            return
        preview_input = self._find_first_image()
        if not preview_input:
            looked = self.config.get("input_root_folder", "").strip()
            if not looked:
                msg = "No input folder set — go to Step 2."
            elif not os.path.isdir(looked):
                msg = f"Input folder not found:  {looked}"
            elif _scanned_input_files(
                    looked, self._INPUT_EXTS,
                    bool(self._vars["seg_skip_subdirs"].get())):
                msg = ("No readable confocal input available for preview. "
                       "Check the file list in Step 2; slide-scanner LIF "
                       "files belong in Slide scanner mode.")
            else:
                msg = f"No images found in:  {looked}"
            self.root.after(0, callback, [], msg)
            return
        path, lif_scene = preview_input
        try:
            import numpy as np
            fname = os.path.basename(path)
            print(f"\n[Preview] Loading: {fname}")

            if path.lower().endswith(".czi"):
                channels_np = None   # filled by fast or slow path below

                # ── Fast path: aicspylibczi — reads ONE Z-slice per channel ──
                # aicspylibczi is a thin C++ wrapper (no tifffile dependency) and
                # supports reading individual dimensions, so we only decompress
                # 1/num_Z of the data for the preview.
                try:
                    from aicspylibczi import CziFile as _AicsCziFile
                    print("[Preview] CZI fast-path via aicspylibczi (single Z per channel)…")
                    with _AicsCziFile(path) as _aczi:
                        _dims = _aczi.get_dims_shape()
                        _d = _dims[0] if _dims else {}
                        if 'C' in _d:
                            _num_c = _d['C'][1]
                            _num_z = _d.get('Z', (0, 1))[1]
                            _mid_z = min(_num_z // 2, max(0, _num_z - 1))
                            print(f"[Preview]   {_num_c} channel(s), {_num_z} Z — reading Z={_mid_z}")
                            channels_np = []
                            for _c in range(_num_c):
                                _kw = {'C': _c}
                                if 'Z' in _d:
                                    _kw['Z'] = _mid_z
                                _img, _ = _aczi.read_image(**_kw)
                                _arr = np.squeeze(_img)
                                while _arr.ndim > 2:
                                    _arr = np.max(_arr, axis=0)
                                channels_np.append(_arr)
                                print(f"[Preview]   ch{_c}: {_arr.shape}")
                except Exception as _fe:
                    print(f"[Preview] aicspylibczi fast-path failed ({_fe}), using czifile…")
                    channels_np = None

                # ── Slow path: czifile full read ─────────────────────────────
                if channels_np is None:
                    from czifile import CziFile
                    print("[Preview] Reading full CZI with czifile (may be slow for large files)…")
                    with CziFile(path) as czi:
                        data = czi.asarray()
                        axes = czi.axes
                        c_axis = axes.find("C")
                        z_axis = axes.find("Z")
                    print(f"[Preview] CZI axes: {axes}, shape: {data.shape}")
                    if c_axis == -1:
                        data_sq = np.squeeze(data)
                        while data_sq.ndim > 2:
                            data_sq = np.max(data_sq, axis=0)
                        channels_np = [data_sq]
                        print("[Preview] No C axis — treated as single channel after MIP")
                    else:
                        num_channels = data.shape[c_axis]
                        num_z = data.shape[z_axis] if z_axis != -1 else 1
                        print(f"[Preview] Channels: {num_channels}, Z-slices: {num_z}")
                        if num_z > 1:
                            print("[Preview] Z-stack — applying MIP per channel")
                        channels_np = []
                        for c in range(num_channels):
                            ch = np.take(data, c, axis=c_axis)
                            ch = np.squeeze(ch)
                            while ch.ndim > 2:
                                ch = np.max(ch, axis=0)
                            channels_np.append(ch)
                            print(f"[Preview]   Channel {c}: shape={ch.shape}")
            elif path.lower().endswith(".lif"):
                from readlif.reader import LifFile
                print("[Preview] Reading Leica LIF via readlif…")
                _lif = LifFile(path)
                _img = _lif.get_image(lif_scene)
                _nc = int(_img.channels)
                _nz = max(int(_img.nz), 1)
                print(f"[Preview] LIF '{_img.name}': {_nc} channel(s), {_nz} Z")
                if _nz > 1:
                    print("[Preview] Z-stack — applying MIP per channel")
                channels_np = []
                for _c in range(_nc):
                    _frames = [np.asarray(_img.get_frame(z=_z, t=0, c=_c))
                               for _z in range(_nz)]
                    _ch = _frames[0] if _nz == 1 else np.max(np.stack(_frames, axis=0), axis=0)
                    _ch = np.squeeze(_ch)
                    while _ch.ndim > 2:
                        _ch = np.max(_ch, axis=0)
                    channels_np.append(_ch)
                    print(f"[Preview]   Channel {_c}: shape={_ch.shape}")
            elif path.lower().endswith((".nd2", ".oib", ".oif")):
                # Nikon ND2 / Olympus OIB-OIF: read array + axis labels, then
                # drop T/P/M axes (first index), MIP Z, split channels.
                if path.lower().endswith(".nd2"):
                    import nd2
                    print("[Preview] Reading Nikon ND2 via nd2…")
                    with nd2.ND2File(path) as _f:
                        _axes = list("".join(_f.sizes.keys()).upper())
                        _data = _f.asarray()
                else:
                    from oiffile import OifFile
                    print("[Preview] Reading Olympus OIB/OIF via oiffile…")
                    with OifFile(path) as _oif:
                        _data = _oif.asarray()
                        _axes = list(_oif.axes.upper())
                print(f"[Preview] axes: {''.join(_axes)}, shape: {_data.shape}")
                for _i in range(len(_axes) - 1, -1, -1):
                    if _axes[_i] not in ("C", "Z", "Y", "X"):
                        _data = np.take(_data, 0, axis=_i)
                        _axes.pop(_i)
                _axes = "".join(_axes)
                _ci = _axes.find("C")
                if _ci == -1:
                    _d = np.squeeze(_data)
                    while _d.ndim > 2:
                        _d = np.max(_d, axis=0)
                    channels_np = [_d]
                else:
                    channels_np = []
                    for _c in range(_data.shape[_ci]):
                        _ch = np.squeeze(np.take(_data, _c, axis=_ci))
                        while _ch.ndim > 2:
                            _ch = np.max(_ch, axis=0)
                        channels_np.append(_ch)
                        print(f"[Preview]   Channel {_c}: shape={_ch.shape}")
            else:
                import tifffile as _tf
                with _tf.TiffFile(path) as tif:
                    data = tif.asarray()
                    axes_meta = (tif.series[0].axes if tif.series else "").upper()
                print(f"[Preview] TIFF shape: {data.shape}, axes: '{axes_meta or 'unknown'}'")
                data = np.squeeze(data)

                if data.ndim == 2:
                    print("[Preview] Single-channel 2D image")
                    channels_np = [data]
                elif data.ndim == 3:
                    has_z = "Z" in axes_meta
                    has_c = "C" in axes_meta
                    if has_z and not has_c:
                        print(f"[Preview] Z-stack ({data.shape[0]} slices) — applying MIP")
                        channels_np = [np.max(data, axis=0)]
                    elif data.shape[0] <= 16 and data.shape[0] < data.shape[1] // 2:
                        print(f"[Preview] Channel-first: {data.shape[0]} channels")
                        channels_np = [data[c] for c in range(data.shape[0])]
                    elif data.shape[2] <= 16 and data.shape[2] < data.shape[0] // 2:
                        print(f"[Preview] Channel-last: {data.shape[2]} channels")
                        channels_np = [data[:, :, c] for c in range(data.shape[2])]
                    else:
                        print(f"[Preview] Ambiguous 3D — applying MIP as single-channel Z-stack")
                        channels_np = [np.max(data, axis=0)]
                elif data.ndim == 4:
                    # (C, Z, Y, X) or (Z, C, Y, X): pick C as smaller of first two dims
                    c_pos = 0 if data.shape[0] <= data.shape[1] else 1
                    num_channels = data.shape[c_pos]
                    print(f"[Preview] 4D — {num_channels} channels, applying MIP over Z")
                    channels_np = []
                    for c in range(num_channels):
                        ch = np.take(data, c, axis=c_pos)
                        channels_np.append(np.max(ch, axis=0))
                else:
                    print(f"[Preview] Unsupported shape {data.shape} — cannot preview")
                    channels_np = []

            from PIL import Image as PILImage
            photos = []
            for i, ch in enumerate(channels_np):
                ch = ch.astype(float)
                lo, hi = np.percentile(ch, [1, 99.5])
                if hi > lo:
                    ch = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
                else:
                    ch = np.zeros_like(ch)
                ch = ch.astype("uint8")
                img = PILImage.fromarray(ch, mode="L")
                img.thumbnail((280, 200), PILImage.LANCZOS)
                photos.append(ImageTk.PhotoImage(img))

            print(f"[Preview] Done — {len(photos)} channel(s) loaded")
            self.root.after(0, callback, photos, fname)
        except Exception as exc:
            print(f"[Preview] Error: {exc}")
            self.root.after(0, callback, [], f"Preview error: {exc}")

    def _trigger_ch_preview_load(self) -> None:
        """Button handler for Step 3 — load channel preview in background."""
        if hasattr(self, "_ch_load_btn") and self._ch_load_btn:
            self._ch_load_btn.config(state="disabled", text="  Loading…")
        self._ch_lbl.config(text="Loading…", fg=ACCENT)
        if hasattr(self, "_ch_load_progress"):
            self._ch_load_progress.pack(anchor="w", pady=(0, 6))
            self._ch_load_progress.start(12)
        threading.Thread(target=self._load_channel_previews, daemon=True).start()

    def _set_channel_previews(self, photos: list, fname: str = "") -> None:
        self._ch_images = photos
        self._ch_photo_refs = photos  # keep refs alive
        self._ch_index = 0
        if hasattr(self, "_ch_load_btn") and self._ch_load_btn:
            try:
                self._ch_load_btn.config(state="normal", text="⟳  Load Preview")
            except tk.TclError:
                pass
        if hasattr(self, "_ch_load_progress"):
            try:
                self._ch_load_progress.stop()
                self._ch_load_progress.pack_forget()
            except tk.TclError:
                pass
        if not photos:
            self._ch_lbl.config(text=fname or "No preview available", fg=MUTED)
            return
        self._show_channel(0)
        # Restore any previously selected channels from the text field
        raw = self._vars["seg_channels"].get().strip()
        self._ch_selected = {int(x.strip()) for x in raw.split(",")
                             if x.strip().isdigit()}

    def _show_channel(self, idx: int) -> None:
        if not self._ch_images:
            return
        idx = max(0, min(idx, len(self._ch_images) - 1))
        self._ch_index = idx
        n = len(self._ch_images)
        selected = "  ✔ selected" if idx in self._ch_selected else "  (click to add)"
        self._ch_lbl.config(
            text=f"Channel {idx} of {n - 1}{selected}",
            fg=GREEN if idx in self._ch_selected else FG,
        )
        self._ch_preview_img.config(image=self._ch_images[idx])
        self._ch_prev_btn.config(state="normal" if idx > 0 else "disabled")
        self._ch_next_btn.config(state="normal" if idx < n - 1 else "disabled")

    def _ch_navigate(self, delta: int) -> None:
        self._show_channel(self._ch_index + delta)

    def _ch_toggle_current(self) -> None:
        if not self._ch_images:
            return
        idx = self._ch_index
        if idx in self._ch_selected:
            self._ch_selected.discard(idx)
        else:
            self._ch_selected.add(idx)
        # Sync back to the text field
        sorted_chs = sorted(self._ch_selected)
        self._vars["seg_channels"].set(",".join(str(c) for c in sorted_chs))
        self._show_channel(idx)

    # ------------------------------------------------------------------
    # Validation channel name / color grid
    # ------------------------------------------------------------------

    _VAL_COLOR_OPTIONS  = ["Blue", "Green", "Red", "Purple", "Cyan", "Yellow", "White"]
    _VAL_COLOR_DEFAULTS = ["Blue", "Green", "Red", "Purple", "Cyan", "Yellow", "White"]

    def _detect_all_channel_labels(self) -> list:
        """Return sorted channel-index strings for ALL channels in the images.

        Scans the first project subfolder of the selected segmentation output
        for *_split_ch*.tif files — these are written for every channel,
        not just the ones that were predicted.  Falls back to seg_channels
        config if no split files are found yet.
        """
        import re as _re
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            return []

        seg_sub = (
            (self._val_input_var.get()
             if hasattr(self, "_val_input_var") and self._val_input_var
             and self._val_input_var.get() else None)
            or (self._find_variants(SUBFOLDER["seg"]) or [SUBFOLDER["seg"]])[-1]
        )
        seg_root = os.path.join(proj, seg_sub)

        if os.path.isdir(seg_root):
            for d in sorted(os.listdir(seg_root)):
                sub = os.path.join(seg_root, d)
                if not os.path.isdir(sub) or d.startswith((".", "DEBUG")):
                    continue
                chs = set()
                for f in os.listdir(sub):
                    m = _re.search(r"_split_ch(\d+)\.tif$", f, _re.IGNORECASE)
                    if m:
                        chs.add(int(m.group(1)))
                if chs:
                    return [str(c) for c in sorted(chs)]

        # Fallback: use seg_channels from config
        return [x.strip() for x in
                self.config.get("seg_channels", "").split(",")
                if x.strip().isdigit()]

    def _detect_predicted_channels(self) -> list:
        """Return sorted channel-index strings that actually have a predicted
        mask (``*_predict_original_ch*.tif``) on disk, scanning the selected
        segmentation output folder.

        This lets Validation / Analysis work from an imported project on
        another computer even when the project file no longer records that a
        segmentation was run.  Falls back to the configured seg_channels.
        """
        import re as _re
        proj = self.config.get("project_output_folder", "").strip()
        if proj and os.path.isdir(proj):
            # The segmentation-folder selection is kept in config and shared
            # between the Validation and Analysis pages; fall back to newest.
            seg_sub = (self.config.get("seg_input_subfolder", "").strip()
                       or (self._find_variants(SUBFOLDER["seg"])
                           or [SUBFOLDER["seg"]])[-1])
            seg_root = os.path.join(proj, seg_sub)
            chs = set()
            if os.path.isdir(seg_root):
                for d in sorted(os.listdir(seg_root)):
                    sub = os.path.join(seg_root, d)
                    if not os.path.isdir(sub) or d.startswith((".", "DEBUG")):
                        continue
                    for f in os.listdir(sub):
                        m = _re.search(r"_predict_original_ch(\d+)", f,
                                       _re.IGNORECASE)
                        if m:
                            chs.add(int(m.group(1)))
            if chs:
                return [str(c) for c in sorted(chs)]
        # Fallback: use seg_channels from config
        return [x.strip() for x in
                self.config.get("seg_channels", "").split(",")
                if x.strip().isdigit()]

    def _build_val_ch_name_rows(self, channel_labels: list) -> None:
        """Populate the channel name/color entry grid inside _val_ch_names_frame.

        channel_labels — list of strings like ["0","1","2"] used as display IDs.
        Existing typed names/colors are preserved via the val_channel_names/_colors vars.
        """
        if not hasattr(self, "_val_ch_names_frame") or self._val_ch_names_frame is None:
            return
        for w in self._val_ch_names_frame.winfo_children():
            w.destroy()
        self._val_ch_name_vars  = []
        self._val_ch_color_vars = []

        n = len(channel_labels)
        if n == 0:
            tk.Label(self._val_ch_names_frame,
                     text="Set channels in Step 3 — Segmentation to configure names.",
                     bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        saved_names  = [s.strip() for s in
                        self._vars["val_channel_names"].get().split(",")]
        saved_colors = [s.strip() for s in
                        self._vars["val_channel_colors"].get().split(",")]

        g = tk.Frame(self._val_ch_names_frame, bg=CARD)
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=100)
        g.grid_columnconfigure(1, minsize=180)
        g.grid_columnconfigure(2, minsize=20)
        g.grid_columnconfigure(3, minsize=120)
        tk.Label(g, text="Name",  bg=CARD, fg=MUTED, font=FONT).grid(
            row=0, column=1, sticky="w", padx=8, pady=(0, 4))
        tk.Label(g, text="Color", bg=CARD, fg=MUTED, font=FONT).grid(
            row=0, column=3, sticky="w", pady=(0, 4))

        for i, ch_label in enumerate(channel_labels):
            def_name  = (saved_names[i]  if i < len(saved_names)  and saved_names[i]
                         else f"Channel {ch_label}")
            def_color = (saved_colors[i] if i < len(saved_colors) and saved_colors[i]
                         else self._VAL_COLOR_DEFAULTS[i % len(self._VAL_COLOR_DEFAULTS)])
            name_var  = tk.StringVar(value=def_name)
            color_var = tk.StringVar(value=def_color)
            name_var.trace_add( "write", lambda *_: self._update_val_channel_names_str())
            color_var.trace_add("write", lambda *_: self._update_val_channel_names_str())
            self._val_ch_name_vars.append(name_var)
            self._val_ch_color_vars.append(color_var)
            tk.Label(g, text=f"Channel {ch_label}:", bg=CARD, fg=FG,
                     font=FONT_BOLD).grid(row=i + 1, column=0, sticky="w", pady=3)
            tk.Entry(g, textvariable=name_var, bg=BG, fg=FG,
                     insertbackground=FG, relief="flat",
                     font=FONT, width=22).grid(
                row=i + 1, column=1, sticky="w", pady=3, padx=8)
            _cb = ttk.Combobox(g, textvariable=color_var,
                               values=self._VAL_COLOR_OPTIONS,
                               state="readonly", width=10)
            _cb.grid(row=i + 1, column=3, sticky="w", pady=3)
            self._bind_combobox_scroll(_cb)

        self._update_val_channel_names_str()

    # ------------------------------------------------------------------
    # Validation channel preview helpers
    # ------------------------------------------------------------------

    def _trigger_val_preview_load(self) -> None:
        """Button handler — disable the button, show spinner, load in background."""
        if hasattr(self, "_val_load_btn") and self._val_load_btn:
            self._val_load_btn.config(state="disabled", text="  Loading…")
        if self._val_lbl:
            self._val_lbl.config(text="Loading…", fg=ACCENT)
        threading.Thread(
            target=lambda: self._load_channel_previews(
                callback=self._set_val_channel_previews),
            daemon=True,
        ).start()

    def _set_val_channel_previews(self, photos: list, fname: str = "") -> None:
        """Called on the main thread after background channel loading."""
        self._val_ch_images = photos
        self._val_ch_photo_refs = photos
        self._val_ch_index = 0

        # Rebuild name/color rows using the actual detected channel count
        if self._val_ch_names_frame:
            n = len(photos) if photos else 0
            if n > 0:
                self._build_val_ch_name_rows([str(i) for i in range(n)])
            elif fname:
                for w in self._val_ch_names_frame.winfo_children():
                    w.destroy()
                tk.Label(self._val_ch_names_frame,
                         text=fname or "Could not detect channels.",
                         bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")

        # Re-enable the load button
        if hasattr(self, "_val_load_btn") and self._val_load_btn:
            try:
                self._val_load_btn.config(state="normal", text="⟳  Load Preview")
            except tk.TclError:
                pass

        if not photos:
            if self._val_lbl:
                self._val_lbl.config(text=fname or "No preview available", fg=MUTED)
            return

        self._show_val_channel(0)

    def _show_val_channel(self, idx: int) -> None:
        if not self._val_ch_images:
            return
        idx = max(0, min(idx, len(self._val_ch_images) - 1))
        self._val_ch_index = idx
        n = len(self._val_ch_images)
        if self._val_lbl:
            self._val_lbl.config(text=f"Channel {idx} of {n - 1}", fg=FG)
        if self._val_preview_img:
            self._val_preview_img.config(image=self._val_ch_images[idx])
        if self._val_prev_btn:
            self._val_prev_btn.config(state="normal" if idx > 0 else "disabled")
        if self._val_next_btn:
            self._val_next_btn.config(
                state="normal" if idx < n - 1 else "disabled")

    def _val_ch_navigate(self, delta: int) -> None:
        self._show_val_channel(self._val_ch_index + delta)

    def _update_val_channel_names_str(self) -> None:
        """Sync name and color entry vars back to the flat config vars."""
        names  = ",".join(v.get().strip() for v in self._val_ch_name_vars)
        colors = ",".join(v.get().strip() for v in self._val_ch_color_vars)
        self._vars["val_channel_names"].set(names)
        self._vars["val_channel_colors"].set(colors)

    def _run_segmentation_step(self, resume: bool = False) -> None:
        """Run segmentation from within the Segmentation step card.

        resume=True ("Continue a previous run"): re-use the existing
        Segmentation output and skip files already segmented.
        """
        if self._is_running("Segmentation"):
            if hasattr(self, "_seg_status"):
                self._seg_status.config(text="Segmentation is already running…", fg=ACCENT)
            return
        self._sync_vars_to_config()
        folder = self.config.get("input_root_folder", "").strip()
        channels = self.config.get("seg_channels", "").strip()

        if not folder:
            self._seg_status.config(text="⚠  No input folder set. Go to Step 2.", fg=RED)
            return
        if not channels:
            self._seg_status.config(text="⚠  Channels to predict is required (red ✱).", fg=RED)
            return

        if resume:
            # Resume precisely the unfinished run shown on this card.
            progress = self._seg_run_progress()
            if not progress or progress.get("finished", True):
                self._seg_status.config(
                    text="⚠  No previous Segmentation output found to continue.", fg=RED)
                return
            out = progress["folder"]
            saved_identity = progress.get("run_identity")
            if saved_identity is not None:
                try:
                    current_identity = self._segmentation_identity_from_config()
                except ValueError:
                    self._seg_status.config(
                        text="⚠  Check the channel list before continuing.", fg=RED)
                    return
                if saved_identity != current_identity:
                    messagebox.showwarning(
                        "Settings do not match this run",
                        "This unfinished run used a different method, model, "
                        "preprocessing mode, or channel list. Match its settings "
                        "before continuing, or start a new run. No masks were changed.",
                        parent=self.root,
                    )
                    return
            elif not messagebox.askyesno(
                "Continue older segmentation run?",
                "This older run has no saved method or settings, so the app "
                "cannot verify that the current choices match it. Continuing "
                "could mix methods in one folder.\n\n"
                f"Continue {os.path.basename(out)} anyway? Choose No to "
                "start a new run instead.",
                parent=self.root, default=messagebox.NO, icon="warning",
            ):
                return
        else:
            out = self._resolve_segmentation_output_dir()
            if out is None:
                return
        self.config["output_root_folder"] = out
        self.config["seg_input_subfolder"] = os.path.basename(out)
        self.config["seg_resume"] = bool(resume)
        save_config(self.config)
        save_project_config(self.config)

        if hasattr(self, "_seg_resume_btn"):
            self._seg_resume_btn.config(state="disabled")
        self._seg_run_btn.config(state="disabled", text="Running…")
        self._seg_progress.start(12)
        self._seg_status.config(text="Starting…", fg=ACCENT)

        # Clear log
        self._seg_log.config(state="normal")
        self._seg_log.delete("1.0", "end")
        self._seg_log.config(state="disabled")

        script = self._script("segmentation.py")

        def _seg_log_line(line: str) -> None:
            try:
                if not (hasattr(self, "_seg_log") and self._seg_log.winfo_exists()):
                    return
                self._seg_log.config(state="normal")
                self._seg_log.insert("end", line + "\n")
                self._seg_log.see("end")
                self._seg_log.config(state="disabled")
                # Also update the one-liner status with the last meaningful line
                if line.strip() and hasattr(self, "_seg_status") \
                        and self._seg_status.winfo_exists():
                    self._seg_status.config(text=line.strip()[:120], fg=MUTED)
            except tk.TclError:
                pass

        def _thread():
            try:
                proc = subprocess.Popen(
                    [sys.executable, script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._track_proc(proc, "Segmentation")
                for line in proc.stdout:
                    self.root.after(0, _seg_log_line, line.rstrip())
                proc.wait()
                ok = proc.returncode == 0
                def _done():
                    try:
                        self._seg_progress.stop()
                        self._seg_run_btn.config(state="normal", text="▶  Run Segmentation")
                        if hasattr(self, "_seg_resume_btn"):
                            self._seg_resume_btn.config(state="normal")
                        if ok:
                            self._seg_status.config(text="✔  Segmentation complete.", fg=GREEN)
                            _seg_log_line("\n✔  Segmentation complete.")
                        else:
                            self._seg_status.config(
                                text=f"✗  Failed (exit {proc.returncode}).", fg=RED)
                            _seg_log_line(f"\n✗  Failed (exit code {proc.returncode}).")
                    except tk.TclError:
                        pass
                self.root.after(0, _done)
            except Exception as exc:
                def _err():
                    try:
                        self._seg_progress.stop()
                        self._seg_run_btn.config(state="normal", text="▶  Run Segmentation")
                        if hasattr(self, "_seg_resume_btn"):
                            self._seg_resume_btn.config(state="normal")
                        self._seg_status.config(text=f"✗  {exc}", fg=RED)
                    except tk.TclError:
                        pass
                self.root.after(0, _err)

        threading.Thread(target=_thread, daemon=True).start()

    def _launch_parameter_wizard(self) -> None:
        """Launch the Parameter Wizard: runs unfiltered segmentation on the
        first sample only (parameter_wizard_compute.py, headless), then
        opens a napari window per channel (parameter_wizard.py) so the user
        can pick example objects by eye instead of guessing filter numbers.

        These are two SEPARATE subprocesses run back-to-back, not one --
        combining the segmentation stack (torch/cv2/micro_sam) with a live
        napari window in a single process crashes the interpreter on macOS
        here, so the heavy computation and the GUI never share a process.
        """
        if self._is_running("Parameter Wizard"):
            self._wizard_status.config(text="Parameter Wizard is already running…", fg=ACCENT)
            return
        self._sync_vars_to_config()
        folder = self.config.get("input_root_folder", "").strip()
        channels = self.config.get("seg_channels", "").strip()
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._wizard_status.config(text="⚠  No project folder set. Go to Step 1.", fg=RED)
            return
        if not folder:
            self._wizard_status.config(text="⚠  No input folder set. Go to Step 2.", fg=RED)
            return
        if not channels:
            self._wizard_status.config(text="⚠  Channels to predict is required (red ✱).", fg=RED)
            return

        save_config(self.config)
        save_project_config(self.config)

        out_dir = os.path.join(proj, "ParameterWizard_preview")
        manifest_path = os.path.join(out_dir, "parameter_wizard_manifest.json")

        self._wizard_btn.config(state="disabled", text="Running…")
        self._wizard_status.config(
            text="Initial prediction in progress — this can take a while "
                 "(first use of a model also downloads it, one time only)…",
            fg=ACCENT)
        self._wizard_progress.start(12)

        def _log_line(line: str) -> None:
            try:
                if hasattr(self, "_seg_log") and self._seg_log.winfo_exists():
                    self._seg_log.config(state="normal")
                    self._seg_log.insert("end", line + "\n")
                    self._seg_log.see("end")
                    self._seg_log.config(state="disabled")
                if line.strip() and hasattr(self, "_wizard_status") and self._wizard_status.winfo_exists():
                    self._wizard_status.config(text=line.strip()[:120], fg=FG)
            except tk.TclError:
                pass

        def _restore(final_text: str, color: str) -> None:
            try:
                self._wizard_progress.stop()
                self._wizard_btn.config(state="normal", text="🧙  Parameter Wizard")
                self._wizard_status.config(text=final_text, fg=color)
            except tk.TclError:
                pass

        def _run_and_stream(cmd: list) -> int:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self._track_proc(proc, "Parameter Wizard")
            for line in proc.stdout:
                self.root.after(0, _log_line, line.rstrip())
            proc.wait()
            return proc.returncode

        def _thread():
            try:
                self.root.after(0, _log_line, f"\n{'─'*60}")
                self.root.after(0, _log_line, "▶  Parameter Wizard — Stage 1: unfiltered segmentation")
                self.root.after(0, _log_line, f"{'─'*60}")
                rc = _run_and_stream([
                    sys.executable, self._script("parameter_wizard_compute.py"),
                    "--config", CONFIG_PATH, "--out-dir", out_dir,
                ])
                if rc != 0:
                    self.root.after(0, _log_line, f"\n✗  Stage 1 failed (exit {rc}).")
                    self.root.after(0, _restore, f"✗  Segmentation failed (exit {rc}).", RED)
                    return

                self.root.after(0, _log_line, f"\n{'─'*60}")
                self.root.after(0, _log_line, "▶  Parameter Wizard — Stage 2: pick objects (napari)")
                self.root.after(0, _log_line, f"{'─'*60}")
                self.root.after(0, lambda: self._wizard_status.config(
                    text="Opening napari — pick your objects there…", fg=ACCENT))
                rc = _run_and_stream([
                    sys.executable, self._script("parameter_wizard.py"),
                    "--config", CONFIG_PATH, "--manifest", manifest_path,
                ])

                def _done():
                    if rc == 0:
                        self.config = load_config()
                        self._per_ch_filt_enabled.set(True)
                        self._toggle_per_ch_filters()
                        _restore("✔  Filter values updated below.", GREEN)
                        _log_line("\n✔  Parameter Wizard finished — filter values updated.")
                    else:
                        _restore(f"✗  Failed (exit {rc}).", RED)
                        _log_line(f"\n✗  Parameter Wizard failed (exit code {rc}).")
                self.root.after(0, _done)
            except Exception as exc:
                self.root.after(0, _restore, f"✗  {exc}", RED)

        threading.Thread(target=_thread, daemon=True).start()

    def _launch_roi_editor(self) -> None:
        """Launch the napari ROI editor so the user can add missed cells before validation."""
        if self._is_running("Review Segmented Images window"):
            if hasattr(self, "_review_status"):
                self._review_status.config(
                    text="A Review window is already open.", fg=ACCENT)
            return
        self._sync_vars_to_config()
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            if hasattr(self, "_review_status"):
                self._review_status.config(text="⚠  No project folder set.", fg=RED)
            return

        # Resolve segmentation input folder (respects the dropdown selection)
        if self._val_input_var and self._val_input_var.get():
            seg_sub = self._val_input_var.get()
        else:
            variants = self._find_variants(SUBFOLDER["seg"])
            seg_sub  = variants[-1] if variants else SUBFOLDER["seg"]
        seg_folder = os.path.join(proj, seg_sub)

        if not os.path.isdir(seg_folder):
            if hasattr(self, "_review_status"):
                self._review_status.config(
                    text=f"⚠  Segmentation output not found: {seg_sub}", fg=RED)
            return

        # Derive channel from the channel dropdown
        channel = 2   # fallback
        if self._val_channel_var and self._val_channel_var.get():
            try:
                channel = int(self._val_channel_var.get().replace("Channel", "").strip())
            except ValueError:
                pass

        if hasattr(self, "_review_btn"):
            self._review_btn.config(state="disabled", text="  Opening…")
        if hasattr(self, "_review_status"):
            self._review_status.config(text="Launching napari…", fg=ACCENT)

        def _restore():
            if hasattr(self, "_review_btn"):
                try:
                    self._review_btn.config(state="normal",
                                            text="🔍  Review Segmented Images")
                except tk.TclError:
                    pass
            if hasattr(self, "_review_status"):
                try:
                    self._review_status.config(text="✔  Editor opened.", fg=GREEN)
                except tk.TclError:
                    pass

        try:
            cmd = [
                sys.executable,
                self._script("roi_editor.py"),
                "--seg-folder", seg_folder,
                "--channel",    str(channel),
            ]
            if self._review_test_mode and self._review_test_mode.get():
                cmd.append("--test-mode")
            self._track_proc(subprocess.Popen(cmd), "Review Segmented Images window")
            self.root.after(4000, _restore)
        except Exception as exc:
            _restore()
            if hasattr(self, "_review_status"):
                self._review_status.config(text=f"✗  {exc}", fg=RED)

    def _run_validation_step(self) -> None:
        """Launch the validator from the Validation step card."""
        if self._is_running("Validation window"):
            if hasattr(self, "_val_status"):
                self._val_status.config(
                    text="A Validation window is already open.", fg=ACCENT)
            return
        self._sync_vars_to_config()
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            if hasattr(self, "_val_status"):
                self._val_status.config(text="⚠  No project folder set. Go to Step 1.", fg=RED)
            return

        # Derive val_mask_pattern from selected channel
        if self._val_channel_var:
            ch_label = self._val_channel_var.get()
            ch = ch_label.replace("Channel ", "").strip()
            self.config["val_mask_pattern"] = f"*_predict_original_ch{ch}*.tif"

        # Use the selected segmentation input (set in Step 4 input selector)
        if self._val_input_var and self._val_input_var.get():
            seg_sub = self._val_input_var.get()
        else:
            seg_variants = self._find_variants(SUBFOLDER["seg"])
            seg_sub = seg_variants[-1] if seg_variants else SUBFOLDER["seg"]
        val_input = os.path.join(proj, seg_sub)

        if not os.path.isdir(val_input):
            if hasattr(self, "_val_status"):
                self._val_status.config(
                    text=f"⚠  Segmentation output not found: {seg_sub}", fg=RED)
            return

        self.config["val_main_folder"] = val_input
        save_config(self.config)
        save_project_config(self.config)

        # Disable button and start loading spinner
        if self._val_run_btn:
            self._val_run_btn.config(state="disabled", text="  Loading…")
        if hasattr(self, "_val_status"):
            self._val_status.config(text="Opening validation window…", fg=ACCENT)
        self._val_spin_frame = 0

        def _spin():
            if not hasattr(self, "_val_run_btn") or not self._val_run_btn:
                return
            try:
                frame = _SPIN[self._val_spin_frame % len(_SPIN)]
                self._val_run_btn.config(text=f"{frame}  Loading…")
                self._val_spin_frame += 1
                self._val_spin_job = self.root.after(100, _spin)
            except tk.TclError:
                pass

        def _done():
            if self._val_spin_job:
                self.root.after_cancel(self._val_spin_job)
                self._val_spin_job = None
            if self._val_run_btn:
                try:
                    self._val_run_btn.config(state="normal",
                                             text="▶  Open Validation Window")
                except tk.TclError:
                    pass
            if hasattr(self, "_val_status"):
                try:
                    self._val_status.config(text="✔  Validation window opened.", fg=GREEN)
                except tk.TclError:
                    pass

        try:
            self._track_proc(subprocess.Popen(
                [sys.executable, self._script("validator.py"), "--config", CONFIG_PATH]
            ), "Validation window")
            _spin()
            # Re-enable after 5 s — enough time for the window to appear
            self.root.after(5000, _done)
        except Exception as exc:
            _done()
            if hasattr(self, "_val_status"):
                self._val_status.config(text=f"✗  {exc}", fg=RED)

    # ==================================================================
    # STEP 4 — Validation
    # ==================================================================

    def _build_validation_step(self, parent) -> None:
        self._section_header(
            parent,
            "Step 4 — Manual Validation  (optional)",
            "Review predicted ROIs one by one and mark them YES / NO / MAYBE / ARTIFACT.",
        )

        # ── Input source selector (which segmentation folder to validate) ──
        _seg_variants = self._find_variants(SUBFOLDER["seg"])
        _default_seg  = _seg_variants[-1] if _seg_variants else SUBFOLDER["seg"]
        # Honour a segmentation-input choice previously made on the Analysis step
        # (the selection is shared between Validation and Analysis).
        _saved_seg = self.config.get("seg_input_subfolder", "").strip()
        if _saved_seg and _saved_seg in _seg_variants:
            _default_seg = _saved_seg
        self._val_input_var = tk.StringVar(value=_default_seg)
        # Persist the (possibly default) choice so Analysis picks up the same one.
        self.config["seg_input_subfolder"] = _default_seg

        toggle_card = self._card(parent, "Enable")
        tk.Checkbutton(
            toggle_card,
            text="Include manual validation step",
            variable=self._vars["val_enabled"],
            bg=CARD, fg=FG, selectcolor=CARD,
            activebackground=CARD, font=FONT_BOLD,
        ).pack(anchor="w")
        tk.Label(
            toggle_card,
            text="Opens its own window. Results are saved as TSV next to the predictions.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=640,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Separator(toggle_card, orient="horizontal").pack(fill="x", pady=(8, 6))
        tk.Label(toggle_card, text="Segmentation run to analyze:",
                 bg=CARD, fg=FG, font=FONT_BOLD).pack(anchor="w", pady=(0, 2))
        _val_input_cur_lbl = tk.Label(
            toggle_card, text=f"Using:  {_default_seg}",
            bg=CARD, fg=FG, font=FONT_MONO, wraplength=660,
        )
        _val_input_cur_lbl.pack(anchor="w", pady=(0, 4))

        if len(_seg_variants) > 1:
            _val_input_ovr_frame = tk.Frame(toggle_card, bg=CARD)
            _val_ovr_lbl = tk.Label(
                toggle_card,
                text="▶  Use a different segmentation output",
                bg=CARD, fg=MUTED,
                font=(FONT[0], FONT[1], "underline"), cursor="hand2",
            )
            _val_ovr_lbl.pack(anchor="w")

            _cb_val_src = ttk.Combobox(
                _val_input_ovr_frame, textvariable=self._val_input_var,
                values=_seg_variants, state="readonly", width=44,
            )
            _cb_val_src.pack(side="left", pady=(4, 0))
            self._bind_combobox_scroll(_cb_val_src)

            def _val_src_changed(*_):
                _val_input_cur_lbl.config(text=f"Using:  {self._val_input_var.get()}")
                # Share the choice with the Analysis step.
                self.config["seg_input_subfolder"] = self._val_input_var.get()
                self._build_val_ch_name_rows(self._detect_all_channel_labels())
            self._val_input_var.trace_add("write", _val_src_changed)

            def _toggle_val_ovr(lbl=_val_ovr_lbl):
                if _val_input_ovr_frame.winfo_ismapped():
                    _val_input_ovr_frame.pack_forget()
                    lbl.config(text="▶  Use a different segmentation output")
                else:
                    _val_input_ovr_frame.pack(fill="x", pady=(0, 4))
                    lbl.config(text="▼  Use a different segmentation output")
            _val_ovr_lbl.bind("<Button-1>", lambda _: _toggle_val_ovr())
        elif not _seg_variants:
            tk.Label(toggle_card,
                     text="⚠  No segmentation output found yet — run Step 3 first.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w")

        # ── Channel dropdown (predicted channels) ────────────────────
        # Prefer channels that actually have a predicted mask on disk so an
        # imported project works even when seg_channels is not recorded;
        # fall back to configured channels.
        cfg_card = self._card(parent, "Channel to Validate")
        channel_options = self._detect_predicted_channels()

        if channel_options:
            tk.Label(cfg_card,
                     text="Select which segmented channel to validate:",
                     bg=CARD, fg=FG, font=FONT).pack(anchor="w", pady=(0, 6))
            self._val_channel_var = tk.StringVar(
                value=f"Channel {channel_options[0]}"
            )
            _cb_val_ch = ttk.Combobox(
                cfg_card, textvariable=self._val_channel_var,
                values=[f"Channel {c}" for c in channel_options],
                state="readonly", width=20,
            )
            _cb_val_ch.pack(anchor="w")
            self._bind_combobox_scroll(_cb_val_ch)
            tk.Label(
                cfg_card,
                text="The mask glob pattern is derived automatically from the channel you select.",
                bg=CARD, fg=MUTED, font=FONT,
            ).pack(anchor="w", pady=(6, 0))
        else:
            tk.Label(
                cfg_card,
                text="⚠  No predicted channels found — run Step 3 — Segmentation, "
                     "or select a segmentation output folder above that contains predictions.",
                bg=CARD, fg=YELLOW, font=FONT, wraplength=660,
            ).pack(anchor="w")
            self._val_channel_var = None

        # ── Pre-validation ROI review ─────────────────────────────────
        review_card = self._card(parent, "Review Segmented Images")
        tk.Label(
            review_card,
            text="Before validating, you can open a napari window to manually add "
                 "any cells the model missed. Each disconnected brush stroke is "
                 "automatically treated as a separate ROI — no need to change colour "
                 "between cells. Changes are saved back to the mask TIFFs and will "
                 "appear in the validator.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=660, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        _review_btn_row = tk.Frame(review_card, bg=CARD)
        _review_btn_row.pack(anchor="w")
        self._review_btn = tk.Button(
            _review_btn_row,
            text="🔍  Review Segmented Images",
            command=self._launch_roi_editor,
            bg=ACCENT, fg="#1e1e2e", relief="flat",
            font=FONT_BOLD, padx=10, pady=6, cursor="hand2",
            activebackground="#a6c8f0", activeforeground="#1e1e2e",
        )
        self._review_btn.pack(side="left")
        self._review_status = tk.Label(
            _review_btn_row, text="", bg=CARD, fg=MUTED, font=FONT,
        )
        self._review_status.pack(side="left", padx=(10, 0))

        self._review_test_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(
            review_card,
            text="Test mode  (saves to DEBUG_ file only, processes one image, prints ROI comparison)",
            variable=self._review_test_mode,
            bg=CARD, fg=MUTED, selectcolor=CARD,
            activebackground=CARD, font=FONT,
        ).pack(anchor="w", pady=(6, 0))

        # ── Channel display names (all channels) ─────────────────────
        names_card = self._card(parent, "Channel Display Names")
        tk.Label(
            names_card,
            text="Set a display name and color for every channel in your images — "
                 "the validator shows ALL channels, not just the predicted ones.\n"
                 "Detected automatically from split_ch TIFFs in the segmentation output. "
                 "Falls back to configured channels if no output exists yet.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=660, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        self._val_ch_names_frame = tk.Frame(names_card, bg=CARD)
        self._val_ch_names_frame.pack(fill="x")
        self._val_ch_name_vars  = []
        self._val_ch_color_vars = []
        # Detect ALL channels from split_ch TIFFs in segmentation output
        _ch_labels = self._detect_all_channel_labels()
        self._build_val_ch_name_rows(_ch_labels)

        # ── Channel preview (read-only, for reference) ────────────────
        has_input = bool(self.config.get("input_root_folder", "").strip())
        prev_card = self._card(parent, "Channel Preview  (for reference)")

        val_nav_row = tk.Frame(prev_card, bg=CARD)
        val_nav_row.pack(anchor="w", pady=(0, 6))

        self._val_prev_btn = tk.Button(
            val_nav_row, text="◀", command=lambda: self._val_ch_navigate(-1),
            bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat", padx=8, pady=4,
            state="disabled",
        )
        self._val_prev_btn.pack(side="left")

        self._val_lbl = tk.Label(
            val_nav_row,
            text="Press 'Load Preview' to load channels." if has_input else "No input folder set.",
            bg=CARD, fg=MUTED, font=FONT, width=36,
        )
        self._val_lbl.pack(side="left", padx=10)

        self._val_next_btn = tk.Button(
            val_nav_row, text="▶", command=lambda: self._val_ch_navigate(1),
            bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat", padx=8, pady=4,
            state="disabled",
        )
        self._val_next_btn.pack(side="left")

        self._val_load_btn = tk.Button(
            val_nav_row, text="⟳  Load Preview",
            command=self._trigger_val_preview_load,
            bg=ACTIVE, fg=ACCENT, font=FONT_BOLD, relief="flat", padx=10, pady=4,
            activeforeground=FG,
            state="normal" if (has_input and _PIL) else "disabled",
        )
        self._val_load_btn.pack(side="left", padx=(12, 0))

        self._val_preview_img = tk.Label(prev_card, bg=CARD)
        self._val_preview_img.pack(anchor="w", pady=(0, 4))

        if not _PIL:
            tk.Label(prev_card, text="Install Pillow (pip install Pillow) to enable preview.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w")

        # ── Run button + progress (combined card) ────────────────────
        run_card = self._card(parent, "Open Validation Window")
        run_row = tk.Frame(run_card, bg=CARD)
        run_row.pack(anchor="w")
        self._val_run_btn = tk.Button(
            run_row, text="▶  Open Validation Window",
            command=self._run_validation_step,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
            padx=18, pady=10,
            state="normal" if channel_options else "disabled",
        )
        self._val_run_btn.pack(side="left")
        self._val_status = tk.Label(run_row, text="", bg=CARD, fg=MUTED, font=FONT)
        self._val_status.pack(side="left", padx=12)
        tk.Label(
            run_card,
            text="ℹ  The validation window is interactive — results save automatically as TSV "
                 "alongside the prediction files.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=660,
        ).pack(anchor="w", pady=(8, 0))

        # ── Validation progress (inside same card, below the button) ─
        ttk.Separator(run_card, orient="horizontal").pack(fill="x", pady=(14, 6))
        prog_hdr = tk.Frame(run_card, bg=CARD)
        prog_hdr.pack(fill="x")
        tk.Label(prog_hdr, text="Validation Progress",
                 bg=CARD, fg=ACCENT, font=FONT_BOLD).pack(side="left")
        tk.Button(prog_hdr, text="↻  Refresh",
                  command=self._refresh_val_status,
                  bg=BG, fg=MUTED, font=FONT, relief="flat", padx=6, pady=2).pack(
            side="left", padx=8)
        self._val_prog_frame = tk.Frame(run_card, bg=CARD)
        self._val_prog_frame.pack(anchor="w", fill="x", pady=(4, 0))
        self.root.after(120, self._refresh_val_status)

    # ==================================================================
    # STEP 5 — Analysis
    # ==================================================================

    def _build_analysis_step(self, parent) -> None:
        self._section_header(
            parent,
            "Step 5 — Analysis  (optional)",
            "Colocalization statistics between two channels using IoU-based matching.",
        )

        # ── Enable toggle ────────────────────────────────────────────
        _a_seg_variants = self._find_variants(SUBFOLDER["seg"])
        _a_default_seg = (_a_seg_variants[0] if len(_a_seg_variants) == 1 else "")
        # Honour the segmentation-input choice made on the Validation step so the
        # same folder (and its validation results) is used for analysis.
        _a_saved_seg = self.config.get("seg_input_subfolder", "").strip()
        if _a_saved_seg and _a_saved_seg in _a_seg_variants:
            _a_default_seg = _a_saved_seg
        self._ana_input_var = tk.StringVar(value=_a_default_seg)
        if _a_default_seg:
            self.config["seg_input_subfolder"] = _a_default_seg
            self._pair_roi_masks_with_run(_a_default_seg)
        else:
            self._sync_roi_mask_to_input()

        toggle_card = self._card(parent, "Enable")
        tk.Checkbutton(
            toggle_card, text="Include analysis step",
            variable=self._vars["ana_enabled"],
            bg=CARD, fg=FG, selectcolor=CARD,
            activebackground=CARD, font=FONT_BOLD,
        ).pack(anchor="w")

        ttk.Separator(toggle_card, orient="horizontal").pack(fill="x", pady=(8, 6))
        tk.Label(toggle_card, text="Segmentation Input:", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w", pady=(0, 2))
        _ana_input_cur_lbl = tk.Label(
            toggle_card, text=(f"Using:  {_a_default_seg}" if _a_default_seg
                               else "Choose a run below before starting analysis."),
            bg=CARD, fg=FG, font=FONT_MONO, wraplength=660,
        )
        _ana_input_cur_lbl.pack(anchor="w", pady=(0, 2))
        _ana_details_lbl = tk.Label(
            toggle_card,
            text=(_segmentation_run_details(
                self.config.get("project_output_folder", "").strip(),
                _a_default_seg) if _a_default_seg else
                "Choose a run to see its method and settings."),
            bg=CARD, fg=MUTED, font=FONT, wraplength=660, justify="left",
        )
        _ana_details_lbl.pack(anchor="w", pady=(0, 4))

        if len(_a_seg_variants) > 1:
            _cb_ana_src = ttk.Combobox(
                toggle_card, textvariable=self._ana_input_var,
                values=_a_seg_variants, state="readonly", width=44,
            )
            _cb_ana_src.pack(anchor="w", pady=(4, 6))
            self._bind_combobox_scroll(_cb_ana_src)

            def _ana_src_changed(*_):
                _ana_input_cur_lbl.config(text=f"Using:  {self._ana_input_var.get()}")
                _ana_details_lbl.config(text=_segmentation_run_details(
                    self.config.get("project_output_folder", "").strip(),
                    self._ana_input_var.get()))
                # Share the choice with the Validation step.
                self.config["seg_input_subfolder"] = self._ana_input_var.get()
                self._pair_roi_masks_with_run(self._ana_input_var.get())
                self._ana_roi_help_lbl.config(
                    text=self._roi_help_for_run(self._ana_input_var.get()))
            self._ana_input_var.trace_add("write", _ana_src_changed)
        elif not _a_seg_variants:
            tk.Label(toggle_card,
                     text="⚠  No segmentation output found yet — run Step 3 first.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w")

        # ── ROI mask folder ───────────────────────────────────────────
        roi_card = self._card(parent, "ROI Mask Folder")
        self._ana_roi_help_lbl = tk.Label(
            roi_card, text=self._roi_help_for_run(_a_default_seg),
            bg=CARD, fg=MUTED, font=FONT, wraplength=700, justify="left",
        )
        self._ana_roi_help_lbl.pack(anchor="w", pady=(0, 6))
        self._path_row(roi_card, "ROI mask folder (optional):",
                       "ana_roi_mask_folder")
        tk.Label(
            roi_card,
            text="For direct-read images, use matching full-size .tif or Fiji .roi "
                 "files. For converted slide scans, use the masks saved by "
                 "that converter run. Leave blank for whole-image analysis.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=700, justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # ── Validation status panel ──────────────────────────────────
        val_status_card = self._card(parent, "Validation Status")
        self._ana_val_status_frame = tk.Frame(val_status_card, bg=CARD)
        self._ana_val_status_frame.pack(anchor="w", fill="x")
        self._ana_partial_var = self._vars["apply_partial_validation"]
        self._ana_partial_check = tk.Checkbutton(
            val_status_card,
            text="Apply validation filter on partially-validated files",
            variable=self._ana_partial_var,
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        )
        tk.Button(
            val_status_card, text="↻  Refresh",
            command=self._refresh_ana_val_status,
            bg=BG, fg=MUTED, font=FONT, relief="flat", padx=6, pady=2,
        ).pack(anchor="w", pady=(6, 0))
        self.root.after(100, self._refresh_ana_val_status)

        # ── Colocalization toggle ────────────────────────────────────
        # Detect channels with a predicted mask on disk so analysis works on an
        # imported project even when seg_channels is not recorded in the file.
        channel_options = self._detect_predicted_channels()
        n_channels = len(channel_options)

        # Default colocalization to True only when ≥2 channels configured
        if "ana_coloc_enabled" not in self.config:
            self._vars["ana_coloc_enabled"].set(n_channels >= 2)
            self.config["ana_coloc_enabled"] = (n_channels >= 2)

        coloc_card = self._card(parent, "Colocalization")
        tk.Checkbutton(
            coloc_card, text="Run colocalization analysis between two channels",
            variable=self._vars["ana_coloc_enabled"],
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT_BOLD,
            command=self._toggle_ana_coloc,
        ).pack(anchor="w")
        if n_channels < 2:
            tk.Label(
                coloc_card,
                text="⚠  Only one channel configured in Step 3 — colocalization requires two channels.",
                bg=CARD, fg=YELLOW, font=FONT, wraplength=660,
            ).pack(anchor="w", pady=(4, 0))

        # ── Colocalization parameters (shown/hidden by toggle) ───────
        # Container is always packed at this position in parent; content
        # inside it is shown/hidden without losing its place in the layout.
        _coloc_container = tk.Frame(parent, bg=BG)
        _coloc_container.pack(fill="x")
        self._ana_coloc_frame = tk.Frame(_coloc_container, bg=BG)

        params_card = self._card(self._ana_coloc_frame, "Channel Selection")
        g = tk.Frame(params_card, bg=CARD)
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=240)
        g.grid_columnconfigure(1, weight=1)

        # Build channel vars from seg_channels
        current_coi = str(self.config.get("ana_red_channel", "1"))
        current_pm  = str(self.config.get("ana_pv_channel",  "3"))

        if channel_options:
            coi_default = f"Channel {current_coi}" if current_coi in channel_options \
                else f"Channel {channel_options[0]}"
            pm_default  = f"Channel {current_pm}"  if current_pm  in channel_options \
                else f"Channel {channel_options[-1] if n_channels > 1 else channel_options[0]}"
            chan_values  = [f"Channel {c}" for c in channel_options]
            state = "readonly"
        else:
            coi_default = f"Channel {current_coi}"
            pm_default  = f"Channel {current_pm}"
            chan_values  = [coi_default, pm_default]
            state = "normal"

        self._ana_coi_chan_var = tk.StringVar(value=coi_default)
        self._ana_pm_chan_var  = tk.StringVar(value=pm_default)

        tk.Label(g, text="Cell of interest channel:", bg=CARD, fg=FG, font=FONT).grid(
            row=0, column=0, sticky="w", pady=3)
        _cb_coi = ttk.Combobox(g, textvariable=self._ana_coi_chan_var,
                               values=chan_values, state=state, width=18)
        _cb_coi.grid(row=0, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_coi)

        tk.Label(g, text="Protein marker channel:", bg=CARD, fg=FG, font=FONT).grid(
            row=1, column=0, sticky="w", pady=3)
        _cb_pm = ttk.Combobox(g, textvariable=self._ana_pm_chan_var,
                              values=chan_values, state=state, width=18)
        _cb_pm.grid(row=1, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_pm)

        iou_card = self._card(self._ana_coloc_frame, "Colocalization Settings")
        g2 = tk.Frame(iou_card, bg=CARD)
        g2.pack(fill="x")
        g2.grid_columnconfigure(0, minsize=240)
        g2.grid_columnconfigure(1, weight=1)
        tk.Label(g2, text="IoU threshold (0.0 – 1.0):", bg=CARD, fg=FG, font=FONT).grid(
            row=0, column=0, sticky="w", pady=3)
        tk.Entry(g2, textvariable=self._vars["ana_iou_threshold"],
                 bg=BG, fg=FG, insertbackground=FG, relief="flat", font=FONT, width=10).grid(
            row=0, column=1, sticky="w", pady=3, padx=8)
        tk.Label(g2, text="ROI cell inclusion rule:", bg=CARD, fg=FG, font=FONT).grid(
            row=1, column=0, sticky="w", pady=3)
        _cb_rule = ttk.Combobox(g2, textvariable=self._vars["ana_roi_cell_rule"],
                                values=["centroid", "any_pixel"],
                                state="readonly", width=14)
        _cb_rule.grid(row=1, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_rule)

        # Show/hide colocalization section based on current state
        if self._vars["ana_coloc_enabled"].get():
            self._ana_coloc_frame.pack(fill="x")

        # ── Other settings ───────────────────────────────────────────
        misc_card = self._card(parent, "Settings")
        g3 = tk.Frame(misc_card, bg=CARD)
        g3.pack(fill="x")
        g3.grid_columnconfigure(0, minsize=240)
        g3.grid_columnconfigure(1, weight=1)
        tk.Label(g3, text="Analysis subfolder name:", bg=CARD, fg=FG, font=FONT).grid(
            row=0, column=0, sticky="w", pady=3)
        tk.Entry(g3, textvariable=self._vars["ana_subfolder_name"],
                 bg=BG, fg=FG, insertbackground=FG, relief="flat", font=FONT, width=30).grid(
            row=0, column=1, sticky="w", pady=3, padx=8)

        out_card = self._card(parent, "Output")
        tk.Label(
            out_card,
            text="Analysis always produces two outputs:\n"
                 "  1.  Per-image object tables (CSV)\n"
                 "  2.  Combined summary Excel — counts, avg intensity & size, "
                 "ROI stats, validation breakdown",
            bg=CARD, fg=MUTED, font=FONT, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        tk.Checkbutton(
            out_card, text="Save QC overlay images",
            variable=self._vars["ana_save_qc"],
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        ).pack(anchor="w")

        # ── Run button ───────────────────────────────────────────────
        run_card = self._card(parent, "Run Analysis")
        run_row = tk.Frame(run_card, bg=CARD)
        run_row.pack(anchor="w")
        tk.Button(
            run_row, text="▶  Run Analysis",
            command=self._run_analysis_step,
            bg=GREEN, fg="#1e1e2e", font=FONT_BOLD, relief="flat",
            padx=18, pady=10,
        ).pack(side="left")
        self._ana_status = tk.Label(run_row, text="", bg=CARD, fg=MUTED, font=FONT)
        self._ana_status.pack(side="left", padx=12)

        self._ana_progress = ttk.Progressbar(run_card, mode="indeterminate", length=500)
        self._ana_progress.pack(anchor="w", pady=(8, 4))

        self._ana_log = scrolledtext.ScrolledText(
            run_card, bg="#0d0d15", fg="#a6e3a1",
            font=FONT_MONO, height=10, relief="flat", wrap="word",
            state="disabled",
        )
        self._ana_log.pack(fill="x", pady=(4, 0))

    def _toggle_ana_coloc(self) -> None:
        """Show/hide the colocalization parameters cards."""
        try:
            if self._vars["ana_coloc_enabled"].get():
                self._ana_coloc_frame.pack(fill="x")
            else:
                self._ana_coloc_frame.pack_forget()
        except (tk.TclError, AttributeError):
            pass

    def _get_val_status_for_channel(self, project_root: str, channel: int) -> str:
        """Include focused mask-change review in the displayed status."""
        import glob as _glob
        from .validation_state import get_change_state
        change = get_change_state(project_root, channel, verify_hash=False)
        if change["status"] in ("pending", "unknown"):
            return change["status"]
        val_dir = os.path.join(project_root, "ROI Validation Results")
        if os.path.exists(os.path.join(val_dir, f"_validation_complete_ch{channel}")):
            return "complete"
        if os.path.exists(os.path.join(val_dir, "_validation_complete")):
            return "complete"
        for tsv in [
            os.path.join(val_dir, f"roi_verification_results_ch{channel}.tsv"),
            os.path.join(val_dir, "roi_verification_results.tsv"),
        ]:
            if os.path.exists(tsv):
                try:
                    import pandas as _pd
                    n = len(_pd.read_csv(tsv, sep="\t"))
                    return "partial" if n > 0 else "none"
                except Exception:
                    pass
        return "none"

    def _refresh_ana_val_status(self) -> None:
        """Rebuild the per-channel validation status labels in the analysis step."""
        if not hasattr(self, "_ana_val_status_frame"):
            return
        frame = self._ana_val_status_frame
        for w in frame.winfo_children():
            w.destroy()

        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            tk.Label(frame, text="No project folder set.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            if hasattr(self, "_ana_partial_check"):
                self._ana_partial_check.pack_forget()
            return

        # Sample folders (with their "ROI Validation Results") live inside the
        # segmentation output subfolder — NOT directly under the project root.
        # Use the same segmentation-input selection the analysis run will use.
        if getattr(self, "_ana_input_var", None) and self._ana_input_var.get():
            seg_sub = self._ana_input_var.get()
        else:
            seg_variants = self._find_variants(SUBFOLDER["seg"])
            seg_sub = seg_variants[-1] if seg_variants else SUBFOLDER["seg"]
        scan_root = os.path.join(proj, seg_sub)
        if not os.path.isdir(scan_root):
            tk.Label(frame, text="No segmentation output found yet.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            if hasattr(self, "_ana_partial_check"):
                self._ana_partial_check.pack_forget()
            return

        coi_ch = self.config.get("ana_red_channel", "")
        pm_ch  = self.config.get("ana_pv_channel",  "")
        channels_to_check = []
        for ch_val in [coi_ch, pm_ch]:
            try:
                channels_to_check.append(int(ch_val))
            except (ValueError, TypeError):
                pass

        # Collect per-channel aggregate status across all sample sub-folders
        import glob as _glob
        project_dirs = [
            d for d in sorted(os.listdir(scan_root))
            if os.path.isdir(os.path.join(scan_root, d))
            and not d.startswith(".")
            and not d.startswith("DEBUG")
        ]

        any_partial = False
        if not channels_to_check:
            tk.Label(frame, text="Set channel numbers above to see validation status.",
                     bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
        else:
            for ch in channels_to_check:
                statuses = [self._get_val_status_for_channel(os.path.join(scan_root, d), ch)
                            for d in project_dirs]
                n_total    = len(statuses)
                n_complete = statuses.count("complete")
                n_partial  = statuses.count("partial")
                n_none     = statuses.count("none")
                n_pending  = statuses.count("pending")
                n_unknown  = statuses.count("unknown")

                _img = "image" if n_total == 1 else "images"
                if n_total == 0:
                    label_text = f"Channel {ch}: no project folders found"
                    fg = MUTED
                elif n_pending or n_unknown:
                    label_text = (
                        f"Channel {ch}: REVIEW NEEDED — {n_pending} changed, "
                        f"{n_unknown} uncertain (of {n_total} {_img}); "
                        "validation filter not applied automatically"
                    )
                    fg = YELLOW
                elif n_complete == n_total:
                    label_text = (f"Channel {ch}: validation COMPLETE "
                                  f"({n_complete}/{n_total} {_img})  ✓  — filter will be applied automatically")
                    fg = GREEN
                elif n_complete + n_partial > 0:
                    label_text = (f"Channel {ch}: PARTIAL — "
                                  f"{n_complete} complete  {n_partial} in-progress  {n_none} not started "
                                  f"(of {n_total} {_img})  ⟳  — filter NOT applied by default")
                    fg = YELLOW
                    any_partial = True
                else:
                    label_text = f"Channel {ch}: not yet validated  –"
                    fg = ORANGE

                tk.Label(frame, text=label_text, bg=CARD, fg=fg, font=FONT_BOLD, justify="left").pack(anchor="w", pady=2)

        if hasattr(self, "_ana_partial_check"):
            if any_partial:
                self._ana_partial_check.pack(anchor="w", pady=(6, 0))
            else:
                self._ana_partial_check.pack_forget()

    def _refresh_val_status(self) -> None:
        """Rebuild per-project validation progress labels in the validation step."""
        if not hasattr(self, "_val_prog_frame"):
            return
        frame = self._val_prog_frame
        for w in frame.winfo_children():
            w.destroy()

        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            tk.Label(frame, text="No project folder set.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        # Show progress for ALL configured channels (not just the selected dropdown value)
        channels_to_check = [
            int(x.strip())
            for x in self.config.get("seg_channels", "").split(",")
            if x.strip().isdigit()
        ]

        if not channels_to_check:
            tk.Label(frame, text="No channels configured — set channels in Step 3.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w")
            return

        # Project sub-folders live inside the segmentation output, not the root folder.
        # Use the same segmentation-input selection that the validator was launched
        # with, so the progress reflects exactly where markers were written.
        if getattr(self, "_val_input_var", None) and self._val_input_var.get():
            seg_sub = self._val_input_var.get()
        else:
            seg_variants = self._find_variants(SUBFOLDER["seg"])
            seg_sub = seg_variants[-1] if seg_variants else SUBFOLDER["seg"]
        val_root = os.path.join(proj, seg_sub)

        if not os.path.isdir(val_root):
            tk.Label(frame, text="No segmentation output found yet.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        project_dirs = [
            d for d in sorted(os.listdir(val_root))
            if os.path.isdir(os.path.join(val_root, d))
            and not d.startswith(".") and not d.startswith("DEBUG")
        ]

        if not project_dirs:
            tk.Label(frame, text="No project folders found.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        # ── Per-project rows (one line per project, all channels on same line) ──
        for d in project_dirs:
            project_root = os.path.join(val_root, d)
            ch_parts = []
            row_status = "complete"   # assume best; downgrade as needed
            for ch in channels_to_check:
                s = self._get_val_status_for_channel(project_root, ch)
                if s == "complete":
                    icon, col = "✓", GREEN
                elif s in ("pending", "unknown"):
                    icon, col = "!", YELLOW
                    row_status = "partial"
                elif s == "partial":
                    icon, col = "⟳", YELLOW
                    if row_status == "complete":
                        row_status = "partial"
                else:
                    icon, col = "–", ORANGE
                    row_status = "none"
                ch_parts.append((f"ch{ch}: {icon}", col))

            # Build a single row frame so each channel tag can be its own colour
            row_frame = tk.Frame(frame, bg=CARD)
            row_frame.pack(anchor="w", pady=1, fill="x")

            name_fg = GREEN if row_status == "complete" else (YELLOW if row_status == "partial" else ORANGE)
            tk.Label(row_frame, text=f"{d}  ", bg=CARD, fg=name_fg,
                     font=FONT_BOLD, justify="left").pack(side="left")
            for tag_text, tag_col in ch_parts:
                tk.Label(row_frame, text=f"[{tag_text}]  ", bg=CARD, fg=tag_col,
                         font=FONT, justify="left").pack(side="left")

    @staticmethod
    def _analysis_run_status(output_dir: str) -> tuple[str, str]:
        """Show sample counts from this run while keeping older outputs readable."""
        summary_path = os.path.join(output_dir, "analysis_run_summary.json")
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            analyzed = int(summary["analyzed"])
            skipped = int(summary["skipped_missing_channel"])
            errors = int(summary["errors"])
            output_error = bool(summary.get("output_error"))
        except (OSError, ValueError, KeyError, TypeError):
            return "Analysis complete.", GREEN

        status = (
            f"Complete — {analyzed} analyzed, "
            f"{skipped} skipped (missing channel), "
            f"{errors} {'error' if errors == 1 else 'errors'}."
        )
        if output_error:
            status += " Excel output warning; see report in log."
        elif skipped or errors:
            status += " See report in log."
        color = YELLOW if skipped or errors or output_error else GREEN
        return status, color

    def _run_analysis_step(self) -> None:
        """Run analysis from the Analysis step card."""
        if self._is_running("Analysis"):
            if hasattr(self, "_ana_status"):
                self._ana_status.config(text="Analysis is already running…", fg=ACCENT)
            return
        self._sync_vars_to_config()

        # Sync channel selections from dropdowns
        if self._ana_coi_chan_var:
            ch = self._ana_coi_chan_var.get().replace("Channel ", "").strip()
            if ch.isdigit():
                self.config["ana_red_channel"] = ch
        if self._ana_pm_chan_var:
            ch = self._ana_pm_chan_var.get().replace("Channel ", "").strip()
            if ch.isdigit():
                self.config["ana_pv_channel"] = ch
        self.config["ana_coloc_enabled"] = bool(self._vars["ana_coloc_enabled"].get())

        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            if hasattr(self, "_ana_status"):
                self._ana_status.config(text="⚠  No project folder set. Go to Step 1.", fg=RED)
            return

        seg_sub = self._choose_analysis_run()
        if seg_sub is None:
            if hasattr(self, "_ana_status"):
                self._ana_status.config(text="Choose a segmentation run first.", fg=YELLOW)
            return
        if not self._pair_roi_masks_with_run(seg_sub, confirm_unknown=True):
            return
        ana_input = os.path.join(proj, seg_sub)

        if not os.path.isdir(ana_input):
            if hasattr(self, "_ana_status"):
                self._ana_status.config(
                    text=f"⚠  Segmentation output not found: {seg_sub}", fg=RED)
            return

        prepared = self._prepare_analysis_output(seg_sub)
        if prepared is None:
            return
        out, backup_dir = prepared

        self.config["ana_microsam_output_root"] = ana_input
        self.config["output_root_folder"] = out
        try:
            save_config(self.config)
            save_project_config(self.config)
        except Exception as exc:
            if backup_dir:
                try:
                    self._restore_analysis_backup(backup_dir)
                except Exception as restore_exc:
                    messagebox.showerror(
                        "Could not restore previous results",
                        f"{restore_exc}\nBackup: {backup_dir}", parent=self.root)
            messagebox.showerror("Could not start analysis",
                                 f"Could not save settings: {exc}", parent=self.root)
            return

        if hasattr(self, "_ana_status"):
            self._ana_status.config(text="Starting…", fg=ACCENT)
        if hasattr(self, "_ana_progress"):
            self._ana_progress.start(12)

        # Clear log
        if hasattr(self, "_ana_log"):
            self._ana_log.config(state="normal")
            self._ana_log.delete("1.0", "end")
            self._ana_log.config(state="disabled")

        script = self._script("analysis.py")

        def _ana_log_line(line: str) -> None:
            try:
                if not (hasattr(self, "_ana_log") and self._ana_log.winfo_exists()):
                    return
                self._ana_log.config(state="normal")
                self._ana_log.insert("end", line + "\n")
                self._ana_log.see("end")
                self._ana_log.config(state="disabled")
                if line.strip() and hasattr(self, "_ana_status") \
                        and self._ana_status.winfo_exists():
                    self._ana_status.config(text=line.strip()[:120], fg=MUTED)
            except tk.TclError:
                pass

        if backup_dir:
            _ana_log_line(f"Previous analysis saved at: {backup_dir}")

        def _thread():
            try:
                proc = subprocess.Popen(
                    [sys.executable, script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._track_proc(proc, "Analysis")
                for line in proc.stdout:
                    self.root.after(0, _ana_log_line, line.rstrip())
                proc.wait()
                ok = (proc.returncode == 0 and
                      self._analysis_workbook_written(out))
                if ok and backup_dir:
                    try:
                        self._set_analysis_backup_state(backup_dir, "committed")
                    except Exception as state_exc:
                        self.root.after(
                            0, _ana_log_line,
                            f"Backup record could not be finalized: {state_exc}. "
                            f"Backup: {backup_dir}",
                        )
                if not ok and backup_dir:
                    try:
                        self._restore_analysis_backup(backup_dir)
                        self.root.after(0, _ana_log_line,
                                        "Previous analysis results were restored.")
                    except Exception as restore_exc:
                        self.root.after(
                            0, _ana_log_line,
                            f"Could not restore previous results: {restore_exc}. "
                            f"Backup: {backup_dir}",
                        )
                def _done():
                    try:
                        if hasattr(self, "_ana_progress") and self._ana_progress.winfo_exists():
                            self._ana_progress.stop()
                        if ok:
                            status, color = self._analysis_run_status(out)
                            if hasattr(self, "_ana_status") and self._ana_status.winfo_exists():
                                self._ana_status.config(text=status, fg=color)
                        else:
                            if hasattr(self, "_ana_status") and self._ana_status.winfo_exists():
                                self._ana_status.config(
                                    text="✗  Analysis failed; see log.", fg=RED)
                            _ana_log_line("\n✗  Analysis failed or Excel output was not saved.")
                    except tk.TclError:
                        pass
                self.root.after(0, _done)
            except Exception as exc:
                failure = str(exc)
                if backup_dir:
                    try:
                        self._restore_analysis_backup(backup_dir)
                        self.root.after(0, _ana_log_line,
                                        "Previous analysis results were restored.")
                    except Exception as restore_exc:
                        self.root.after(
                            0, _ana_log_line,
                            f"Could not restore previous results: {restore_exc}. "
                            f"Backup: {backup_dir}",
                        )
                def _err():
                    try:
                        if hasattr(self, "_ana_progress") and self._ana_progress.winfo_exists():
                            self._ana_progress.stop()
                        if hasattr(self, "_ana_status") and self._ana_status.winfo_exists():
                            self._ana_status.config(text=f"✗  {failure}", fg=RED)
                    except tk.TclError:
                        pass
                self.root.after(0, _err)

        threading.Thread(target=_thread, daemon=True).start()

    # ==================================================================
    # STEP 6 — Run Pipeline
    # ==================================================================

    def _step_is_done(self, step: str) -> bool:
        """Return True if the given pipeline step has produced output files."""
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            return False
        if step == "seg":
            for v in self._find_variants(SUBFOLDER["seg"]):
                if glob.glob(os.path.join(proj, v, "**", "*_predict_*.tif"), recursive=True):
                    return True
            return False
        if step == "val":
            variants = self._find_variants(SUBFOLDER["seg"])
            selected = str(self.config.get("seg_input_subfolder", "")).strip()
            if selected not in variants and self._val_input_var is not None:
                selected = self._val_input_var.get()
            if selected not in variants:
                selected = variants[0] if len(variants) == 1 else ""
            if not selected:
                return False
            seg_folder = os.path.join(proj, selected)
            mask_pat = (self.config.get("val_mask_pattern", "") or "*_predict_original_ch2*.tif").strip()
            channel_match = re.search(r"_ch(\d+)", mask_pat)
            if channel_match:
                channels_to_check = [int(channel_match.group(1))]
            else:
                channels_to_check = [
                    int(part.strip())
                    for part in str(self.config.get("seg_channels") or "").split(",")
                    if part.strip().isdigit()
                ]
            if not channels_to_check:
                return False
            projects = [
                os.path.join(seg_folder, d)
                for d in os.listdir(seg_folder)
                if os.path.isdir(os.path.join(seg_folder, d))
                and not d.startswith((".", "DEBUG"))
                and glob.glob(os.path.join(seg_folder, d, mask_pat))
            ]
            if not projects:
                return False
            return all(
                self._get_val_status_for_channel(p, channel) == "complete"
                for p in projects for channel in channels_to_check
            )
        if step == "ana":
            variants = self._find_variants(SUBFOLDER["seg"])
            selected = (self._ana_input_var.get()
                        if self._ana_input_var is not None else "")
            if selected not in variants:
                selected = str(self.config.get("seg_input_subfolder", "")).strip()
            if selected not in variants:
                selected = variants[0] if len(variants) == 1 else ""
            if not selected:
                return False
            return os.path.isfile(os.path.join(
                self._analysis_output_dir(selected),
                "combined_analysis_summary.xlsx",
            ))
        return False

    def _build_run_step(self, parent) -> None:
        self._section_header(
            parent,
            "Step 6 — Run Pipeline",
            "Review your configuration, then launch each stage individually or run all at once.",
        )

        cfg = self.config

        # ── Configuration Summary ──────────────────────────────────────
        summary_card = self._card(parent, "Configuration Summary")
        proj  = cfg.get("project_output_folder") or "—"
        lines = [
            f"Project:         {cfg.get('project_name') or '—'}",
            f"Output folder:   {proj}",
            f"Input root:      {cfg.get('input_root_folder') or '—'}",
            f"Channels:        {cfg.get('seg_channels', '?')}",
            (f"Backend:         Cellpose-SAM  (diameter "
             f"{cfg.get('seg_cellpose_diameter') or 'auto'})  /  {cfg.get('seg_preprocess_mode', '?')}"
             if str(cfg.get('seg_backend', 'microsam')).lower() == 'cellpose'
             else f"Backend:         Micro-SAM  {cfg.get('seg_model_type', '?')}  /  {cfg.get('seg_preprocess_mode', '?')}"),
            f"Object size:     {cfg.get('seg_min_object_size')} – {cfg.get('seg_max_object_size')} px",
            f"Validation:      {'enabled' if cfg.get('val_enabled') else 'skipped'}",
            f"Analysis:        {'enabled  Ch' + str(cfg.get('ana_red_channel')) + ' ↔ Ch' + str(cfg.get('ana_pv_channel')) + '  IoU ≥ ' + str(cfg.get('ana_iou_threshold')) if cfg.get('ana_enabled') else 'skipped'}",
        ]
        txt = tk.Text(summary_card, bg=BG, fg=FG, font=FONT_MONO,
                      height=len(lines) + 1, relief="flat", state="normal", wrap="none")
        for ln in lines:
            txt.insert("end", ln + "\n")
        txt.config(state="disabled")
        txt.pack(fill="x")

        # ── Pipeline Steps ─────────────────────────────────────────────
        steps_card = self._card(parent, "Pipeline Steps")

        step_defs = [
            ("seg", "1. Segmentation",
             "Predict cell masks on all images using micro-sam.",
             ACCENT, self._run_segmentation),
            ("val", "2. Validation",
             "Manually review predictions in the interactive validation window.",
             YELLOW, self._run_validation),
            ("ana", "3. Analysis",
             "Compute colocalization statistics and generate output tables.",
             GREEN, self._run_analysis),
        ]

        first = True
        for key, label, desc, color, cmd in step_defs:
            if key == "val" and not cfg.get("val_enabled"):
                continue

            if not first:
                ttk.Separator(steps_card, orient="horizontal").pack(fill="x", pady=6)
            first = False

            row_frame = tk.Frame(steps_card, bg=CARD)
            row_frame.pack(fill="x", pady=2)

            done = self._step_is_done(key)
            sym  = "✔" if done else "✗"
            sym_color = GREEN if done else RED

            tk.Label(row_frame, text=sym, bg=CARD, fg=sym_color,
                     font=("Segoe UI", 13, "bold"), width=3).pack(side="left", padx=(0, 6))

            info_frame = tk.Frame(row_frame, bg=CARD)
            info_frame.pack(side="left", fill="x", expand=True)
            tk.Label(info_frame, text=label, bg=CARD, fg=FG,
                     font=FONT_BOLD, anchor="w").pack(anchor="w")
            tk.Label(info_frame, text=desc,  bg=CARD, fg=MUTED,
                     font=FONT, anchor="w").pack(anchor="w")

            btn_text = "▶  Open" if key == "val" else "▶  Run"
            tk.Button(
                row_frame, text=btn_text,
                command=cmd, bg=color, fg="#1e1e2e",
                font=FONT_BOLD, relief="flat", padx=12, pady=6,
            ).pack(side="right", padx=4)

        # ── Run All ────────────────────────────────────────────────────
        run_all_card = self._card(parent, "Run All")
        tk.Label(
            run_all_card,
            text="Runs segmentation if needed, then analyzes the run you choose. "
                 "If analysis results already exist, you can replace them or cancel. "
                 "Validation opens separately.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=660,
        ).pack(anchor="w", pady=(0, 8))
        if cfg.get("val_enabled"):
            tk.Label(
                run_all_card,
                text="ℹ  Validation opens its own window — use the Run button above for that step.",
                bg=CARD, fg=YELLOW, font=FONT, wraplength=660,
            ).pack(anchor="w", pady=(0, 8))
        tk.Button(
            run_all_card, text="▶▶  Run All",
            command=self._run_all,
            bg=RED, fg="white", font=FONT_BOLD, relief="flat",
            padx=18, pady=10,
        ).pack(anchor="w")

        # ── Pipeline Log ───────────────────────────────────────────────
        log_card = self._card(parent, "Pipeline Log")
        self._log_box = scrolledtext.ScrolledText(
            log_card, bg="#0d0d15", fg="#a6e3a1",
            font=FONT_MONO, height=22, relief="flat", wrap="word",
        )
        self._log_box.pack(fill="both", expand=True)
        self._run_status = tk.Label(log_card, text="Ready.", bg=CARD, fg=MUTED, font=FONT)
        self._run_status.pack(anchor="w", pady=(4, 0))

    # ==================================================================
    # Launch actions
    # ==================================================================

    def _launch_vsi(self) -> None:
        # Guard: don't allow re-launch while already running
        if self._spinning or self._is_running("VSI Converter window"):
            return

        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._vsi_warn.config(text="⚠  Set project output folder first (Step 1 — Create Project)")
            return
        self._vsi_warn.config(text="")
        self._sync_vars_to_config()

        # Auto-resolve output path into project subfolder
        vsi_out = os.path.join(proj, SUBFOLDER["vsi"])
        if os.path.isdir(vsi_out):
            dlg = _ConflictDialog(self.root, vsi_out)
            if dlg.result is None:
                return
            if dlg.result != "replace":
                vsi_out = os.path.join(proj, f"{SUBFOLDER['vsi']}_{dlg.result}")
        previous_vsi_output = self.config.get("vsi_output_dir", "").strip()
        self.config["vsi_output_dir"] = vsi_out
        save_config(self.config)

        self._spinner_start(self._vsi_spin)
        self._vsi_warn.config(text="")
        # Show log frame and clear it
        self._vsi_log_frame.pack(fill="x", pady=(4, 0))
        self._vsi_log.config(state="normal")
        self._vsi_log.delete("1.0", "end")
        self._vsi_log.config(state="disabled")

        proc = subprocess.Popen(
            [sys.executable, self._script("vsi_converter.py"), "--config", CONFIG_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout so we capture everything
            text=True, bufsize=1,
        )
        self._track_proc(proc, "VSI Converter window")

        def _watch():
            output_lines = []
            for line in proc.stdout:
                output_lines.append(line.rstrip())
                self.root.after(0, self._vsi_log_append, line.rstrip())
            proc.wait()
            ok = proc.returncode == 0

            def _done():
                self._spinner_stop(self._vsi_spin)
                if ok:
                    exports = os.path.join(vsi_out, "exports")
                    self.config["input_root_folder"] = exports
                    self._vars["input_root_folder"].set(exports)
                    self._align_converter_roi_folder(
                        exports, previous_output=previous_vsi_output)
                    save_config(self.config)
                    save_project_config(self.config)
                    try:
                        self._vsi_auto_label.config(
                            text=f"TIF exports will be read from:  {exports}", fg=FG)
                    except Exception:
                        pass
                    self._vsi_warn.config(text="✓  Converter finished successfully.", fg=GREEN)
                else:
                    # Grab the last non-empty, non-boilerplate line as the headline error
                    skip = {"traceback (most recent call last):"}
                    err_line = next(
                        (l.strip() for l in reversed(output_lines)
                         if l.strip() and l.strip().lower() not in skip),
                        "VSI converter failed — see log below."
                    )
                    try:
                        self._vsi_warn.config(text=f"✗  {err_line[:120]}", fg=RED)
                    except Exception:
                        pass
            self.root.after(0, _done)

        threading.Thread(target=_watch, daemon=True).start()

    def _run_segmentation(self) -> None:
        if self._is_running("Segmentation"):
            self._log("Segmentation is already running…")
            return
        self._sync_vars_to_config()
        out = self._resolve_segmentation_output_dir()
        if out is None:
            return
        self.config["output_root_folder"] = out
        self.config["seg_input_subfolder"] = os.path.basename(out)
        self.config["seg_resume"] = False
        save_config(self.config)
        save_project_config(self.config)
        self._run_subprocess_in_log(self._script("segmentation.py"), "Segmentation")

    def _run_validation(self) -> None:
        if self._is_running("Validation window"):
            self._log("A Validation window is already open.")
            return
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set.")
            return

        if self._val_input_var:
            seg_sub = self._val_input_var.get()
        else:
            vs = self._find_variants(SUBFOLDER["seg"])
            seg_sub = vs[-1] if vs else SUBFOLDER["seg"]
        val_input = os.path.join(proj, seg_sub)

        self._sync_vars_to_config()
        self.config["val_main_folder"] = val_input
        save_config(self.config)

        self._log(f"\n▶  Validation — opening window (input: {seg_sub})…")
        self._track_proc(subprocess.Popen(
            [sys.executable, self._script("validator.py"), "--config", CONFIG_PATH]
        ), "Validation window")

    def _run_analysis(self) -> None:
        if self._is_running("Analysis"):
            self._log("Analysis is already running…")
            return
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set.")
            return

        self._sync_vars_to_config()
        seg_sub = self._choose_analysis_run(ask_if_multiple=True)
        if seg_sub is None:
            self._log("Analysis cancelled — choose a segmentation run first.")
            return
        if not self._pair_roi_masks_with_run(seg_sub, confirm_unknown=True):
            self._log("Analysis canceled; choose the correct ROI mask folder in Step 5.")
            return
        ana_input = os.path.join(proj, seg_sub)
        if not os.path.isdir(ana_input):
            self._log(f"Segmentation output not found: {seg_sub}")
            return

        prepared = self._prepare_analysis_output(seg_sub)
        if prepared is None:
            return
        out, backup_dir = prepared

        self.config["ana_microsam_output_root"] = ana_input
        self.config["output_root_folder"] = out
        try:
            save_config(self.config)
            save_project_config(self.config)
        except Exception as exc:
            if backup_dir:
                try:
                    self._restore_analysis_backup(backup_dir)
                except Exception as restore_exc:
                    self._log(f"Could not restore previous results: "
                              f"{restore_exc}. Backup: {backup_dir}")
            self._log(f"Could not save analysis settings: {exc}")
            return
        if backup_dir:
            self._log(f"Previous analysis saved at: {backup_dir}")

        def _done(ok: bool) -> None:
            if not ok and backup_dir:
                try:
                    self._restore_analysis_backup(backup_dir)
                    self._log("Previous analysis results were restored.")
                except Exception as exc:
                    self._log(f"Could not restore previous results: {exc}. "
                              f"Backup: {backup_dir}")
            elif ok and backup_dir:
                try:
                    self._set_analysis_backup_state(backup_dir, "committed")
                except Exception as exc:
                    self._log(f"Backup record could not be finalized: {exc}. "
                              f"Backup: {backup_dir}")
            if ok:
                status, color = self._analysis_run_status(out)
                self._set_run_status(status, color)

        self._run_subprocess_in_log(
            self._script("analysis.py"), "Analysis", on_done=_done,
            success_check=lambda: self._analysis_workbook_written(out),
        )

    def _run_all(self) -> None:
        if self._is_running("Segmentation") or self._is_running("Analysis"):
            self._log("A pipeline run is already in progress…")
            return
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set. Go to Step 1 — Create Project.")
            return

        self._sync_vars_to_config()
        seg_done = self._step_is_done("seg")
        run_ana = bool(self.config.get("ana_enabled"))
        if seg_done:
            choice = _ExistingSegmentationChoiceDialog(
                self.root,
                self.config.get("input_root_folder", "").strip(),
                run_ana,
            ).result
            if choice is None:
                self._log("Run Pipeline canceled; existing results were kept.")
                return
            run_seg = choice == "new"
        else:
            run_seg = True

        seg_out = None
        if run_seg:
            self._sync_roi_mask_to_input()
            roi_folder = str(self.config.get("ana_roi_mask_folder") or "").strip()
            source_variant = None
            expected_masks = ""
            if run_ana and self.config.get("input_type") == "vsi":
                input_folder = self.config.get("input_root_folder", "").strip()
                source_variant = self._input_converter_variant(input_folder)
                expected_masks = (os.path.join(
                    proj, source_variant, "DONOTDELETE-MASK_FOR_VSI")
                    if source_variant else "")
                if (not expected_masks or os.path.normpath(roi_folder)
                        != os.path.normpath(expected_masks)):
                    if not messagebox.askyesno(
                            "Confirm ROI masks for this input",
                            f"Scanner input:\n{input_folder}\n\n"
                            f"ROI mask folder:\n{roi_folder or 'None (whole image)'}\n\n"
                            "These folders could not be paired automatically. "
                            "Use this ROI selection for the new run? "
                            "Yes continues; No cancels so you can set the "
                            "correct folder in Step 5.",
                            parent=self.root, default=messagebox.NO,
                            icon="warning"):
                        return
            has_masks = (_has_converter_roi_masks(roi_folder)
                         if source_variant and expected_masks
                         and os.path.normpath(roi_folder)
                         == os.path.normpath(expected_masks)
                         else _has_analysis_roi_masks(roi_folder)) if roi_folder else False
            if (run_ana and self.config.get("input_type") == "vsi"
                    and roi_folder and not has_masks):
                if not messagebox.askyesno(
                        "ROI masks are missing",
                        f"No ROI masks were found in:\n{roi_folder}\n\n"
                        "Continue with whole-image analysis after segmentation? "
                        "Yes continues without ROI-specific counts. "
                        "No cancels so you can export or select the masks.",
                        parent=self.root, default=messagebox.NO,
                        icon="warning"):
                    return
                self._set_roi_mask_folder("")
            seg_out = self._resolve_segmentation_output_dir(auto_new=seg_done)
            if seg_out is None:
                return
            seg_sub_for_ana = os.path.basename(seg_out)
            self.config["seg_input_subfolder"] = seg_sub_for_ana
            if self._ana_input_var is not None:
                self._ana_input_var.set(seg_sub_for_ana)
        elif run_ana:
            seg_sub_for_ana = self._choose_analysis_run(ask_if_multiple=True)
            if seg_sub_for_ana is None:
                self._log("Analysis cancelled — choose a segmentation run first.")
                return
            if not self._pair_roi_masks_with_run(
                    seg_sub_for_ana, confirm_unknown=True):
                self._log("Analysis canceled; choose the correct ROI mask folder in Step 5.")
                return
        else:
            seg_sub_for_ana = None

        ana_out = None
        backup_dir = None
        if run_ana:
            prepared = self._prepare_analysis_output(seg_sub_for_ana)
            if prepared is None:
                return
            ana_out, backup_dir = prepared

        if seg_out:
            self.config["output_root_folder"] = seg_out
            self.config["seg_resume"] = False
        try:
            save_config(self.config)
            save_project_config(self.config)
        except Exception as exc:
            if backup_dir:
                try:
                    self._restore_analysis_backup(backup_dir)
                except Exception as restore_exc:
                    self._log(f"Could not restore previous results: "
                              f"{restore_exc}. Backup: {backup_dir}")
            self._log(f"Could not save run settings: {exc}")
            return

        seg_script = self._script("segmentation.py")
        ana_script = self._script("analysis.py")

        self._log(f"\n{'═'*60}")
        self._log("▶▶  Run All")
        self._log(f"{'═'*60}")
        if backup_dir:
            self._log(f"Previous analysis saved at: {backup_dir}")
        if self.config.get("val_enabled"):
            self._log("ℹ  Validation is interactive — use the Run button on the Validation row.")
        self._set_run_status("Pipeline running…", ACCENT)

        def _thread():
            def _restore_previous() -> None:
                if not backup_dir:
                    return
                try:
                    self._restore_analysis_backup(backup_dir)
                    self.root.after(0, self._log,
                                    "Previous analysis results were restored.")
                except Exception as exc:
                    self.root.after(
                        0, self._log,
                        f"Could not restore previous results: {exc}. "
                        f"Backup: {backup_dir}",
                    )

            try:
                if run_seg:
                    proc1 = subprocess.Popen(
                        [sys.executable, seg_script, "--config", CONFIG_PATH],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    self._track_proc(proc1, "Segmentation")
                    for line in proc1.stdout:
                        self.root.after(0, self._log, line.rstrip())
                    proc1.wait()
                    if proc1.returncode != 0:
                        _restore_previous()
                        self.root.after(0, self._log,
                                        "\n✗  Segmentation failed — stopping.")
                        self.root.after(0, self._set_run_status,
                                        "Pipeline failed (segmentation).", RED)
                        return
                    self.root.after(0, self._log, "\n✓  Segmentation complete.")
                else:
                    self.root.after(0, self._log,
                                    "\n✔  Segmentation already complete — skipping.")

                if run_ana and ana_out:
                    self.config["ana_microsam_output_root"] = os.path.join(
                        proj, seg_sub_for_ana)
                    self.config["output_root_folder"] = ana_out
                    save_config(self.config)
                    save_project_config(self.config)

                    self.root.after(0, self._log, "\n" + "─"*40)
                    self.root.after(0, self._log,
                                    f"▶  Analyzing {seg_sub_for_ana}…")
                    self.root.after(0, self._set_run_status,
                                    "Analysis running…", ACCENT)

                    proc2 = subprocess.Popen(
                        [sys.executable, ana_script, "--config", CONFIG_PATH],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    self._track_proc(proc2, "Analysis")
                    for line in proc2.stdout:
                        self.root.after(0, self._log, line.rstrip())
                    proc2.wait()

                    if (proc2.returncode == 0 and
                            self._analysis_workbook_written(ana_out)):
                        if backup_dir:
                            try:
                                self._set_analysis_backup_state(
                                    backup_dir, "committed")
                            except Exception as state_exc:
                                self.root.after(
                                    0, self._log,
                                    f"Backup record could not be finalized: "
                                    f"{state_exc}. Backup: {backup_dir}",
                                )
                        status, color = self._analysis_run_status(ana_out)
                        self.root.after(0, self._set_run_status, status, color)
                    else:
                        _restore_previous()
                        self.root.after(0, self._log,
                                        "\n✗  Analysis failed or Excel output was not saved.")
                        self.root.after(0, self._set_run_status,
                                        "Pipeline failed (analysis).", RED)
                else:
                    self.root.after(0, self._set_run_status,
                                    "Pipeline complete.", GREEN)
            except Exception as exc:
                _restore_previous()
                self.root.after(0, self._log, f"\n[ERROR] Pipeline failed: {exc}")
                self.root.after(0, self._set_run_status,
                                "Pipeline failed.", RED)

        threading.Thread(target=_thread, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    CellMatchrLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
