"""
CellMatchr Launcher — step-by-step wizard UI.
"""

import json
import os
import re
import shutil
import sys
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, simpledialog

from .batch_processing import (BATCH_FOLDER, automatic_batch_masks, batch_input_folder,
                               batch_settings_issue, batch_target_problem, create_batch_folder,
                               run_batch)
from .config import (load_config, save_config, save_project_config,
                     load_project_config, CONFIG_PATH, DEFAULT_CONFIG, PROJECT_CONFIG_FILE)
from .legacy_cellpose import segmentation_command
from .microsam2 import (
    DEFAULT_MODEL as MICROSAM2_DEFAULT_MODEL, DISPLAY_NAME as MICROSAM2_DISPLAY,
    MODELS as MICROSAM2_MODELS, preprocessing_name as microsam2_preprocessing,
)
from .file_paths import filesystem_path, matching_paths, prediction_channel
from .model_wizard_archive import FOLDER as MODEL_PREVIEW_FOLDER
from .model_download import DOWNLOADS_TITLE, MICROSAM_MODELS
from .roi_mask_matching import roi_choice_context, roi_mask_files, roi_match_candidates
from .slide_signal import is_brightfield, run_is_brightfield

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
SECONDARY_FG = "#a6adc8"
ACTIVE   = "#45475a"
ORANGE   = "#fab387"   # warm peach — "not started" state (less alarming than RED)

FONT       = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_CARD_TITLE = ("Segoe UI", 12, "bold")
FONT_STEP  = ("Segoe UI", 11)
FONT_MONO  = ("Courier New", 9)

LOGO_PATH = Path(__file__).resolve().parent.parent / "CellMatchr_Logo.png"

SEGMENTATION_BACKENDS = {"microsam": "Micro-SAM", "microsam2": MICROSAM2_DISPLAY,
                         "cellpose": "Cellpose", "stardist": "StarDist 2D"}
SEGMENTATION_CHECKPOINT_KEYS = {"microsam": "seg_custom_model_path",
                                "cellpose": "seg_cellpose_model_path",
                                "stardist": "seg_stardist_model_path"}

STEPS = [
    ("1", "Create Project"),
    ("2", "Input & Paths"),
    ("3", "Segmentation"),
    ("4", "Review Segmentation"),
    ("5", "Analysis"),
    ("6", "Retrain Models"),
]
BATCH_PAGE = len(STEPS)
BATCH_TITLE = "Large Batch Processing"

# Spinner frames
_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Subfolder names written inside project_output_folder
SUBFOLDER = {
    "vsi":  "VSI_Converted_TIFF",
    "seg":  "Segmentation",
    "val":  "Validation",
    "ana":  "Analysis",
}


def _brightfield_preprocess_for_model(backend: str, variant: str,
                                      model_path: str, project: str) -> str:
    """Use a registered model's training input; invert built-in bright-signal models."""
    if model_path:
        from .retraining import PREPROCESS_MODES, list_trained_models
        for record in list_trained_models(project, backend):
            if os.path.normpath(record["model_path"]) == os.path.normpath(model_path):
                recorded = str(record.get("preprocess") or "")
                if recorded in PREPROCESS_MODES:
                    return recorded
    if backend == "cellpose" and variant == "rat_nissl_v2":
        return "raw_percentile"
    return "brightfield_invert"


def _registered_cellpose_inference(model_path: str, project: str) -> dict:
    """Return tested prediction settings attached to a saved Cellpose model."""
    if not model_path:
        return {}
    from .retraining import list_trained_models
    for record in list_trained_models(project, "cellpose"):
        if os.path.normpath(record["model_path"]) == os.path.normpath(model_path):
            settings = record.get("suggested_inference")
            return settings if isinstance(settings, dict) else {}
    return {}


def _model_wizard_options(config):
    """Starting options and additional models available in Step 3."""
    from .retraining import PREPROCESS_MODES, list_trained_models

    is_brightfield = config.get("input_type") == "vsi" and config.get("slide_signal_type") == "brightfield"
    project = config.get("project_output_folder", "")
    models = {}

    def add(label, backend, model=None, path="", record=None):
        settings = {"seg_backend": backend}
        if backend in SEGMENTATION_CHECKPOINT_KEYS:
            settings[SEGMENTATION_CHECKPOINT_KEYS[backend]] = path
            preprocess = config.get("seg_preprocess_mode") or DEFAULT_CONFIG["seg_preprocess_mode"]
            if is_brightfield:
                recorded = (record or {}).get("preprocess")
                preprocess = (recorded if recorded in PREPROCESS_MODES else
                              _brightfield_preprocess_for_model(backend, model or "cpsam", "", project))
            settings["seg_preprocess_mode"] = preprocess
        if backend == "microsam":
            settings["seg_model_type"] = model or DEFAULT_CONFIG["seg_model_type"]
        elif backend == "microsam2":
            settings["seg_microsam2_model"] = model
        elif backend == "cellpose":
            settings["seg_cellpose_variant"] = model or DEFAULT_CONFIG["seg_cellpose_variant"]
            suggested = (record or {}).get("suggested_inference")
            if not isinstance(suggested, dict):
                suggested = {}
            for name in ("diameter", "flow_threshold", "cellprob_threshold"):
                key = f"seg_cellpose_{name}"
                settings[key] = (suggested.get(name, DEFAULT_CONFIG[key]) if name != "diameter"
                                 else DEFAULT_CONFIG[key])
        unique = label
        number = 2
        while unique in models:
            unique = f"{label} ({number})"
            number += 1
        models[unique] = settings

    sam1 = DEFAULT_CONFIG["seg_model_type"]
    add(f"Micro-SAM · {sam1}", "microsam", sam1)
    add(f"{MICROSAM2_DISPLAY} · {MICROSAM2_DEFAULT_MODEL}", "microsam2", MICROSAM2_DEFAULT_MODEL)
    add("Cellpose-SAM", "cellpose", "cpsam")
    add("StarDist · 2D versatile fluorescence", "stardist")
    defaults = tuple(models)
    for model in MICROSAM_MODELS:
        if model != sam1:
            add(f"Micro-SAM · {model}", "microsam", model)
    for model in MICROSAM2_MODELS:
        if model != MICROSAM2_DEFAULT_MODEL:
            add(f"{MICROSAM2_DISPLAY} · {model}", "microsam2", model)
    add("Cellpose 2 · cyto2", "cellpose", "cyto2")
    add("Cellpose · Rat Nissl v2", "cellpose", "rat_nissl_v2")
    registered_paths = set()
    for record in list_trained_models(project):
        backend = record["backend"]
        path = record["model_path"]
        registered_paths.add(os.path.normcase(os.path.abspath(path)))
        add(f"{record['name']} · {SEGMENTATION_BACKENDS[backend]}", backend, path=path, record=record)
    for backend, key in SEGMENTATION_CHECKPOINT_KEYS.items():
        path = str(config.get(key) or "").strip()
        if path and os.path.exists(path) and os.path.normcase(os.path.abspath(path)) not in registered_paths:
            add(f"{os.path.basename(path)} · {SEGMENTATION_BACKENDS[backend]}", backend, path=path)
    return models, defaults


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
    """Accept the TIFF and ImageJ masks supported by analysis."""
    try:
        return bool(roi_mask_files(folder))
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
        method = {"microsam": "Micro-SAM", "microsam2": MICROSAM2_DISPLAY,
                  "cellpose": "Cellpose",
                  "stardist": "StarDist 2D"}.get(
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


class _ROIMaskChoiceDialog(tk.Toplevel):
    def __init__(self, parent, sample, filenames, previous=None):
        super().__init__(parent)
        self.result = None
        self.title("Choose an ROI mask")
        self.transient(parent)
        self.configure(bg=BG)
        self.resizable(False, False)
        tk.Label(self, text=f"Image: {sample}", wraplength=570, justify="left",
                 bg=BG, fg=FG, font=FONT_BOLD).pack(anchor="w", padx=20, pady=(18, 6))
        tk.Label(self, text="Confirm the matching mask, or leave ROI counts blank.",
                 bg=BG, fg=FG, font=FONT).pack(anchor="w", padx=20)
        whole_image = "Whole image only"
        selection = ttk.Combobox(self, values=(whole_image, *filenames),
                                 width=75, state="readonly")
        selection.current(filenames.index(previous) + 1 if previous in filenames else 0)
        selection.pack(padx=20, pady=12)

        def accept():
            self.result = "" if selection.current() == 0 else selection.get()
            self.destroy()

        buttons = tk.Frame(self, bg=BG)
        buttons.pack(padx=20, pady=(0, 16))
        ttk.Button(buttons, text="Use selection", command=accept).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel analysis", command=self.destroy).pack(side="left", padx=4)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.grab_set()
        selection.focus_set()
        self.wait_window(self)


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------

class _ChannelPreview(tk.Frame):
    """Shared channel preview controls for Segmentation and Review Segmentation."""

    _LOAD_TEXT = "⟳  Load Preview"

    def __init__(self, parent, *, has_input, navigate, load, select=None):
        super().__init__(parent, bg=CARD)
        self.request = None
        row = tk.Frame(self, bg=CARD)
        row.pack(anchor="w", pady=(0, 6))
        button_style = dict(bg=ACTIVE, font=FONT_BOLD, relief="flat", padx=8, pady=4)
        if sys.platform != "darwin":
            button_style.update(fg=FG, activebackground=ACTIVE, activeforeground=FG)
        self.previous = tk.Button(row, text="◀", command=lambda: navigate(-1), **button_style)
        self.previous.pack(side="left")
        self.status = tk.Label(row, bg=CARD, font=FONT, width=28, wraplength=200)
        self.status.pack(side="left", padx=10)
        self.next = tk.Button(row, text="▶", command=lambda: navigate(1), **button_style)
        self.next.pack(side="left")
        self.load = tk.Button(row, command=load, **button_style)
        self.load.pack(side="left", padx=(12, 0))
        self.preview_image = tk.Label(self, bg=CARD)
        self.preview_image.pack(anchor="w", pady=(0, 4))
        if select is not None:
            self.preview_image.config(cursor="hand2")
            self.preview_image.bind("<Button-1>", lambda _: select())
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=380)
        self.reset(has_input)

    def reset(self, has_input):
        self.finish_load(enabled=has_input and _PIL)
        if not _PIL:
            message = "Install Pillow to preview channels."
        else:
            message = "No preview" if has_input else "No input images"
        self.show(message, color=SECONDARY_FG)

    def start_load(self):
        self.request = object()
        self.load.config(state="disabled", text="Loading…")
        self.status.config(text="Loading…", fg=ACCENT)
        self.progress.pack(anchor="w", pady=(0, 6))
        self.progress.start(12)
        return self.request

    def finish_load(self, *, enabled=True):
        self.request = None
        self.load.config(text=self._LOAD_TEXT, state="normal" if enabled else "disabled")
        self.progress.stop()
        self.progress.pack_forget()

    def show(self, text, image="", *, color=FG, previous=False, next_channel=False):
        self.status.config(text=text, fg=color)
        self.preview_image.config(image=image)
        self.previous.config(state="normal" if previous else "disabled")
        self.next.config(state="normal" if next_channel else "disabled")

    def show_channel(self, photos, index, *, indicator="", color=FG):
        last = len(photos) - 1
        label = f"Channel {index}" if last == 0 else f"Channel {index} of {last}"
        self.show(label + indicator, photos[index], color=color, previous=index > 0,
                  next_channel=index < last)


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
        self._batch_return_step = 0
        self._spinning = False
        self._spin_idx  = 0
        # Live background processes (segmentation/analysis runs, validation
        # windows). Used to warn before navigating away mid-run.
        self._active_procs: list = []
        self._batch_stop = None
        self._batch_lock = threading.Lock()
        self._batch_project = None
        self._batch_targets = {}

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
        self._val_ch_index: int = 0
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
        self._loaded_project_file = ""
        self._project_mode = "new"

        self._vars: dict = {}
        self._build_vars()
        self._build_layout()
        self._open_json_var.trace_add("write", self._on_project_file_changed)
        self._show_step(0)

    # ------------------------------------------------------------------
    # Variable binding
    # ------------------------------------------------------------------

    def _build_vars(self) -> None:
        str_fields = [
            "project_name", "project_output_folder",
            "input_type", "input_root_folder", "slide_signal_type",
            "seg_channels", "seg_backend", "seg_model_type", "seg_preprocess_mode",
            "seg_microsam2_model",
            "seg_custom_model_path", "seg_cellpose_model_path", "seg_cellpose_variant",
            "seg_stardist_model_path",
            "val_mask_pattern", "val_channel_names", "val_channel_colors",
            "ana_red_channel", "ana_pv_channel", "ana_iou_threshold",
            "ana_roi_cell_rule", "ana_roi_mask_folder", "ana_subfolder_name",
        ]
        num_fields = [
            "seg_min_object_size", "seg_max_object_size",
            "seg_tile_h", "seg_tile_w", "seg_halo_h", "seg_halo_w",
            "seg_qc_downsample", "seg_max_ratio", "seg_raw_mean_min",
            "seg_raw_mean_max",
            "seg_cellpose_diameter", "seg_cellpose_flow_threshold",
            "seg_cellpose_cellprob_threshold",
        ]
        bool_fields = [
            "seg_test_run", "seg_dry_run", "seg_skip_subdirs", "seg_resume",
            "ana_enabled", "ana_coloc_enabled", "ana_save_qc",
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
        if not (_PIL and LOGO_PATH.is_file()):
            return
        try:
            with Image.open(LOGO_PATH) as img:
                if (sys.platform.startswith("linux")
                        and self.root.tk.call("tk", "windowingsystem") == "x11"):
                    # X11 can reject the full-size logo as an oversized window icon.
                    img.thumbnail((128, 128), Image.LANCZOS)
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
        if _PIL and LOGO_PATH.is_file():
            with Image.open(LOGO_PATH) as img:
                img.thumbnail((120, 120), Image.LANCZOS)
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
        tk.Label(self.sidebar, text="Microscopy image analysis",
                 bg=SIDEBAR, fg=SECONDARY_FG, font=("Segoe UI", 8)).pack(pady=(0, 16))

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
        batch_button = tk.Button(
            self.sidebar, text=f"  {BATCH_TITLE}", bg=SIDEBAR, fg=MUTED, font=FONT_STEP,
            anchor="w", relief="flat", bd=0, activebackground=ACTIVE, activeforeground=FG,
            command=lambda: self._show_step(BATCH_PAGE),
        )
        batch_button.pack(side="bottom", fill="x", padx=8, ipady=7, pady=(0, 10))
        self._step_btns.append(batch_button)

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
        widget = event.widget
        if not hasattr(widget, "winfo_toplevel") or widget.winfo_toplevel() is not self.root:
            return
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
        for entry in list(self._active_procs):
            try:
                if entry[0].poll() is None:
                    live.append(entry)
                    continue
            except Exception:
                pass
            # A batch worker may register the next subprocess during this scan.
            try:
                self._active_procs.remove(entry)
            except ValueError:
                pass
        return live

    def _is_running(self, label: str) -> bool:
        """True if a tracked process with this label is still alive."""
        return any(lbl == label for _, lbl in self._running_procs())

    def _terminate_running(self) -> None:
        """Terminate (then kill) every tracked process that is still alive."""
        with self._batch_lock:
            if self._batch_stop is not None:
                self._batch_stop.set()
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
        active = {label for _, label in running}
        if self._batch_stop is not None:
            active.add(BATCH_TITLE)
        if not active:
            return True
        labels = "\n      • ".join(sorted(active))
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
        wizard = getattr(self, "_model_wizard", None)
        if wizard is not None and wizard.window.winfo_exists():
            wizard.close()
            return
        running = self._running_procs()
        active = {label for _, label in running}
        if self._batch_stop is not None:
            active.add(BATCH_TITLE)
        if active:
            labels = "\n      • ".join(sorted(active))
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
        self._finish_close()

    def _finish_close(self) -> None:
        self._sync_vars_to_config()
        save_config(self.config)
        project = self.config.get("project_output_folder", "").strip()
        if project and os.path.isfile(os.path.join(project, PROJECT_CONFIG_FILE)):
            save_project_config(self.config)
        cached = getattr(self, "_model_preview_cache", None)
        if cached:
            cached["directory"].cleanup()
        self.root.quit()

    def _show_step(self, idx: int) -> None:
        if not self._can_leave_project_step() and idx != 0:
            return
        # Guard against navigating away while a run/window is in progress.
        if idx != self.current_step and not self._confirm_leave_running():
            return
        self._sync_vars_to_config()
        save_config(self.config)
        save_project_config(self.config)
        if idx == BATCH_PAGE and self.current_step != BATCH_PAGE:
            self._batch_return_step = self.current_step
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
            self._build_retraining_step,
            self._build_batch_step,
        ]
        builders[idx](self.content_frame)

        self.btn_back.config(state="normal" if idx > 0 else "disabled")
        if idx == BATCH_PAGE:
            self.btn_next.pack_forget()
        else:
            self.btn_next.pack(side="right", padx=14, pady=10)
            if idx == len(STEPS) - 1:
                self.btn_next.config(text="Save & Close", command=self._save_and_close)
            else:
                self.btn_next.config(text="Next  ▶", command=self._next)
        self._update_project_navigation()

        self._canvas.yview_moveto(0)
        self._update_project_display()

    def _next(self):
        if self.current_step < len(STEPS) - 1:
            self._show_step(self.current_step + 1)

    def _back(self):
        if self.current_step == BATCH_PAGE:
            self._show_step(self._batch_return_step)
        elif self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _save_and_close(self):
        if not self._confirm_leave_running():
            return
        self._finish_close()

    def _update_project_display(self) -> None:
        name   = self.config.get("project_name", "").strip() or "No project set"
        folder = self.config.get("project_output_folder", "").strip()
        short  = os.path.basename(folder) if folder else "—"
        self._proj_name_lbl.config(text=name)
        self._proj_folder_lbl.config(text=short)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _section_header(self, parent, step: int | str, subtitle: str, *, optional=False) -> None:
        if isinstance(step, int):
            number, name = STEPS[step - 1]
            title = f"Step {number} — {name}"
        else:
            title = step
        if optional:
            title += " (optional)"
        tk.Label(parent, text=title, bg=BG, fg=FG,
                 font=FONT_TITLE, anchor="w").pack(
            fill="x", padx=32, pady=(32, 4)
        )
        if subtitle:
            tk.Label(parent, text=subtitle, bg=BG, fg=SECONDARY_FG,
                     font=FONT, anchor="w", wraplength=700).pack(
                fill="x", padx=32, pady=(0, 12)
            )
        ttk.Separator(parent).pack(fill="x", padx=32, pady=(0, 20))

    def _card(self, parent, title: str = "") -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD, padx=20, pady=14)
        outer.pack(fill="x", padx=32, pady=6)
        if title:
            tk.Label(outer, text=title, bg=CARD, fg=ACCENT,
                     font=FONT_CARD_TITLE).pack(anchor="w", pady=(0, 8))
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
        if backend not in ("microsam", "microsam2", "cellpose", "stardist"):
            backend = "microsam"
        if backend == "cellpose":
            variant = str(self.config.get("seg_cellpose_variant") or "cpsam").strip()
            if variant != "cpsam":
                backend = f"cellpose_{variant}"
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
        brightfield = (self.config.get("input_type") == "vsi" and
                       self.config.get("slide_signal_type") == "brightfield")
        model = (str(self.config.get("seg_microsam2_model") or MICROSAM2_DEFAULT_MODEL)
                 if backend == "microsam2" else
                 str(self.config.get("seg_model_type") or "vit_b_lm") if backend == "microsam" else
                 str(self.config.get("seg_cellpose_variant") or "cpsam")
                 if backend == "cellpose" else "2D_versatile_fluo")
        raw_channels = str(self.config.get("seg_channels") or "")
        channels = ([0] if brightfield else
                    [int(part.strip()) for part in raw_channels.split(",") if part.strip()])
        identity = {
            "backend": backend,
            "model": model,
            "preprocess": (microsam2_preprocessing(brightfield) if backend == "microsam2"
                           else str(self.config.get("seg_preprocess_mode") or "local_bg_subtract")),
            "channels": channels,
        }
        if brightfield:
            identity["slide_signal_type"] = "brightfield"
        key = {"microsam": "seg_custom_model_path",
               "cellpose": "seg_cellpose_model_path",
               "stardist": "seg_stardist_model_path"}.get(backend)
        chosen = str(self.config.get(key) or "").strip() if key else ""
        if chosen:
            identity["custom_model"] = os.path.abspath(chosen)
        return identity

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

    def _confirm_legacy_roi_matches(self, run_folder: str, *, review=False) -> bool:
        folder = str(self.config.get("ana_roi_mask_folder") or "").strip()
        if not folder:
            if review:
                messagebox.showinfo("ROI mask folder", "Choose an ROI mask folder first.",
                                    parent=self.root)
            return True
        try:
            masks = roi_mask_files(folder)
            if not masks:
                if review:
                    messagebox.showinfo("ROI mask folder", "No TIFF or Fiji ROI masks were found.",
                                        parent=self.root)
                return True
            with os.scandir(filesystem_path(run_folder)) as entries:
                samples = sorted(entry.name for entry in entries if entry.is_dir()
                                 and not entry.name.startswith((".", "DEBUG")))
        except OSError as exc:
            messagebox.showerror("Could not check ROI masks", str(exc), parent=self.root)
            return False
        context = roi_choice_context(run_folder, folder)
        saved = self.config.get("ana_roi_mask_choices")
        previous = (saved.get("samples", {}) if isinstance(saved, dict)
                    and saved.get("context") == context else {})
        if not isinstance(previous, dict):
            previous = {}
        choices = {}
        for sample in samples:
            _, candidates = roi_match_candidates(sample, masks)
            if not candidates:
                continue
            names = [os.path.basename(path) for path in candidates]
            saved_choice = previous.get(sample)
            if not review and (saved_choice == "" or saved_choice in names):
                choices[sample] = saved_choice
                continue
            dialog = _ROIMaskChoiceDialog(self.root, sample, names, saved_choice)
            if dialog.result is None:
                return False
            choices[sample] = dialog.result
        self.config["ana_roi_mask_choices"] = {"context": context, "samples": choices}
        if review and not choices:
            messagebox.showinfo("ROI mask matches", "There are no mask choices to confirm. "
                                "Samples with matching names are paired automatically.",
                                parent=self.root)
        return True

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
            return "Add ROI masks if you want measurements within regions."
        kind, converter_run = self._segmentation_source(segmentation_run)
        if kind == "converted":
            return f"CellMatchR selects matching ROI masks from {converter_run} by default."
        if kind == "direct":
            return "Leave blank for whole-image measurements, or choose matching TIFF or Fiji .roi masks."
        return "For this earlier run, check that your ROI masks match the selected images."

    def _analysis_generated_files(self, segmentation_run: str,
                                  output_dir: str) -> list[str]:
        """List only established analysis files for the selected run."""
        files = []
        for filename in (
            "combined_analysis_summary.xlsx", "combined_analysis_summary.csv",
            "analysis_run_summary.json", "analysis_errors.log",
        ):
            path = os.path.join(output_dir, filename)
            if os.path.isfile(filesystem_path(path)):
                files.append(path)

        proj = self.config.get("project_output_folder", "").strip()
        seg_root = os.path.join(proj, segmentation_run)
        configured = str(self.config.get("ana_subfolder_name") or
                         "analysis_results").strip()
        subfolders = {configured, "analysis_results", "colocalization_analysis"}
        if not os.path.isdir(filesystem_path(seg_root)):
            return files
        for sample in os.listdir(filesystem_path(seg_root)):
            if sample.startswith((".", "DEBUG")):
                continue
            sample_dir = os.path.join(seg_root, sample)
            if not os.path.isdir(filesystem_path(sample_dir)):
                continue
            output_pattern = re.compile(
                rf"^{re.escape(sample)}_(?:summary\.csv|"
                r"(?:channel_\d+|cell_of_interest|protein_marker)_object_table\.csv|"
                r"channel_\d+_to_channel_\d+_(?:matches|iou_candidates)\.csv|"
                r"roi_mask_resized\.tif|QC_(?:overlay|roi_preview)\.(?:tif|png))$"
            )
            for subfolder in subfolders:
                results_dir = os.path.join(sample_dir, subfolder)
                if not os.path.isdir(filesystem_path(results_dir)):
                    continue
                for filename in os.listdir(filesystem_path(results_dir)):
                    path = os.path.join(results_dir, filename)
                    if output_pattern.fullmatch(filename) and os.path.isfile(filesystem_path(path)):
                        files.append(path)
        return sorted(set(files))

    @staticmethod
    def _analysis_realpath(path: str) -> str:
        """Resolve the actual target while keeping paths comparable on Windows."""
        resolved = os.path.realpath(filesystem_path(path))
        if os.name == "nt":
            if resolved.startswith("\\\\?\\UNC\\"):
                resolved = "\\\\" + resolved[8:]
            elif resolved.startswith("\\\\?\\"):
                resolved = resolved[4:]
        return os.path.normcase(resolved)

    @staticmethod
    def _write_analysis_backup_manifest(backup_dir: str, manifest: dict) -> None:
        """Persist recovery instructions before moving any result file."""
        path = os.path.join(backup_dir, "backup_manifest.json")
        pending = path + ".tmp"
        with open(filesystem_path(pending), "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(filesystem_path(pending), filesystem_path(path))

    def _set_analysis_backup_state(self, backup_dir: str, state: str) -> None:
        with open(filesystem_path(os.path.join(backup_dir, "backup_manifest.json"))) as f:
            manifest = json.load(f)
        manifest["state"] = state
        self._write_analysis_backup_manifest(backup_dir, manifest)

    def _recover_analysis_backups(self, segmentation_run: str) -> int:
        """Finish or roll back an interrupted replacement for this run."""
        proj = os.path.abspath(
            self.config.get("project_output_folder", "").strip())
        backup_root = os.path.join(proj, ".analysis_backups")
        if not os.path.isdir(filesystem_path(backup_root)):
            return 0
        proj_real = self._analysis_realpath(proj)
        if os.path.commonpath((proj_real, self._analysis_realpath(backup_root))) != proj_real:
            raise ValueError(f"Backup folder points outside the project: {backup_root}")
        recovered = 0
        for name in sorted(os.listdir(filesystem_path(backup_root))):
            backup_dir = os.path.join(backup_root, name)
            manifest_path = os.path.join(backup_dir, "backup_manifest.json")
            if not os.path.isfile(filesystem_path(manifest_path)):
                continue
            with open(filesystem_path(manifest_path)) as f:
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
        if not os.path.isdir(filesystem_path(backup_root)):
            return 0
        proj_real = self._analysis_realpath(proj)
        if os.path.commonpath((proj_real, self._analysis_realpath(backup_root))) != proj_real:
            raise ValueError(f"Backup folder points outside the project: {backup_root}")
        runs = set()
        for name in os.listdir(filesystem_path(backup_root)):
            manifest_path = os.path.join(backup_root, name,
                                         "backup_manifest.json")
            if not os.path.isfile(filesystem_path(manifest_path)):
                continue
            with open(filesystem_path(manifest_path)) as f:
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
        proj_real = self._analysis_realpath(proj)
        backup_dir = os.path.join(
            proj, ".analysis_backups",
            f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}",
        )
        moved = []
        try:
            originals = [os.path.abspath(path) for path in existing]
            for original in originals:
                if os.path.commonpath((proj_real, self._analysis_realpath(original))) != proj_real:
                    raise ValueError(
                        f"Analysis output is outside the project: {original}")
            if os.path.commonpath((
                    proj_real, self._analysis_realpath(os.path.dirname(backup_dir)))) != proj_real:
                raise ValueError("Analysis backup folder points outside the project")
            os.makedirs(filesystem_path(backup_dir), exist_ok=False)
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
                os.makedirs(filesystem_path(os.path.dirname(saved)), exist_ok=True)
                os.replace(filesystem_path(original), filesystem_path(saved))
                moved.append((original, saved))
            manifest["state"] = "ready"
            self._write_analysis_backup_manifest(backup_dir, manifest)
        except Exception as exc:
            for original, saved in reversed(moved):
                try:
                    os.makedirs(filesystem_path(os.path.dirname(original)), exist_ok=True)
                    os.replace(filesystem_path(saved), filesystem_path(original))
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
        proj_real = self._analysis_realpath(proj)
        with open(filesystem_path(os.path.join(backup_dir, "backup_manifest.json"))) as f:
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
            if os.path.commonpath((proj_real, self._analysis_realpath(generated))) != proj_real:
                raise ValueError(f"Analysis output is outside the project: {generated}")
            if generated not in original_paths:
                os.remove(filesystem_path(generated))
        for relative in manifest["original_files"]:
            original = os.path.join(proj, relative)
            saved = os.path.join(backup_dir, relative)
            if os.path.commonpath((
                    proj_real, self._analysis_realpath(os.path.dirname(original)))) != proj_real:
                raise ValueError(f"Cannot restore outside the project: {original}")
            if os.path.isfile(filesystem_path(saved)):
                os.makedirs(filesystem_path(os.path.dirname(original)), exist_ok=True)
                os.replace(filesystem_path(saved), filesystem_path(original))
            elif not os.path.isfile(filesystem_path(original)):
                raise FileNotFoundError(
                    f"Neither original nor backup is available: {original}")
        manifest["state"] = "restored"
        self._write_analysis_backup_manifest(backup_dir, manifest)
        return backup_dir

    @staticmethod
    def _analysis_workbook_written(output_dir: str) -> bool:
        """A zero exit alone does not mean the new workbook was saved."""
        try:
            with open(filesystem_path(os.path.join(output_dir, "analysis_run_summary.json"))) as f:
                written = bool(json.load(f)["workbook_written"])
            return written and os.path.isfile(filesystem_path(os.path.join(
                output_dir, "combined_analysis_summary.xlsx")))
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
                                on_done=None, success_check=None, extra_args=()) -> None:
        self._log(f"\n{'─'*60}")
        self._log(f"▶  {label}")
        self._log(f"{'─'*60}")
        self._set_run_status(f"{label} running…", ACCENT)

        def _thread():
            try:
                if (os.path.basename(script) == "segmentation.py"
                        and self.config.get("seg_backend") == "cellpose"
                        and self.config.get("seg_cellpose_variant", "cpsam") != "cpsam"):
                    self.root.after(0, self._log,
                                    "Preparing the selected Cellpose model; first use "
                                    "downloads its weights and separate runtime…")
                command, environment = segmentation_command(script, CONFIG_PATH, self.config)
                proc = subprocess.Popen(
                    [*command, *extra_args], env=environment,
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
        self._project_mode = "new"
        self._section_header(
            parent, 1,
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
                 text="Select your project's cellmatchr_project.json, then click Load Project Settings.",
                 bg=CARD, fg=MUTED, font=FONT, wraplength=660).pack(anchor="w", pady=(0, 10))

        # File-picker row for the JSON file
        _json_row = tk.Frame(self._pane_open, bg=CARD)
        _json_row.pack(fill="x", pady=(0, 4))
        tk.Label(_json_row, text="Project file (.json):", bg=CARD, fg=FG,
                 font=FONT, width=26, anchor="w").pack(side="left")
        project_entry = tk.Entry(_json_row, textvariable=self._open_json_var,
                                 bg="#3c3c4e", fg=FG, insertbackground=FG,
                                 relief="flat", font=FONT)
        project_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
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
        self._project_mode = mode
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
        self._update_project_navigation()

    def _can_leave_project_step(self) -> bool:
        if self.current_step != 0 or self._project_mode != "open":
            return True
        path = self._open_json_var.get().strip()
        return bool(path and os.path.abspath(path) == self._loaded_project_file
                    and os.path.isfile(path))

    def _update_project_navigation(self) -> None:
        state = "normal" if self._can_leave_project_step() else "disabled"
        self.btn_next.config(state=state)
        for button in self._step_btns[1:]:
            button.config(state=state)

    def _on_project_file_changed(self, *_args) -> None:
        if self.current_step == 0 and self._project_mode == "open":
            waiting = self._open_json_var.get().strip() and not self._can_leave_project_step()
            self._open_status.config(text="Click Load Project Settings to continue." if waiting else "",
                                     fg=MUTED)
            self._update_project_navigation()

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

    def _open_project(self) -> bool:
        json_path = self._open_json_var.get().strip()
        self._loaded_project_file = ""
        if os.path.basename(json_path) != PROJECT_CONFIG_FILE or not os.path.isfile(json_path):
            self._open_status.config(
                text="⚠  Select a cellmatchr_project.json file first.", fg=RED)
            self._update_project_navigation()
            return False
        folder = os.path.dirname(os.path.abspath(json_path))
        cfg = load_project_config(folder)
        if cfg is None:
            self._open_status.config(
                text=f"✗  Could not read project file:\n   {json_path}",
                fg=RED)
            self._update_project_navigation()
            return False
        # Load config into memory and re-sync all vars
        self.config = cfg
        self._loaded_project_file = os.path.abspath(json_path)
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
        self.config["input_root_folder"] = self._resolve_existing_input_folder()
        if self.config.get("input_type") == "vsi":
            self._align_converter_roi_folder(self.config["input_root_folder"])
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
        self._update_project_navigation()
        return True

    # ==================================================================
    # STEP 2 — Input & Paths
    # ==================================================================

    def _build_input_step(self, parent) -> None:
        self._section_header(
            parent, 2,
            "Select your data source and convert slide scans to TIFF if needed.",
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
            (".tif / .tiff", "TIFF or BigTIFF slide", "tested",
             "draw regions on a large TIFF or a previously converted scan"),
        ]:
            self._format_status_row(self._fmt_slide_frame, ext, brand, status, note)

        # ── VSI section (shown when input_type == "vsi") ─────────────
        self._vsi_section = tk.Frame(parent, bg=BG)

        signal_card = self._card(self._vsi_section, "Slide Scanner Signal")
        tk.Label(signal_card,
                 text="What kind of signal is on this slide?",
                 bg=CARD, fg=FG, font=FONT_BOLD).pack(anchor="w", pady=(0, 5))
        for value, label in [
            ("fluorescence", "Laser fluorescence — bright labeled cells"),
            ("brightfield", "Brightfield stain — dark cells, such as Nissl"),
        ]:
            tk.Radiobutton(
                signal_card, text=label, variable=self._vars["slide_signal_type"],
                value=value, command=self._on_slide_signal_choice,
                bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD,
                font=FONT,
            ).pack(anchor="w", pady=2)
        tk.Label(signal_card,
                 text="Brightfield keeps the visible stain image unchanged. "
                      "Model input and the Parameter Wizard adapt automatically.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 9),
                 wraplength=700, justify="left").pack(anchor="w", pady=(4, 0))

        vsi_card = self._card(self._vsi_section, "Slide Scanner Converter  (VSI · LIF · TIFF)")
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
        count = self._folder_image_count(folder)
        if count:
            text = f"{count} saved TIFF crop(s) found in:\n{folder}"
            masks = self._saved_roi_mask_text()
            return f"{text}\n{masks}" if masks else text
        if folder:
            return f"No TIFF crops found in:\n{folder}"
        return "TIF input folder will be set automatically after the converter closes."

    def _saved_roi_mask_text(self) -> str:
        folder = self.config.get("ana_roi_mask_folder", "").strip()
        return f"Saved ROI masks: {folder}" if folder and _has_analysis_roi_masks(folder) else ""

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

    def _on_slide_signal_choice(self) -> None:
        """Apply a reversible starting profile when the user changes slide signal."""
        choice = self._vars["slide_signal_type"].get()
        previous = self.config.get("slide_signal_type", "fluorescence")
        if choice == previous:
            return
        profile_keys = (
            "seg_channels", "seg_backend", "seg_model_type", "seg_preprocess_mode",
            "seg_microsam2_model",
            "seg_custom_model_path", "seg_cellpose_model_path",
            "seg_cellpose_variant", "seg_stardist_model_path",
            "seg_min_object_size", "seg_max_object_size", "seg_max_ratio",
            "seg_raw_mean_min", "seg_raw_mean_max",
        )
        if choice == "brightfield":
            self.config["slide_fluorescence_settings"] = {
                key: self._vars[key].get() for key in profile_keys
            }
            self.config["slide_fluorescence_settings"]["per_channel_enabled"] = bool(
                self.config.get("seg_per_channel_filters_enabled", False))
            self.config["slide_fluorescence_settings"]["per_channel_filters"] = str(
                self.config.get("seg_per_channel_filters", "{}"))
            self._vars["seg_channels"].set("0")
            backend = self._vars["seg_backend"].get()
            model_key = {"microsam": "seg_custom_model_path",
                         "cellpose": "seg_cellpose_model_path",
                         "stardist": "seg_stardist_model_path"}.get(backend, "")
            model_path = self._vars[model_key].get().strip() if model_key else ""
            mode = _brightfield_preprocess_for_model(
                backend, self._vars["seg_cellpose_variant"].get(), model_path,
                self.config.get("project_output_folder", ""))
            if backend != "microsam2":
                self._vars["seg_preprocess_mode"].set(mode)
            for key in ("seg_min_object_size", "seg_max_object_size",
                        "seg_max_ratio", "seg_raw_mean_min", "seg_raw_mean_max"):
                self._vars[key].set("0")
            self.config["seg_per_channel_filters_enabled"] = False
        else:
            saved = self.config.get("slide_fluorescence_settings")
            if isinstance(saved, dict):
                for key in profile_keys:
                    if key in saved:
                        self._vars[key].set(saved[key])
                self.config["seg_per_channel_filters_enabled"] = bool(
                    saved.get("per_channel_enabled", False))
                self.config["seg_per_channel_filters"] = str(
                    saved.get("per_channel_filters", "{}"))
            elif self._vars["seg_preprocess_mode"].get() == "brightfield_invert":
                self._vars["seg_preprocess_mode"].set("local_bg_subtract")
        self._sync_vars_to_config()
        save_config(self.config)
        save_project_config(self.config)

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
        slide_scanner = self._vars["input_type"].get() == "vsi"
        is_brightfield = slide_scanner and self._vars["slide_signal_type"].get() == "brightfield"
        if is_brightfield:
            self._vars["seg_channels"].set("0")
        self._section_header(
            parent, 3,
            "Choose a model and segment cells in your images.",
        )

        # ── Input folder status ──────────────────────────────────────
        input_card = self._card(parent, "Input Folder")

        # Converter outputs available to segment (VSI_Converted_TIFF[_*]).
        self._seg_input_variants = [(name, path) for name, path in self._find_input_variants()
                                    if self._direct_image_count(path)] if slide_scanner else []
        _path_by_name = {n: p for n, p in self._seg_input_variants}

        folder = self._resolve_existing_input_folder()
        self.config["input_root_folder"] = folder
        self._vars["input_root_folder"].set(folder)
        if slide_scanner:
            self._align_converter_roi_folder(folder)
        input_ok = self._folder_image_count(folder) > 0

        self._seg_input_warn = tk.Label(
            input_card,
            text=("" if input_ok else "Choose the converted output for this run." if _path_by_name
                  else "⚠  No input images found — check your data source in Step 2."),
            bg=CARD, fg=RED, font=FONT, wraplength=700,
        )
        self._seg_input_warn.pack(anchor="w")

        self._seg_input_display = tk.Label(
            input_card,
            text=folder if input_ok else "—",
            bg=CARD, fg=FG if input_ok else MUTED, font=FONT_MONO, wraplength=700,
        )
        self._seg_input_display.pack(anchor="w", pady=(2, 0))
        if slide_scanner:
            roi_status = tk.Label(input_card, text=self._saved_roi_mask_text(), bg=CARD,
                                  fg=MUTED, font=FONT, wraplength=700, justify="left")
            roi_status.pack(anchor="w")

        # Dropdown to choose among multiple converter outputs (like Val/Analysis)
        if len(self._seg_input_variants) > 1 or (self._seg_input_variants and not input_ok):
            _names = [n for n, _ in self._seg_input_variants]
            _cur = next((n for n, p in self._seg_input_variants
                         if os.path.normpath(p) == os.path.normpath(folder)), "")
            self._seg_src_var = tk.StringVar(value=_cur)

            ttk.Separator(input_card, orient="horizontal").pack(fill="x", pady=(8, 6))
            tk.Label(input_card, text="Converted output to use:", bg=CARD, fg=MUTED,
                     font=FONT).pack(anchor="w")
            self._seg_src_cur_lbl = tk.Label(input_card, text=f"Using:  {_cur}" if _cur else "",
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
                roi_status.config(text=self._saved_roi_mask_text())
                self._seg_run_btn.config(state="normal" if has else "disabled")
                self._wizard_btn.config(state="normal" if has else "disabled")
                self._model_wizard_btn.config(state="normal" if (has and _PIL) else "disabled")
                resume_btn = getattr(self, "_seg_resume_btn", None)
                if resume_btn is not None and resume_btn.winfo_exists():
                    resume_btn.config(state="normal" if has else "disabled")
                # reset the channel preview for the newly chosen source
                self._ch_images = []
                self._ch_index = 0
                preview = getattr(self, "_ch_preview", None)
                if preview is not None and preview.winfo_exists():
                    preview.reset(has)
            self._seg_src_var.trace_add("write", _seg_src_changed)

            def _toggle_seg_src(lbl=_ovr_lbl):
                if _ovr_frame.winfo_ismapped():
                    _ovr_frame.pack_forget()
                    lbl.config(text="▶  Use a different converted output")
                else:
                    _ovr_frame.pack(fill="x", pady=(0, 4))
                    lbl.config(text="▼  Use a different converted output")
            _ovr_lbl.bind("<Button-1>", lambda _: _toggle_seg_src())
            if not input_ok:
                _toggle_seg_src()

        # ── Channel preview + picker ─────────────────────────────────
        ch_card = self._card(parent, "Channel Preview")
        self._ch_images: list = []
        self._ch_index: int = 0
        self._ch_selected: set = set()

        self._ch_preview = _ChannelPreview(
            ch_card, has_input=input_ok, navigate=self._ch_navigate,
            load=self._trigger_ch_preview_load, select=self._ch_toggle_current,
        )
        self._ch_preview.pack(anchor="w")

        self._ch_field_lbl = tk.Label(
            ch_card,
            text="Click the preview to add or remove this channel from prediction.",
            bg=CARD, fg=SECONDARY_FG, font=FONT,
        )
        self._ch_field_lbl.pack(anchor="w")

        # ── Channels field ───────────────────────────────────────────
        model_card = self._card(parent, "Segmentation Settings")

        # ── Backend selector ─────────────────────────────────────────
        _BK_DISPLAY = SEGMENTATION_BACKENDS
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
        ttk.Button(_bk_frame, text=f"{DOWNLOADS_TITLE}…",
                   command=self._open_builtin_model_downloads).pack(side="left", padx=(10, 0))
        tk.Label(model_card, text="Optional. The wizard downloads the models it needs. "
                 "Use this button to download additional models ahead of time.",
                 bg=CARD, fg=MUTED, font=FONT, wraplength=660, justify="left"
                 ).pack(anchor="w", pady=(0, 10))

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
                 bg=BG, fg=FG, insertbackground=FG, relief="flat", font=FONT, width=16,
                 state="readonly" if is_brightfield else "normal").pack(side="left")

        tk.Label(g, text=("Brightfield RGB is combined into one cell signal: channel 0."
                          if is_brightfield else
                          "0-based IDs, comma-separated  (e.g.  0  or  0,2)"),
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))

        tk.Label(g, text="Model type:", bg=CARD, fg=FG, font=FONT).grid(
            row=2, column=0, sticky="w", pady=3)
        sam1_models = MICROSAM_MODELS
        _cb_model = ttk.Combobox(
            g, textvariable=self._vars["seg_model_type"],
            values=sam1_models,
            state="readonly", width=18,
        )
        _cb_model.grid(row=2, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_model)

        legacy_model_card = tk.Frame(model_card, bg=CARD)
        choice_row = tk.Frame(legacy_model_card, bg=CARD)
        choice_row.pack(fill="x", pady=(6, 2))
        tk.Label(choice_row, text="Model for this run:", bg=CARD, fg=FG,
                 font=FONT, width=24, anchor="w").pack(side="left")
        model_choice = tk.StringVar()
        model_options = {}  # label -> (Cellpose variant, custom checkpoint path)
        model_choice_box = ttk.Combobox(choice_row, textvariable=model_choice,
                                        state="readonly", width=48)
        model_choice_box.pack(side="left", padx=8, fill="x", expand=True)
        self._bind_combobox_scroll(model_choice_box)

        custom_card = tk.Frame(legacy_model_card, bg=CARD)
        tk.Label(custom_card, text="Trained model (optional):", bg=CARD,
                 fg=FG, font=FONT).pack(anchor="w")
        model_path_row = tk.Frame(custom_card, bg=CARD)
        model_path_row.pack(fill="x", pady=(2, 4))
        custom_path = tk.StringVar()
        tk.Entry(model_path_row, textvariable=custom_path, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat", font=FONT).pack(
                     side="left", fill="x", expand=True)
        def _browse_model():
            if self._vars["seg_backend"].get() == "stardist":
                chosen = filedialog.askdirectory(parent=self.root,
                                                   title="Choose trained StarDist model folder")
            else:
                chosen = filedialog.askopenfilename(parent=self.root,
                                                     title="Choose trained model file")
            if chosen:
                custom_path.set(chosen)
        tk.Button(model_path_row, text="Browse…", command=_browse_model,
                  bg=ACTIVE, fg=FG, relief="flat", font=FONT).pack(side="left", padx=6)
        custom_card.pack(fill="x", pady=(4, 8))
        model_keys = SEGMENTATION_CHECKPOINT_KEYS
        selected_key = [model_keys.get(self.config.get("seg_backend"), "seg_custom_model_path")]
        custom_path.set(self._vars[selected_key[0]].get())
        def _model_path_changed(*_):
            self._vars[selected_key[0]].set(custom_path.get())
            if (self._vars["seg_backend"].get() == "cellpose"
                    and custom_path.get().strip()):
                self._vars["seg_cellpose_variant"].set("cpsam")
            if is_brightfield:
                self._vars["seg_preprocess_mode"].set(_brightfield_preprocess_for_model(
                    self._vars["seg_backend"].get(),
                    self._vars["seg_cellpose_variant"].get(), custom_path.get().strip(),
                    self.config.get("project_output_folder", "")))
            refresh_model_choices(self._vars["seg_backend"].get())
        custom_path.trace_add("write", _model_path_changed)

        def refresh_model_choices(backend):
            from .retraining import list_trained_models
            builtin = {"microsam": "Built-in Micro-SAM model",
                       "cellpose": "Built-in Cellpose-SAM (cpsam)",
                       "stardist": "Built-in StarDist (2D_versatile_fluo)"}[backend]
            model_options.clear()
            model_options[builtin] = ("cpsam", "")
            if backend == "cellpose":
                model_options["Cellpose 2 (cyto2)"] = ("cyto2", "")
                model_options["Rat Nissl v2 (Meystre et al.)"] = ("rat_nissl_v2", "")
            project = self.config.get("project_output_folder", "")
            for record in list_trained_models(project, backend):
                label = (f"{record['name']}  (ch{record['channel']}, "
                         f"{record['location']}, {record['record_folder'][-15:]})")
                model_options[label] = ("cpsam", record["model_path"])
            selected = custom_path.get().strip()
            if selected and selected not in (path for _, path in model_options.values()):
                model_options["External model (current path)"] = ("cpsam", selected)
            model_choice_box.configure(values=list(model_options))
            variant = (self._vars["seg_cellpose_variant"].get()
                       if backend == "cellpose" else "cpsam")
            model_choice.set(next((label for label, pair in model_options.items()
                                   if pair == (variant, selected)), builtin))

        def choose_model(_event=None):
            variant, path = model_options.get(model_choice.get(), ("cpsam", ""))
            if self._vars["seg_backend"].get() == "cellpose":
                self._vars["seg_cellpose_variant"].set(variant)
            custom_path.set(path)
            if self._vars["seg_backend"].get() == "cellpose":
                suggested = _registered_cellpose_inference(
                    path, self.config.get("project_output_folder", ""))
                self._vars["seg_cellpose_cellprob_threshold"].set("0.0")
                self._vars["seg_cellpose_flow_threshold"].set("0.4")
                for setting, key in (("cellprob_threshold", "seg_cellpose_cellprob_threshold"),
                                     ("flow_threshold", "seg_cellpose_flow_threshold")):
                    if setting in suggested:
                        self._vars[key].set(str(suggested[setting]))
            if is_brightfield:
                self._vars["seg_preprocess_mode"].set(_brightfield_preprocess_for_model(
                    self._vars["seg_backend"].get(), variant, path,
                    self.config.get("project_output_folder", "")))
        model_choice_box.bind("<<ComboboxSelected>>", choose_model)

        tk.Label(g, text="Preprocessing mode:", bg=CARD, fg=FG, font=FONT).grid(
            row=3, column=0, sticky="w", pady=3)
        _cb_prep = ttk.Combobox(
            g, textvariable=self._vars["seg_preprocess_mode"],
            values=["local_bg_subtract", "local_bg_subtract_mild",
                    "raw_percentile", "tile_clahe_mild", "tile_clahe"]
                   + (["brightfield_invert"] if is_brightfield else []),
            state="readonly", width=24,
        )
        _cb_prep.grid(row=3, column=1, sticky="w", pady=3, padx=8)
        self._bind_combobox_scroll(_cb_prep)
        sam2_input = tk.StringVar(value="Original RGB" if is_brightfield else "Original channel")
        sam2_hint = tk.Label(
            model_card,
            text=("Base is recommended for Nissl. The first run sets up micro-SAM2 "
                  "and downloads the selected model."),
            bg=CARD, fg=MUTED, font=("Segoe UI", 8), wraplength=650, justify="left",
        )
        if is_brightfield:
            tk.Label(g, text=("Preprocessing changes model input only. Images and the "
                              "Parameter Wizard retain the original stain brightness."),
                     bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                     wraplength=650, justify="left").grid(
                         row=4, column=0, columnspan=2, sticky="w", pady=(2, 5))

        # ── Cellpose parameters (shown only when that backend is chosen) ────
        self._cellpose_frame = tk.Frame(model_card, bg=CARD)
        tk.Label(self._cellpose_frame,
                 text="Cellpose parameters  (tune on a test image if needed):",
                 bg=CARD, fg=FG, font=FONT_BOLD, wraplength=700, justify="left").pack(anchor="w", pady=(0, 4))
        _cg = tk.Frame(self._cellpose_frame, bg=CARD)
        _cg.pack(fill="x")
        _cg.grid_columnconfigure(0, minsize=240)
        _cg.grid_columnconfigure(1, weight=1)
        # (label, config key, short unit hint, full explanation line)
        _cp_fields = [
            ("Cell diameter (px)", "seg_cellpose_diameter", "0 = auto",
             "Expected cell size in pixels. 0 uses the model default: Cellpose-SAM "
             "handles scale itself; older Cellpose models use 30 px. Set the typical "
             "diameter if detections look too coarse or too fine."),
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
                 text="Cellpose-SAM is the current default. Cellpose 2 and Rat Nissl v2 use a separate "
                      "legacy runtime prepared automatically on first use; the rat model was trained on "
                      "juvenile rat cresyl-violet images. Preprocessing and object filters still apply. "
                      "Selecting a saved model applies its recorded tested thresholds when available.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), wraplength=700, justify="left").pack(anchor="w", pady=(6, 0))

        def _on_seg_backend_changed(*_):
            bk = _BK_REVERSE.get(self._seg_backend_disp.get(), "microsam")
            self._vars["seg_backend"].set(bk)
            self._cellpose_frame.pack_forget()
            if bk == "microsam2":
                _cb_model.configure(textvariable=self._vars["seg_microsam2_model"],
                                    values=MICROSAM2_MODELS, state="readonly")
                _cb_prep.configure(textvariable=sam2_input, state="disabled")
                legacy_model_card.pack_forget()
                sam2_hint.pack(anchor="w", pady=(4, 8), after=g)
                return
            sam2_hint.pack_forget()
            legacy_model_card.pack(fill="x", after=g)
            _cb_model.configure(textvariable=self._vars["seg_model_type"],
                                values=sam1_models,
                                state="readonly" if bk == "microsam" else "disabled")
            _cb_prep.configure(textvariable=self._vars["seg_preprocess_mode"], state="readonly")
            if bk == "cellpose":
                self._cellpose_frame.pack(fill="x", pady=(6, 0))
            selected_key[0] = model_keys[bk]
            custom_path.set(self._vars[selected_key[0]].get())
            refresh_model_choices(bk)
            if is_brightfield:
                self._vars["seg_preprocess_mode"].set(_brightfield_preprocess_for_model(
                    bk, self._vars["seg_cellpose_variant"].get(),
                    custom_path.get().strip(),
                    self.config.get("project_output_folder", "")))
        self._seg_backend_disp.trace_add("write", _on_seg_backend_changed)
        _on_seg_backend_changed()   # apply initial state

        def apply_model_preview(settings, result, comparison, finished):
            channel = comparison["channel"]
            backend = settings["seg_backend"]
            checkpoint = settings.get(model_keys.get(backend)) or ""
            self._sync_vars_to_config()
            config = {**self.config, **settings}
            if not config["seg_channels"].strip():
                config["seg_channels"] = str(channel)

            def apply_to_ui(cached, can_reuse):
                try:
                    backend_name = SEGMENTATION_BACKENDS[backend]
                    if self._seg_backend_disp.get() != backend_name:
                        self._seg_backend_disp.set(backend_name)
                    if backend in model_keys and custom_path.get() != checkpoint:
                        custom_path.set(checkpoint)
                    variant_changed = (backend == "cellpose" and
                                       self._vars["seg_cellpose_variant"].get() != config["seg_cellpose_variant"])
                    for key in (*settings, "seg_channels"):
                        if self._vars[key].get() != str(config[key]):
                            self._vars[key].set(config[key])
                    if variant_changed:
                        refresh_model_choices(backend)
                except (OSError, KeyError, ValueError, tk.TclError) as error:
                    cached["directory"].cleanup()
                    finished(str(error))
                    return
                self.config.update(config)
                self._replace_model_preview_cache(cached)
                message = f"Selected {result['name']}."
                if can_reuse:
                    message += " The Parameter Wizard can reuse this preview."
                model_feedback.configure(text=message)
                model_feedback.pack(anchor="w", pady=(4, 0))
                finished()

            def worker():
                cached = None
                try:
                    if checkpoint and not os.path.exists(filesystem_path(checkpoint)):
                        raise ValueError("This checkpoint is unavailable on this computer.")
                    cached = self._build_model_preview_cache(
                        config, settings, result, channel, comparison["preview"], comparison)
                    can_reuse = self._preview_matches(cached, config, self._configured_channels(config))
                    save_config(config)
                    save_project_config(config)
                except Exception as error:
                    if cached:
                        cached["directory"].cleanup()
                    self.root.after(0, finished, str(error))
                else:
                    self.root.after(0, apply_to_ui, cached, can_reuse)

            threading.Thread(target=worker, daemon=True).start()

        wizard_row = tk.Frame(model_card, bg=CARD)
        wizard_row.pack(fill="x", pady=(7, 0))
        project_folder = self.config.get("project_output_folder", "")
        has_saved_preview = project_folder and (Path(project_folder) / MODEL_PREVIEW_FOLDER).is_dir()
        self._model_wizard_btn = tk.Button(
            wizard_row, text="🪄  Model Selection Wizard",
            command=lambda: self._launch_model_wizard(apply_model_preview),
            bg=ACCENT, fg="#000000", activeforeground="#000000", relief="flat",
            font=FONT_BOLD, padx=12, pady=6,
            state="normal" if _PIL and (input_ok or has_saved_preview) else "disabled",
        )
        self._model_wizard_btn.pack(side="left")
        tk.Label(wizard_row, text="Recommended. Compare models on a small patch and apply "
                 "the settings for the one you choose.",
                 bg=CARD, fg=FG, font=FONT, wraplength=410, justify="left").pack(side="left", padx=10)
        model_feedback = tk.Label(model_card, text="", bg=CARD, fg=GREEN, font=FONT,
                                  wraplength=650, justify="left")

        filt_card = self._card(parent, "Object Filtering")
        wizard_row = tk.Frame(filt_card, bg=CARD)
        wizard_row.pack(anchor="w", fill="x", pady=(0, 10))
        tk.Label(
            wizard_row,
            text="Recommended. Pick example cells and the wizard fills in size, shape, "
                 "and intensity limits. A limit of 0 turns that filter off.",
            bg=CARD, fg=MUTED, font=FONT, justify="left", wraplength=650,
        ).pack(anchor="w")
        _wiz_btn_row = tk.Frame(wizard_row, bg=CARD)
        _wiz_btn_row.pack(anchor="w", pady=(4, 0))
        self._wizard_btn = tk.Button(
            _wiz_btn_row, text="🧙  Parameter Wizard",
            command=self._launch_parameter_wizard,
            bg=ACCENT, fg="#000000", activeforeground="#000000", font=FONT_BOLD, relief="flat",
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
            (("Max mean raw brightness:", "seg_raw_mean_max", "0 = off; darker cells kept")
             if is_brightfield else
             ("Min mean raw intensity:", "seg_raw_mean_min", "e.g. 5000")),
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
        per_channel_check = tk.Checkbutton(
            filt_card,
            text="Set individual filter thresholds per channel",
            variable=self._per_ch_filt_enabled,
            command=self._toggle_per_ch_filters,
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        )
        per_channel_check.pack(anchor="w", pady=(12, 2))
        if is_brightfield:
            per_channel_check.config(state="disabled")
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
        if _prog and not _prog.get("finished", True):
            self._seg_resume_btn = tk.Button(
                btn_row, text="⟲  Continue a previous run",
                command=lambda: self._run_segmentation_step(resume=True),
                bg=ACTIVE, fg=FG, font=FONT_BOLD, relief="flat",
                padx=14, pady=10,
                state="normal" if input_ok else "disabled",
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

    def _toggle_per_ch_filters(self, *, persist: bool = True) -> None:
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
        if persist:
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
        if self.config.get("input_type") == "vsi":
            saved_output = self.config.get("vsi_output_dir", "").strip()
            # Some early projects pointed to the converter output itself.
            # Scan its exports so the neighboring ROI-mask TIFFs stay separate.
            if cfg_folder:
                exports = os.path.join(cfg_folder, "exports")
                if self._direct_image_count(exports):
                    return exports
                if (not self._direct_image_count(cfg_folder)
                        and os.path.isdir(os.path.join(cfg_folder, "DONOTDELETE-MASK_FOR_VSI"))):
                    cfg_folder = exports
            if self._folder_image_count(cfg_folder):
                return cfg_folder
            if saved_output:
                for path in (os.path.join(saved_output, "exports"), saved_output):
                    if self._direct_image_count(path):
                        return path
            populated = [path for _, path in self._find_input_variants()
                         if self._direct_image_count(path)]
            return populated[0] if len(populated) == 1 else cfg_folder

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

    def _load_channel_previews(self, callback) -> None:
        """Run in daemon thread. Posts result back to main thread via callback."""
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
            roi_path = None
            display_native = False

            if self.config.get("input_type") == "vsi" and path.lower().endswith((".tif", ".tiff")):
                from .slide_roi_preview import slide_roi_for_sample
                from .slide_tiff import TiffSlide

                slide = TiffSlide(path)
                try:
                    scene = slide.get_scene(0)
                    w, h = scene.size
                    factor = min(1, 640 / max(w, h))
                    data = scene.read_block(size=(max(1, round(w * factor)), max(1, round(h * factor))))
                finally:
                    slide.close()
                display_native = is_brightfield(self.config)
                channels_np = ([data[..., :3] if data.ndim == 3 and data.shape[-1] >= 3
                                else data[..., 0] if data.ndim == 3 else data]
                               if display_native or data.ndim == 2
                               else [data[..., i] for i in range(data.shape[-1])])
                roi_path = slide_roi_for_sample(os.path.basename(path), source=path)
            elif path.lower().endswith(".czi"):
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
            from .slide_roi_preview import overlay_slide_outline
            photos = []
            for ch in channels_np:
                if not (display_native and ch.dtype == np.uint8):
                    lo, hi = np.percentile(ch, [1, 99.5])
                    ch = np.uint8(np.clip((ch.astype(float) - lo) / max(hi - lo, 1) * 255, 0, 255))
                img = PILImage.fromarray(ch)
                img.thumbnail((280, 200), PILImage.LANCZOS)
                img = overlay_slide_outline(img, roi_path)
                photos.append(ImageTk.PhotoImage(img))

            print(f"[Preview] Done — {len(photos)} channel(s) loaded")
            self.root.after(0, callback, photos, fname)
        except Exception as exc:
            print(f"[Preview] Error: {exc}")
            self.root.after(0, callback, [], f"Preview error: {exc}")

    def _trigger_ch_preview_load(self) -> None:
        """Button handler for Step 3 — load channel preview in background."""
        self._start_preview_load(self._ch_preview, self._set_channel_previews)

    def _start_preview_load(self, preview, on_loaded) -> None:
        request = preview.start_load()

        def receive(photos, message):
            if preview.winfo_exists() and preview.request is request:
                preview.finish_load()
                on_loaded(photos, message)

        threading.Thread(target=self._load_channel_previews, args=(receive,), daemon=True).start()

    def _set_channel_previews(self, photos: list, fname: str = "") -> None:
        self._ch_images = photos
        self._ch_index = 0
        if not photos:
            self._ch_preview.show(fname or "No preview available", color=SECONDARY_FG)
            return
        # Restore any previously selected channels from the text field
        raw = self._vars["seg_channels"].get().strip()
        self._ch_selected = {int(x.strip()) for x in raw.split(",")
                             if x.strip().isdigit()}
        self._show_channel(0)

    def _show_channel(self, idx: int) -> None:
        if not self._ch_images:
            return
        idx = max(0, min(idx, len(self._ch_images) - 1))
        self._ch_index = idx
        selected = idx in self._ch_selected
        self._ch_preview.show_channel(self._ch_images, idx,
                                      indicator="  ✔ selected" if selected else "",
                                      color=GREEN if selected else FG)

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
        if run_is_brightfield(seg_root, self.config):
            return ["0"]

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

    def _detect_predicted_channels(self, *, fallback_to_config: bool = True) -> list:
        """Return sorted channel-index strings that actually have a predicted
        mask (``*_predict_original_ch*.tif``) on disk, scanning the selected
        segmentation output folder.

        This lets Validation / Analysis work from an imported project on
        another computer even when the project file no longer records that a
        segmentation was run.  Falls back to the configured seg_channels.
        """
        proj = self.config.get("project_output_folder", "").strip()
        if proj and os.path.isdir(proj):
            # The segmentation-folder selection is kept in config and shared
            # between the Validation and Analysis pages; fall back to newest.
            seg_sub = (self.config.get("seg_input_subfolder", "").strip()
                       or (self._find_variants(SUBFOLDER["seg"])
                           or [SUBFOLDER["seg"]])[-1])
            seg_root = os.path.join(proj, seg_sub)
            brightfield = run_is_brightfield(seg_root, self.config)
            if brightfield and fallback_to_config:
                return ["0"]
            chs = set()
            if os.path.isdir(seg_root):
                for d in sorted(os.listdir(seg_root)):
                    sub = os.path.join(seg_root, d)
                    if not os.path.isdir(sub) or d.startswith((".", "DEBUG")):
                        continue
                    for name in os.listdir(sub):
                        if name.startswith((".", "DEBUG")) or not name.lower().endswith(".tif"):
                            continue
                        channel = prediction_channel(name)
                        if channel is not None and os.path.isfile(os.path.join(sub, name)):
                            chs.add(channel)
            if brightfield:
                chs.intersection_update({0})
            if chs:
                return [str(c) for c in sorted(chs)]
        if not fallback_to_config:
            return []
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

    # ------------------------------------------------------------------
    # Validation channel preview helpers
    # ------------------------------------------------------------------

    def _trigger_val_preview_load(self) -> None:
        self._start_preview_load(self._val_preview, self._set_val_channel_previews)

    def _set_val_channel_previews(self, photos: list, fname: str = "") -> None:
        """Called on the main thread after background channel loading."""
        self._val_ch_images = photos
        if photos:
            self._show_val_channel(self._selected_validation_channel())
        else:
            self._val_preview.show(fname or "No preview available", color=SECONDARY_FG)

    def _selected_validation_channel(self) -> int | None:
        value = self._val_channel_var.get().removeprefix("Channel ")
        return int(value) if value.isdecimal() else None

    def _show_val_channel(self, idx: int | None) -> None:
        if not self._val_ch_images:
            return
        n = len(self._val_ch_images)
        if idx is not None and not 0 <= idx < n:
            self._val_ch_index = n
            self._val_preview.show(f"Channel {idx} unavailable in this image", color=SECONDARY_FG,
                                   previous=True)
            return
        self._val_ch_index = idx = 0 if idx is None else idx
        selected = idx == self._selected_validation_channel()
        self._val_preview.show_channel(self._val_ch_images, idx,
                                       indicator="  ✔ selected" if selected else "",
                                       color=GREEN if selected else FG)

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
        if (self.config.get("input_type") == "vsi"
                and self.config.get("slide_signal_type") == "brightfield"
                and channels != "0"):
            self._seg_status.config(
                text="Brightfield stain uses one combined signal. Set Channels to predict to 0.",
                fg=RED)
            return

        # A named model is trained from one channel. Keep multi-channel runs
        # possible, but make cross-channel reuse an explicit choice.
        from .retraining import list_trained_models
        backend = self.config.get("seg_backend", "microsam")
        model_key = {"microsam": "seg_custom_model_path",
                     "cellpose": "seg_cellpose_model_path",
                     "stardist": "seg_stardist_model_path"}.get(backend)
        selected_model = str(self.config.get(model_key) or "") if model_key else ""
        if selected_model:
            try:
                requested_channels = {int(part.strip()) for part in channels.split(",")}
            except ValueError:
                requested_channels = set()
            for record in list_trained_models(
                    self.config.get("project_output_folder", ""), backend):
                if os.path.normpath(record["model_path"]) != os.path.normpath(selected_model):
                    continue
                if requested_channels - {record["channel"]} and not messagebox.askyesno(
                    "Use a model on other channels?",
                    f"{record['name']} was trained on channel {record['channel']}, "
                    f"but this run includes {channels}. The same model will be used "
                    "on every selected channel. Continue?",
                    parent=self.root, default=messagebox.NO, icon="warning"):
                    return
                break

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

        self._archive_model_preview(out, _seg_log_line)

        def _thread():
            try:
                if (self.config.get("seg_backend") == "cellpose"
                        and self.config.get("seg_cellpose_variant", "cpsam") != "cpsam"):
                    self.root.after(0, _seg_log_line,
                                    "Preparing the selected Cellpose model; first use "
                                    "downloads its weights and separate runtime…")
                command, environment = segmentation_command(script, CONFIG_PATH, self.config)
                proc = subprocess.Popen(
                    command, env=environment,
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
                error_text = str(exc)
                def _err():
                    try:
                        self._seg_progress.stop()
                        self._seg_run_btn.config(state="normal", text="▶  Run Segmentation")
                        if hasattr(self, "_seg_resume_btn"):
                            self._seg_resume_btn.config(state="normal")
                        self._seg_status.config(text=f"✗  {error_text}", fg=RED)
                        _seg_log_line(f"[ERROR] {error_text}")
                    except tk.TclError:
                        pass
                self.root.after(0, _err)

        threading.Thread(target=_thread, daemon=True).start()

    def _open_builtin_model_downloads(self) -> None:
        from .model_wizard import open_model_downloads

        if self._is_running("Model Selection Wizard"):
            messagebox.showinfo("Model comparison is running",
                                f"Use {DOWNLOADS_TITLE} after the comparison finishes.",
                                parent=self.root)
            return
        self._sync_vars_to_config()
        models, _ = _model_wizard_options(self.config)
        open_model_downloads(self.root, models, self._track_proc)

    def _launch_model_wizard(self, apply_model) -> None:
        if self._running_procs():
            messagebox.showinfo("A process is running",
                                "Finish the current task before comparing models.", parent=self.root)
            return
        self._sync_vars_to_config()
        input_folder = self.config.get("input_root_folder", "").strip()
        from .model_wizard import ModelWizard
        from .model_wizard_archive import load, saved

        snapshot = self.config.copy()
        project = snapshot.get("project_output_folder")
        previews = list(saved(project))
        review = next((item for item in previews if item["signature"].get("input_root_folder") == input_folder),
                      previews[0] if previews else None)
        recent = getattr(self, "_model_preview_cache", None)
        if recent and recent["signature"].get("project_output_folder") == project and (
                recent["signature"].get("input_root_folder") == input_folder and recent["archive"].is_dir()):
            try:
                review = load(recent["archive"])
            except (OSError, KeyError, TypeError, ValueError):
                pass
        if review is None and not os.path.isdir(input_folder):
            messagebox.showwarning("Choose an input folder", "Select your images in Step 2.",
                                   parent=self.root)
            return

        try:
            models, defaults = _model_wizard_options(snapshot)
            self._model_wizard = ModelWizard(self.root, snapshot, models, defaults, apply_model,
                                              self._launch_parameter_wizard, self._track_proc,
                                              saved=review, can_reuse=bool(review and self._preview_matches(
                                                  self._archived_preview_key(review), snapshot,
                                                  self._configured_channels())))
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open the Model Selection Wizard", str(error),
                                 parent=self.root)

    def _configured_channels(self, config=None):
        config = self.config if config is None else config
        if is_brightfield(config):
            return {0}
        return {int(part.strip()) for part in str(config.get("seg_channels") or "").split(",")
                if part.strip().isdigit()}

    def _cache_selected_model_preview(self, settings, result, channel, preview, comparison=None) -> None:
        cached = self._build_model_preview_cache(self.config, settings, result, channel, preview, comparison)
        self._replace_model_preview_cache(cached)

    @staticmethod
    def _build_model_preview_cache(config, settings, result, channel, preview, comparison=None):
        from .model_wizard_compute import file_identity, write_json

        inputs = ("project_output_folder", "input_root_folder", "input_type",
                  "slide_signal_type", "seg_skip_subdirs", "seg_tile_h", "seg_tile_w",
                  "seg_halo_h", "seg_halo_w")
        signature = {key: config.get(key) for key in (*inputs, *settings)}
        model_key = SEGMENTATION_CHECKPOINT_KEYS.get(settings["seg_backend"])
        model = signature.get(model_key) if model_key else None
        model_identity = file_identity(model) if model else None
        directory = TemporaryDirectory(prefix="cellmatchr-filter-preview-")
        dest = Path(directory.name)
        entry = {"channel": channel, "n_objects": result["count"], "model_preview": True}
        for item in ("raw", "labels", "source_rgb"):
            if result.get(item):
                target = dest / f"{item}.tif"
                shutil.copyfile(result[item], target)
                entry[item] = str(target)
        manifest = dest / "parameter_wizard_manifest.json"
        write_json(manifest, {
            "base_name": preview["sample"], "channels": [entry],
            "slide_signal_type": "brightfield" if preview["brightfield"] else "fluorescence",
        })
        archive = dest / "comparison"
        if comparison:
            from .model_wizard_archive import capture
            capture(archive, comparison, result, signature, model_identity)
        return {
            "directory": directory, "manifest": str(manifest), "channel": channel,
            "signature": signature, "source": tuple(preview["source_path"]), "scene": preview["scene"],
            "model": model_identity, "archive": archive,
        }

    def _replace_model_preview_cache(self, cached):
        old = getattr(self, "_model_preview_cache", None)
        self._model_preview_cache = cached
        if old:
            old["directory"].cleanup()

    @staticmethod
    def _archived_preview_key(comparison):
        preview = comparison["preview"]
        return {"channel": comparison["channel"], "signature": comparison["signature"],
                "source": tuple(preview["source_path"]), "scene": preview["scene"],
                "model": tuple(comparison["model_identity"]) if comparison["model_identity"] else None}

    def _preview_matches(self, cached, config, channels):
        if (not cached or cached["channel"] not in channels or
                any(config.get(key) != value for key, value in cached["signature"].items())):
            return False
        from .model_wizard_compute import file_identity

        try:
            source = _first_confocal_preview_input(
                config["input_root_folder"], self._INPUT_EXTS, bool(config["seg_skip_subdirs"]))
            return bool(source and source[1] == cached["scene"] and file_identity(source[0]) == cached["source"]
                        and (not cached["model"] or file_identity(cached["model"][0]) == cached["model"]))
        except OSError:
            return False

    def _matching_model_preview(self, config, channels):
        cached = getattr(self, "_model_preview_cache", None)
        if self._preview_matches(cached, config, channels):
            dest = Path(cached["directory"].name)
            if all((dest / name).is_file() for name in ("raw.tif", "labels.tif", "parameter_wizard_manifest.json")):
                return cached
        from .model_wizard_archive import saved

        for comparison in saved(config.get("project_output_folder")):
            if self._preview_matches(self._archived_preview_key(comparison), config, channels):
                result = next(item for item in comparison["results"] if item["name"] == comparison["chosen"])
                self._cache_selected_model_preview(comparison["models"][comparison["chosen"]],
                                                   result, comparison["channel"], comparison["preview"])
                return self._model_preview_cache
        return None

    def _archive_model_preview(self, out, log):
        cached = getattr(self, "_model_preview_cache", None)
        if (not cached or not cached["archive"].is_dir() or
                any(self.config.get(key) for key in ("seg_test_run", "seg_dry_run", "seg_resume")) or
                not self._preview_matches(cached, self.config, {cached["channel"]})):
            return
        from .model_wizard_archive import publish

        try:
            path = publish(cached["archive"], self.config["project_output_folder"], out)
        except OSError as exc:
            log(f"Could not save the model comparison: {exc}")
        else:
            log(f"Model comparison saved to {path}")

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
        try:
            requested_channels = {int(part.strip()) for part in channels.split(",") if part.strip()}
        except ValueError:
            self._wizard_status.config(text="Use comma-separated channel numbers in Step 3.", fg=RED)
            return

        save_config(self.config)
        save_project_config(self.config)
        config = self.config.copy()
        cached = self._matching_model_preview(config, requested_channels)

        out_dir = os.path.join(proj, "ParameterWizard_preview")
        manifest_path = os.path.join(out_dir, "parameter_wizard_manifest.json")

        self._wizard_btn.config(state="disabled", text="Running…")
        self._wizard_status.config(
            text=(f"Reusing the model comparison patch for channel {cached['channel']}…"
                  if cached else "Preparing unfiltered predictions…"), fg=ACCENT)
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

        def _run_and_stream(cmd: list, environment: dict | None = None) -> int:
            proc = subprocess.Popen(
                cmd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
                display_manifest = manifest_path
                reuse_args = []
                if cached:
                    self.root.after(0, _log_line,
                                    f"Using the model comparison patch for channel {cached['channel']}.")
                    reuse_args = ["--reuse-manifest", cached["manifest"]]
                if cached and requested_channels == {cached["channel"]}:
                    display_manifest = cached["manifest"]
                else:
                    self.root.after(0, _log_line, "▶  Parameter Wizard — unfiltered segmentation")
                    command, environment = segmentation_command(
                        self._script("parameter_wizard_compute.py"), CONFIG_PATH, config)
                    rc = _run_and_stream(command + ["--out-dir", out_dir] + reuse_args, environment)
                    if rc != 0:
                        self.root.after(0, _log_line, f"\n✗  Segmentation failed (exit {rc}).")
                        self.root.after(0, _restore, f"✗  Segmentation failed (exit {rc}).", RED)
                        return

                self.root.after(0, _log_line, f"\n{'─'*60}")
                self.root.after(0, _log_line, "▶  Parameter Wizard — Stage 2: pick objects (napari)")
                self.root.after(0, _log_line, f"{'─'*60}")
                self.root.after(0, lambda: self._wizard_status.config(
                    text="Opening napari — pick your objects there…", fg=ACCENT))
                rc = _run_and_stream([
                    sys.executable, self._script("parameter_wizard.py"),
                    "--config", CONFIG_PATH, "--manifest", display_manifest,
                ])

                def _done():
                    if rc == 0:
                        updated = load_config()
                        global_filters = ("seg_min_object_size", "seg_max_object_size",
                                          "seg_max_ratio", "seg_raw_mean_min", "seg_raw_mean_max")
                        filter_keys = (*global_filters, "seg_per_channel_filters",
                                       "seg_per_channel_filters_enabled")
                        if all(updated.get(key) == config.get(key) for key in filter_keys):
                            _restore("Filter settings unchanged.", FG)
                            return
                        self.config.update({key: updated[key] for key in filter_keys})
                        for key in global_filters:
                            self._vars[key].set(str(self.config[key]))
                        self._per_ch_filt_enabled.set(self.config["seg_per_channel_filters_enabled"])
                        self._toggle_per_ch_filters(persist=False)
                        _log_line("\n✔  Parameter Wizard finished — filter values updated.")
                        _restore("✔  Filter values updated below.", GREEN)
                    else:
                        _restore(f"✗  Failed (exit {rc}).", RED)
                        _log_line(f"\n✗  Parameter Wizard failed (exit code {rc}).")
                self.root.after(0, _done)
            except Exception as exc:
                self.root.after(0, _restore, f"✗  {exc}", RED)

        threading.Thread(target=_thread, daemon=True).start()

    def _launch_roi_editor(self) -> None:
        """Open the whole-image editor for the selected channel."""
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

        channel = self._selected_validation_channel()
        if channel is None:
            return

        if hasattr(self, "_review_btn"):
            self._review_btn.config(state="disabled", text="  Opening…")
        if hasattr(self, "_review_status"):
            self._review_status.config(text="Launching napari…", fg=ACCENT)

        def _restore():
            if hasattr(self, "_review_btn"):
                try:
                    self._review_btn.config(state="normal",
                                            text="▶  Review Segmented Images")
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

        if self._selected_validation_channel() is None:
            return

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

        def _done(process=None):
            if self._val_spin_job:
                self.root.after_cancel(self._val_spin_job)
                self._val_spin_job = None
            if self._val_run_btn:
                try:
                    self._val_run_btn.config(state="normal",
                                             text="▶  Review Individual Objects")
                except tk.TclError:
                    pass
            if hasattr(self, "_val_status"):
                try:
                    code = process.poll() if process is not None else None
                    if code is None:
                        self._val_status.config(text="✔  Validation window opened.", fg=GREEN)
                    elif code == 0:
                        self._val_status.config(text="Validation window closed.", fg=MUTED)
                    else:
                        self._val_status.config(text=f"✗  Validator exited with code {code}.", fg=RED)
                except tk.TclError:
                    pass

        try:
            process = subprocess.Popen(
                [sys.executable, self._script("validator.py"), "--config", CONFIG_PATH]
            )
            self._track_proc(process, "Validation window")
            _spin()
            # Re-enable after 5 s — enough time for the window to appear
            self.root.after(5000, _done, process)
        except Exception as exc:
            _done()
            if hasattr(self, "_val_status"):
                self._val_status.config(text=f"✗  {exc}", fg=RED)

    # ==================================================================
    # STEP 4 — Validation
    # ==================================================================

    def _build_validation_step(self, parent) -> None:
        self._section_header(
            parent, 4,
            "Choose a channel, then review whole images, individual objects, or both. "
            "Analysis uses your saved corrections and review decisions automatically.",
            optional=True,
        )

        channels_card = self._card(parent, "Channel to Validate")
        source_row = tk.Frame(channels_card, bg=CARD)
        source_row.pack(anchor="w")
        self._val_channel_var = tk.StringVar()
        channel_picker = ttk.Combobox(
            source_row, textvariable=self._val_channel_var, width=16,
        )
        channel_picker.pack(side="left")
        self._bind_combobox_scroll(channel_picker)

        runs = self._find_variants(SUBFOLDER["seg"])
        saved_run = self.config.get("seg_input_subfolder", "").strip()
        selected_run = saved_run if saved_run in runs else (runs or [SUBFOLDER["seg"]])[-1]
        self.config["seg_input_subfolder"] = selected_run
        self._val_input_var = tk.StringVar(value=selected_run)
        if len(runs) > 1:
            tk.Label(source_row, text="Segmentation run", bg=CARD, fg=FG,
                     font=FONT).pack(side="left", padx=(16, 6))
            run_picker = ttk.Combobox(
                source_row, textvariable=self._val_input_var,
                values=runs, state="readonly", width=28,
            )
            run_picker.pack(side="left")
            self._bind_combobox_scroll(run_picker)

        channel_help = tk.Label(channels_card, bg=CARD, fg=YELLOW, font=FONT,
                                wraplength=660, justify="left")

        self._val_ch_images = []
        self._val_ch_index = 0
        has_input = bool(self.config.get("input_root_folder", "").strip())
        self._val_preview = _ChannelPreview(
            channels_card, has_input=has_input, navigate=self._val_ch_navigate,
            load=self._trigger_val_preview_load,
        )
        self._val_preview.pack(anchor="w", pady=(12, 0))

        bulk_card = self._card(parent, "Bulk Review")
        tk.Label(
            bulk_card,
            text="Scan whole images quickly. Paint in missed cells and remove obvious false "
                 "detections, then save the corrected masks. Small errors may remain.",
            bg=CARD, fg=FG, font=FONT, wraplength=660, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        bulk_row = tk.Frame(bulk_card, bg=CARD)
        bulk_row.pack(anchor="w")
        self._review_btn = tk.Button(
            bulk_row, text="▶  Review Segmented Images", command=self._launch_roi_editor,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat", padx=12, pady=8,
        )
        self._review_btn.pack(side="left")
        self._review_status = tk.Label(bulk_row, text="", bg=CARD, fg=MUTED, font=FONT)
        self._review_status.pack(side="left", padx=10)

        self._review_test_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bulk_card, text="Test mode (one image, saves a separate debug mask)",
            variable=self._review_test_mode,
            bg=CARD, fg=MUTED, selectcolor=CARD, activebackground=CARD, font=FONT,
        ).pack(anchor="w", pady=(6, 0))

        individual_card = self._card(parent, "Individual Object Review")
        tk.Label(
            individual_card,
            text="Complete Bulk Review and save your corrections before starting Individual Object Review.",
            bg=CARD, fg=FG, font=FONT_BOLD, wraplength=660, justify="left",
        ).pack(anchor="w", pady=(0, 6))
        tk.Label(
            individual_card,
            text="Inspect each detected object and save a decision. This takes longer and "
                 "gives you finer control over which detections to keep and how to label them.",
            bg=CARD, fg=FG, font=FONT, wraplength=660, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        individual_row = tk.Frame(individual_card, bg=CARD)
        individual_row.pack(anchor="w")
        self._val_run_btn = tk.Button(
            individual_row, text="▶  Review Individual Objects",
            command=self._run_validation_step,
            bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD, relief="flat", padx=12, pady=8,
        )
        self._val_run_btn.pack(side="left")
        self._val_status = tk.Label(individual_row, text="", bg=CARD, fg=MUTED, font=FONT)
        self._val_status.pack(side="left", padx=10)

        self._val_ch_names_frame = tk.Frame(individual_card, bg=CARD)
        self._val_ch_name_vars = []
        self._val_ch_color_vars = []

        def toggle_channel_display():
            if self._val_ch_names_frame.winfo_manager():
                self._val_ch_names_frame.pack_forget()
                display_button.config(text="Channel names and colors…")
            else:
                self._build_val_ch_name_rows(self._detect_all_channel_labels())
                self._val_ch_names_frame.pack(fill="x", pady=(6, 0), after=display_button)
                display_button.config(text="Hide channel names and colors")

        display_button = ttk.Button(individual_card, text="Channel names and colors…",
                                    command=toggle_channel_display)
        display_button.pack(anchor="w", pady=(8, 0))

        ttk.Separator(individual_card, orient="horizontal").pack(fill="x", pady=(12, 6))
        prog_hdr = tk.Frame(individual_card, bg=CARD)
        prog_hdr.pack(fill="x")
        tk.Label(prog_hdr, text="Individual review progress",
                 bg=CARD, fg=FG, font=FONT_BOLD).pack(side="left")
        ttk.Button(prog_hdr, text="Refresh", command=self._refresh_val_status).pack(
            side="left", padx=8)
        self._val_prog_frame = tk.Frame(individual_card, bg=CARD)
        self._val_prog_frame.pack(anchor="w", fill="x", pady=(4, 0))

        def select_channel(*_):
            selected = self._selected_validation_channel()
            if selected is None:
                self._show_val_channel(None)
                return
            pattern = f"*_predict_original_ch{selected}*.tif"
            self.config["val_mask_pattern"] = pattern
            self._vars["val_mask_pattern"].set(pattern)
            self._show_val_channel(selected)

        def refresh_run(*_):
            self.config["seg_input_subfolder"] = self._val_input_var.get()
            options = self._detect_predicted_channels(fallback_to_config=False)
            saved = str(prediction_channel(self.config.get("val_mask_pattern", "")))
            selected = saved if saved in options else (options or [""])[0]
            channel_picker.config(values=[f"Channel {ch}" for ch in options],
                                  state="readonly" if options else "disabled")
            self._val_channel_var.set(f"Channel {selected}" if selected else "")
            state = "normal" if options else "disabled"
            for button in (self._review_btn, self._val_run_btn, display_button):
                button.config(state=state)
            if options:
                channel_help.pack_forget()
            else:
                channel_help.config(text="No predictions in this run. Run segmentation in Step 3 first.")
                channel_help.pack(anchor="w", pady=(5, 0), before=self._val_preview)
            if self._val_ch_names_frame.winfo_manager():
                self._build_val_ch_name_rows(self._detect_all_channel_labels())
            self._refresh_val_status()

        self._val_channel_var.trace_add("write", select_channel)
        self._val_input_var.trace_add("write", refresh_run)
        refresh_run()

    # ==================================================================
    # STEP 5 — Analysis
    # ==================================================================

    def _build_analysis_step(self, parent) -> None:
        self._section_header(
            parent, 5,
            "Measure cells in one channel or compare detections across two channels.",
            optional=True,
        )

        def details(card, title, on_open=None):
            body = tk.Frame(card, bg=CARD)

            def toggle():
                if body.winfo_manager():
                    body.pack_forget()
                    button.config(text=f"▸  {title}")
                else:
                    body.pack(fill="x", after=button, pady=(4, 0))
                    button.config(text=f"▾  {title}")
                    if on_open is not None:
                        on_open()

            button = ttk.Button(card, text=f"▸  {title}", command=toggle)
            button.pack(anchor="w", pady=(8, 0))
            return body

        runs = self._find_variants(SUBFOLDER["seg"])
        saved_run = self.config.get("seg_input_subfolder", "").strip()
        selected_run = saved_run if saved_run in runs else runs[0] if len(runs) == 1 else ""
        self._ana_input_var = tk.StringVar(value=selected_run)
        if not selected_run:
            self._sync_roi_mask_to_input()

        input_card = self._card(parent, "Input")
        source_row = tk.Frame(input_card, bg=CARD)
        source_row.pack(fill="x")
        tk.Label(source_row, text="Segmentation run", bg=CARD, fg=FG,
                 font=FONT_BOLD).pack(side="left", padx=(0, 10))
        if len(runs) > 1:
            run_picker = ttk.Combobox(source_row, textvariable=self._ana_input_var,
                                      values=runs, state="readonly", width=35)
            run_picker.pack(side="left")
            self._bind_combobox_scroll(run_picker)
        else:
            tk.Label(source_row, text=selected_run or "Run segmentation in Step 3 first.",
                     bg=CARD, fg=FG if selected_run else YELLOW, font=FONT).pack(side="left")
        run_details = tk.Label(input_card, bg=CARD, fg=FG, font=FONT,
                               wraplength=600, justify="left")
        if runs:
            run_details.pack(anchor="w", pady=(4, 2))

        self._path_row(input_card, "ROI mask folder (optional)", "ana_roi_mask_folder")
        roi_help = tk.Label(input_card, bg=CARD, fg=FG, font=FONT,
                            wraplength=600, justify="left")
        roi_help.pack(anchor="w", pady=(4, 0))

        roi_options = details(input_card, "ROI options")
        roi_rule_row = tk.Frame(roi_options, bg=CARD)
        roi_rule_row.pack(anchor="w")
        tk.Label(roi_rule_row, text="Cell inclusion rule", bg=CARD, fg=FG,
                 font=FONT).pack(side="left", padx=(0, 10))
        rule_picker = ttk.Combobox(
            roi_rule_row, textvariable=self._vars["ana_roi_cell_rule"],
            values=("centroid", "any_pixel"), state="readonly", width=14,
        )
        rule_picker.pack(side="left")
        self._bind_combobox_scroll(rule_picker)

        def review_roi_matches():
            self._sync_vars_to_config()
            run = self._choose_analysis_run()
            if run is None:
                messagebox.showinfo("ROI mask matches", "Choose a segmentation run first.",
                                    parent=self.root)
                return
            project = self.config.get("project_output_folder", "").strip()
            self._confirm_legacy_roi_matches(os.path.join(project, run), review=True)

        ttk.Button(roi_options, text="Review ROI mask matches",
                   command=review_roi_matches).pack(anchor="w", pady=(6, 0))

        measurement_card = self._card(parent, "Measurements")
        primary_row = tk.Frame(measurement_card, bg=CARD)
        primary_row.pack(fill="x")
        primary_row.columnconfigure(0, minsize=190)
        self._ana_coi_chan_var = tk.StringVar(value=f"Channel {self._vars['ana_red_channel'].get()}")
        self._ana_pm_chan_var = tk.StringVar(value=f"Channel {self._vars['ana_pv_channel'].get()}")
        tk.Label(primary_row, text="Channel to measure", bg=CARD, fg=FG,
                 font=FONT).grid(row=0, column=0, sticky="w")
        primary_picker = ttk.Combobox(primary_row, textvariable=self._ana_coi_chan_var, width=16)
        primary_picker.grid(row=0, column=1, sticky="w")
        self._bind_combobox_scroll(primary_picker)

        coloc_fields = tk.Frame(measurement_card, bg=CARD)
        coloc_fields.columnconfigure(0, minsize=190)

        def show_colocalization():
            if coloc_toggle.winfo_manager() and self._vars["ana_coloc_enabled"].get():
                coloc_fields.pack(fill="x", after=coloc_toggle, pady=(2, 0))
            else:
                coloc_fields.pack_forget()
            refresh_visible_review_status()

        coloc_toggle = tk.Checkbutton(
            measurement_card, text="Compare with another channel (colocalization)",
            variable=self._vars["ana_coloc_enabled"], command=show_colocalization,
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        )
        coloc_toggle.pack(anchor="w", pady=(6, 0))

        tk.Label(coloc_fields, text="Protein marker channel", bg=CARD, fg=FG,
                 font=FONT).grid(row=0, column=0, sticky="w", pady=3)
        marker_picker = ttk.Combobox(coloc_fields, textvariable=self._ana_pm_chan_var, width=16)
        marker_picker.grid(row=0, column=1, sticky="w", pady=3)
        self._bind_combobox_scroll(marker_picker)
        tk.Label(coloc_fields, text="Minimum overlap (IoU)", bg=CARD, fg=FG,
                 font=FONT).grid(row=1, column=0, sticky="w", pady=3)
        tk.Entry(coloc_fields, textvariable=self._vars["ana_iou_threshold"],
                 bg=BG, fg=FG, insertbackground=FG, relief="flat", font=FONT, width=8).grid(
            row=1, column=1, sticky="w", pady=3)
        tk.Label(coloc_fields, text="0 to 1", bg=CARD, fg=FG,
                 font=FONT).grid(row=1, column=2, sticky="w", padx=(8, 0))

        review_details = details(measurement_card, "Review status", self._refresh_ana_val_status)
        tk.Label(
            review_details, text="Analysis uses saved mask corrections and object decisions.",
            bg=CARD, fg=FG, font=FONT, wraplength=600, justify="left",
        ).pack(anchor="w")
        self._ana_val_status_frame = tk.Frame(review_details, bg=CARD)
        self._ana_val_status_frame.pack(fill="x", pady=(4, 0))
        ttk.Button(review_details, text="Refresh",
                   command=self._refresh_ana_val_status).pack(anchor="w", pady=(4, 0))

        def refresh_visible_review_status():
            if review_details.winfo_manager():
                self._refresh_ana_val_status()

        def save_channels(event=None):
            channels = primary_picker.cget("values")
            if len(channels) > 1:
                primary = self._ana_coi_chan_var.get()
                marker_options = tuple(ch for ch in channels if ch != primary)
                marker_picker.config(values=marker_options)
                if self._ana_pm_chan_var.get() not in marker_options:
                    self._ana_pm_chan_var.set(marker_options[0])
            for key, selection in (("ana_red_channel", self._ana_coi_chan_var),
                                   ("ana_pv_channel", self._ana_pm_chan_var)):
                channel = selection.get().removeprefix("Channel ").strip()
                if channel.isdigit():
                    self._vars[key].set(channel)
            if event is not None:
                refresh_visible_review_status()

        primary_picker.bind("<<ComboboxSelected>>", save_channels)
        marker_picker.bind("<<ComboboxSelected>>", save_channels)

        def refresh_input(*_):
            if not input_card.winfo_exists():
                return
            run = self._ana_input_var.get()
            if run:
                self.config["seg_input_subfolder"] = run
                self._pair_roi_masks_with_run(run)
                project = self.config.get("project_output_folder", "").strip()
                run_details.config(text=_segmentation_run_details(project, run))
            else:
                run_details.config(text="Choose the segmentation run you want to measure.")
            roi_help.config(text=self._roi_help_for_run(run))
            channels = tuple(f"Channel {ch}" for ch in self._detect_predicted_channels()) if run else ()
            for picker in (primary_picker, marker_picker):
                picker.config(values=channels, state="readonly" if channels else "disabled")
            if channels:
                if self._ana_coi_chan_var.get() not in channels:
                    self._ana_coi_chan_var.set(channels[0])
                if self._ana_pm_chan_var.get() not in channels:
                    self._ana_pm_chan_var.set(channels[-1])
                save_channels()
            if len(channels) == 1:
                self._vars["ana_coloc_enabled"].set(False)
            if len(channels) > 1:
                coloc_toggle.pack(anchor="w", pady=(6, 0), after=primary_row)
            else:
                coloc_toggle.pack_forget()
            show_colocalization()
            run_button.config(state="normal" if run else "disabled")

        run_card = self._card(parent, "Run Analysis")
        tk.Label(
            run_card, text="Saves an Excel summary and a CSV table for each image.",
            bg=CARD, fg=FG, font=FONT, wraplength=600, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        run_button = tk.Button(
            run_card, text="▶  Run Analysis", command=self._run_analysis_step,
            bg=GREEN, fg="#1e1e2e", font=FONT_BOLD, relief="flat", padx=18, pady=8,
        )
        run_button.pack(anchor="w")
        self._ana_status = tk.Label(run_card, bg=CARD, fg=FG, font=FONT,
                                    wraplength=600, justify="left")
        self._ana_status.pack(anchor="w")
        self._ana_progress = ttk.Progressbar(run_card, mode="indeterminate")

        output_options = details(run_card, "Output options")
        output_row = tk.Frame(output_options, bg=CARD)
        output_row.pack(anchor="w")
        tk.Label(output_row, text="Analysis subfolder name", bg=CARD, fg=FG,
                 font=FONT).pack(side="left", padx=(0, 10))
        tk.Entry(output_row, textvariable=self._vars["ana_subfolder_name"],
                 bg=BG, fg=FG, insertbackground=FG, relief="flat", font=FONT, width=24).pack(
            side="left")
        tk.Checkbutton(
            output_options, text="Save QC overlay images",
            variable=self._vars["ana_save_qc"],
            bg=CARD, fg=FG, selectcolor=CARD, activebackground=CARD, font=FONT,
        ).pack(anchor="w", pady=(4, 0))

        tk.Label(run_card, text="Analysis log", bg=CARD, fg=FG, font=FONT_BOLD).pack(
            anchor="w", pady=(10, 4))
        self._ana_log = scrolledtext.ScrolledText(
            run_card, bg="#0d0d15", fg="#a6e3a1",
            font=FONT_MONO, height=10, relief="flat", wrap="word",
            state="disabled",
        )
        self._ana_log.pack(fill="x")

        self._ana_input_var.trace_add("write", refresh_input)
        refresh_input()

    def _get_val_status_for_channel(self, project_root: str, channel: int) -> str:
        """Include focused mask-change review in the displayed status."""
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
        frame = getattr(self, "_ana_val_status_frame", None)
        if frame is None or not frame.winfo_exists():
            return
        for w in frame.winfo_children():
            w.destroy()

        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            tk.Label(frame, text="No project folder set.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        # Sample folders (with their "ROI Validation Results") live inside the
        # segmentation output subfolder — NOT directly under the project root.
        # Use the same segmentation-input selection the analysis run will use.
        seg_sub = self._ana_input_var.get()
        if not seg_sub:
            tk.Label(frame, text="Choose a segmentation run to see review progress.",
                     bg=CARD, fg=FG, font=FONT).pack(anchor="w")
            return
        scan_root = os.path.join(proj, seg_sub)
        if not os.path.isdir(scan_root):
            tk.Label(frame, text="No segmentation output found yet.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        channel_keys = ["ana_red_channel"]
        if self._vars["ana_coloc_enabled"].get():
            channel_keys.append("ana_pv_channel")
        channels_to_check = dict.fromkeys(
            int(value) for key in channel_keys
            if (value := self._vars[key].get()).isdigit()
        )

        # Collect per-channel aggregate status across all sample sub-folders
        project_dirs = [
            d for d in sorted(os.listdir(scan_root))
            if os.path.isdir(os.path.join(scan_root, d))
            and not d.startswith(".")
            and not d.startswith("DEBUG")
        ]

        if not channels_to_check:
            tk.Label(frame, text="Choose a measurement channel to see review progress.",
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
                        "decisions needing re-review are excluded"
                    )
                    fg = YELLOW
                elif n_complete == n_total:
                    label_text = (f"Channel {ch}: validation COMPLETE "
                                  f"({n_complete}/{n_total} {_img})  ✓")
                    fg = GREEN
                elif n_complete + n_partial > 0:
                    label_text = (f"Channel {ch}: PARTIAL — "
                                  f"{n_complete} complete  {n_partial} in-progress  {n_none} not started "
                                  f"(of {n_total} {_img})")
                    fg = YELLOW
                else:
                    label_text = f"Channel {ch}: not yet validated  –"
                    fg = ORANGE

                tk.Label(frame, text=label_text, bg=CARD, fg=fg, font=FONT_BOLD,
                         wraplength=600, justify="left").pack(anchor="w", pady=2)

    def _refresh_val_status(self) -> None:
        """Rebuild per-project validation progress labels in the validation step."""
        frame = getattr(self, "_val_prog_frame", None)
        if frame is None or not frame.winfo_exists():
            return
        for w in frame.winfo_children():
            w.destroy()

        proj = self.config.get("project_output_folder", "").strip()
        if not proj or not os.path.isdir(proj):
            tk.Label(frame, text="No project folder set.", bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            return

        channels_to_check = [int(ch) for ch in
                             self._detect_predicted_channels(fallback_to_config=False)]

        if not channels_to_check:
            tk.Label(frame, text="Review progress appears when predictions are available.",
                     bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
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
                self._vars["ana_red_channel"].set(ch)
        if self._ana_pm_chan_var:
            ch = self._ana_pm_chan_var.get().replace("Channel ", "").strip()
            if ch.isdigit():
                self.config["ana_pv_channel"] = ch
                self._vars["ana_pv_channel"].set(ch)
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
        if not self._confirm_legacy_roi_matches(ana_input):
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
            self._ana_progress.pack(fill="x", after=self._ana_status, pady=(4, 0))
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
                    [sys.executable, script, "--config", CONFIG_PATH, "--use-saved-reviews"],
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
                            self._ana_progress.pack_forget()
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
                            self._ana_progress.pack_forget()
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
                if matching_paths(os.path.join(proj, v), os.path.join("**", "*_predict_*.tif"),
                                  recursive=True):
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
                and any(prediction_channel(mask_pat) in (None, prediction_channel(path))
                        for path in matching_paths(os.path.join(seg_folder, d), mask_pat))
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

    def _build_batch_step(self, parent) -> None:
        self._section_header(
            parent, BATCH_TITLE, "Apply this project's segmentation and analysis settings to several image folders.",
        )
        project = self.config.get("project_output_folder", "").strip()
        if project != self._batch_project:
            self._batch_project = project
            self._batch_targets.clear()
        targets = self._batch_targets
        output_base = (Path(project) / BATCH_FOLDER).resolve() if project else None
        settings_issue = batch_settings_issue(self.config)

        warning_card = self._card(parent)
        tk.Label(
            warning_card, text="⚠  Check that your filters fit every folder", fg=YELLOW,
            bg=CARD, font=FONT_CARD_TITLE,
        ).pack(anchor="w", pady=(0, 6))
        tk.Label(
            warning_card,
            text="Stain, tissue, microscope settings, and magnification can change cell size "
                 "and intensity. Test the current model and filters on representative images "
                 "from each group before running a large batch.",
            bg=CARD, fg=FG, font=FONT, wraplength=600, justify="left",
        ).pack(anchor="w")

        settings_card = self._card(parent, "Settings from this project")
        if settings_issue:
            step, detail = settings_issue
            tk.Label(settings_card, text=detail, fg=YELLOW, bg=CARD, font=FONT,
                     wraplength=600, justify="left").pack(anchor="w")
            ttk.Button(settings_card, text=f"Go to Step {step + 1}",
                       command=lambda: self._show_step(step)).pack(anchor="w", pady=(6, 0))
        else:
            cfg = self.config
            backend = cfg.get("seg_backend")
            model = {
                "microsam": cfg.get("seg_model_type"),
                "microsam2": cfg.get("seg_microsam2_model"),
                "cellpose": cfg.get("seg_cellpose_variant"),
                "stardist": "2D versatile fluorescence",
            }[backend]
            model_path = cfg.get(SEGMENTATION_CHECKPOINT_KEYS.get(backend, ""), "")
            if model_path:
                model = os.path.basename(os.path.normpath(model_path))
            analysis = f"Measure channel {0 if is_brightfield(cfg) else cfg.get('ana_red_channel')}"
            if (not is_brightfield(cfg) and cfg.get("ana_coloc_enabled")
                    and len({part.strip() for part in cfg["seg_channels"].split(",") if part.strip()}) > 1):
                analysis += (f" · compare with channel {cfg.get('ana_pv_channel')}"
                             f" · minimum IoU {cfg.get('ana_iou_threshold')}")
            text = (
                f"{cfg.get('project_name') or Path(project).name}\n"
                f"{SEGMENTATION_BACKENDS[backend]} · {model} · predict channel(s) {cfg['seg_channels']}\n"
                f"Size {cfg.get('seg_min_object_size')}–{cfg.get('seg_max_object_size')} px"
                f" · Intensity {cfg.get('seg_raw_mean_min')}–{cfg.get('seg_raw_mean_max')}\n"
                f"{analysis}"
            )
            if cfg.get("seg_per_channel_filters_enabled"):
                text += "\nChannel-specific filters are enabled."
            tk.Label(settings_card, text=text, bg=CARD, fg=FG, font=FONT,
                     wraplength=600, justify="left").pack(anchor="w")

        folders_card = self._card(parent, "Image folders")
        tk.Label(
            folders_card,
            text=("Add folders of exported slide TIFFs. Set ROI masks for each folder as needed."
                  if self.config.get("input_type") == "vsi" else
                  "Add folders that use the template project's input format and channel order."),
            bg=CARD, fg=FG, font=FONT, wraplength=600, justify="left",
        ).pack(anchor="w", pady=(0, 6))
        table = tk.Frame(folders_card, bg=CARD)
        table.pack(fill="x", pady=(0, 6))
        table.columnconfigure(0, weight=1)
        tree = ttk.Treeview(table, show="headings", columns=("folder", "masks", "status"), height=6)
        for key, label, width in (("folder", "Image folder", 330), ("masks", "ROI masks", 120),
                                  ("status", "Status", 110)):
            tree.heading(key, text=label)
            tree.column(key, width=width, minwidth=85, stretch=key == "folder")
        tree.grid(row=0, column=0, sticky="ew")
        scroll_y = ttk.Scrollbar(table, orient="vertical", command=tree.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x = ttk.Scrollbar(table, orient="horizontal", command=tree.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")
        tree.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        def redraw():
            tree.delete(*tree.get_children())
            for folder, masks in targets.items():
                tree.insert("", "end", iid=str(folder),
                            values=(str(folder), masks.name if masks else "Whole image", "Ready"))
            sync_buttons()

        def add_folders(folders):
            skipped = []
            for value in folders:
                try:
                    folder = batch_input_folder(value, self.config)
                    masks = automatic_batch_masks(folder)
                    problem = batch_target_problem(folder, masks, self.config)
                except OSError as exc:
                    skipped.append((Path(value), f"Could not read input folders. {exc}"))
                    continue
                if output_base and (folder == output_base or folder in output_base.parents
                                    or output_base in folder.parents):
                    problem = "Choose an image folder outside the batch output directory."
                if not problem and not self.config.get("seg_skip_subdirs"):
                    if any(other != folder and (folder in other.parents or other in folder.parents)
                           for other in targets):
                        problem = "This folder overlaps an image folder already queued."
                if problem:
                    skipped.append((folder, problem))
                else:
                    targets.setdefault(folder, masks)
            redraw()
            if skipped:
                detail = f"{skipped[0][0].name} · {skipped[0][1]}"
                if len(skipped) > 1:
                    detail = f"{len(skipped)} folders skipped. {detail}"
                queue_status.config(text=detail, fg=YELLOW)
            else:
                queue_status.config(text=f"{len(targets)} folder(s) queued.", fg=FG)

        def add_one():
            folder = filedialog.askdirectory(parent=self.root, title="Choose an image folder")
            if folder:
                add_folders([folder])

        def add_subfolders():
            folder = filedialog.askdirectory(parent=self.root, title="Choose a parent of image folders")
            if folder:
                try:
                    children = sorted(p for p in Path(folder).iterdir()
                                      if p.is_dir() and not p.name.startswith("."))
                    if children:
                        add_folders(children)
                    else:
                        queue_status.config(text="This folder has no subfolders.", fg=YELLOW)
                except OSError as exc:
                    queue_status.config(text=str(exc), fg=YELLOW)

        def remove_selected():
            for key in tree.selection():
                targets.pop(Path(key), None)
            redraw()
            queue_status.config(text=f"{len(targets)} folder(s) queued.", fg=FG)

        def set_masks():
            keys = tree.selection()
            if len(keys) != 1:
                return
            mask_folder = filedialog.askdirectory(parent=self.root, title="Choose ROI masks for this folder")
            if mask_folder:
                folder = Path(keys[0])
                masks = Path(mask_folder).resolve()
                problem = batch_target_problem(folder, masks, self.config)
                if problem:
                    queue_status.config(text=problem, fg=YELLOW)
                    return
                targets[folder] = masks
                redraw()
                queue_status.config(text=f"Using ROI masks from {masks}.", fg=FG)

        def whole_image():
            for key in tree.selection():
                targets[Path(key)] = None
            redraw()
            queue_status.config(text="Selected folders will use whole-image analysis.", fg=FG)

        toolbar = tk.Frame(folders_card, bg=CARD)
        toolbar.pack(anchor="w")
        add_btn = ttk.Button(toolbar, text="Add Folder…", command=add_one)
        sub_btn = ttk.Button(toolbar, text="Add Subfolders…", command=add_subfolders)
        remove_btn = ttk.Button(toolbar, text="Remove", command=remove_selected)
        for button in (add_btn, sub_btn, remove_btn):
            button.pack(side="left", padx=(0, 6))
        mask_row = tk.Frame(folders_card, bg=CARD)
        mask_row.pack(anchor="w", pady=(4, 0))
        mask_btn = ttk.Button(mask_row, text="Set ROI Masks…", command=set_masks)
        whole_btn = ttk.Button(mask_row, text="Use Whole Image", command=whole_image)
        for button in (mask_btn, whole_btn):
            button.pack(side="left", padx=(0, 6))
        queue_status = tk.Label(folders_card, bg=CARD, fg=FG, font=FONT,
                                wraplength=600, justify="left")
        queue_status.pack(anchor="w", pady=(4, 0))

        run_card = self._card(parent, "Run batch")
        if output_base:
            tk.Label(run_card, text=f"Results folder · {output_base}", bg=CARD, fg=FG,
                     font=FONT, wraplength=600, justify="left").pack(anchor="w", pady=(0, 6))
        run_row = tk.Frame(run_card, bg=CARD)
        run_row.pack(anchor="w")
        run_button = tk.Button(
            run_row, text="▶  Run Batch", command=lambda: start_batch(),
            bg=GREEN, fg=BG, font=FONT_BOLD, relief="flat", padx=12, pady=7,
        )
        run_button.pack(side="left")
        cancel_button = ttk.Button(run_row, text="Cancel Batch", command=lambda: cancel_batch())
        cancel_button.pack(side="left", padx=8)
        progress = ttk.Progressbar(run_card, mode="determinate")
        progress.pack(fill="x", pady=(8, 0))
        run_status = tk.Label(run_card, bg=CARD, fg=FG, font=FONT, wraplength=600, justify="left")
        run_status.pack(anchor="w", pady=(4, 0))
        tk.Label(run_card, text="Batch log", bg=CARD, fg=FG, font=FONT_BOLD).pack(
            anchor="w", pady=(6, 4))
        log_box = scrolledtext.ScrolledText(
            run_card, bg="#0d0d15", fg=GREEN, font=FONT_MONO, height=10, wrap="word", state="disabled",
        )
        log_box.pack(fill="x")

        def sync_buttons(_event=None):
            running = self._batch_stop is not None
            selected = len(tree.selection())
            add_btn.config(state="disabled" if running else "normal")
            sub_btn.config(state="disabled" if running else "normal")
            remove_btn.config(state="normal" if selected and not running else "disabled")
            whole_btn.config(state="normal" if selected and not running else "disabled")
            mask_btn.config(state="normal" if selected == 1 and not running else "disabled")
            run_button.config(state="normal" if targets and not running and not settings_issue else "disabled")
            cancel_button.config(state="normal" if running else "disabled")

        def append_log(line):
            if log_box.winfo_exists():
                log_box.config(state="normal")
                log_box.insert("end", line + "\n")
                log_box.see("end")
                log_box.config(state="disabled")

        def cancel_batch():
            if self._batch_stop is not None and messagebox.askyesno(
                    "Cancel batch?", "Stop the current folder and keep the results already written?",
                    parent=self.root, default=messagebox.NO):
                self._terminate_running()
                run_status.config(text="Cancelling…", fg=YELLOW)
                cancel_button.config(state="disabled")

        def start_batch():
            if self._batch_stop is not None or self._running_procs() or not targets:
                return
            self._sync_vars_to_config()
            issue = batch_settings_issue(self.config)
            if issue:
                run_status.config(text=f"Step {issue[0] + 1} · {issue[1]}", fg=YELLOW)
                return
            try:
                batch_folder = create_batch_folder(output_base)
            except OSError as exc:
                run_status.config(text=f"Could not create batch results folder. {exc}", fg=RED)
                return
            config = dict(self.config)
            queue = list(targets.items())
            stop = threading.Event()
            self._batch_stop = stop
            progress.config(maximum=len(queue), value=0)
            run_status.config(text="Starting batch…", fg=ACCENT)
            log_box.config(state="normal")
            log_box.delete("1.0", "end")
            log_box.config(state="disabled")
            for folder, _ in queue:
                tree.set(str(folder), "status", "Queued")
            sync_buttons()

            def update(index, status, detail):
                if not tree.winfo_exists():
                    return
                tree.set(str(queue[index][0]), "status", status.capitalize())
                if status not in ("queued", "segmenting", "analyzing"):
                    progress.config(value=float(progress.cget("value")) + 1)
                run_status.config(text=f"{index + 1}/{len(queue)} · {status.capitalize()}"
                                       + (f" · {detail}" if detail else ""), fg=FG)

            def finished(text, color):
                if self._batch_stop is stop:
                    self._batch_stop = None
                if run_status.winfo_exists():
                    run_status.config(text=text, fg=color)
                    sync_buttons()

            def worker():
                try:
                    with (batch_folder / "batch.log").open("w", encoding="utf-8") as saved_log:
                        def log(line):
                            saved_log.write(line + "\n")
                            saved_log.flush()
                            self.root.after(0, append_log, line)

                        def run_stage(command, env):
                            environment = dict(env or os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                            with self._batch_lock:
                                if stop.is_set():
                                    return 1
                                proc = subprocess.Popen(
                                    command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                                )
                                self._track_proc(proc, BATCH_TITLE)
                            with proc.stdout:
                                for line in proc.stdout:
                                    log(line.rstrip())
                            return proc.wait()

                        log(f"Saving batch results in {batch_folder}")
                        rows = run_batch(config, queue, batch_folder, run_stage=run_stage,
                                         stopped=stop.is_set, log=log,
                                         on_update=lambda *args: self.root.after(0, update, *args))
                        complete = sum(row["status"] == "complete" for row in rows)
                        warnings = sum(row["status"] == "warnings" for row in rows)
                        failed = sum(row["status"] == "failed" for row in rows)
                        skipped = len(rows) - complete - warnings - failed
                        outcome = "Batch cancelled" if stop.is_set() else "Batch finished"
                        text = f"{outcome} · {complete} complete, {warnings} with warnings, {failed} failed, {skipped} cancelled or not run."
                        log(text)
                        log(f"Report: {batch_folder / 'batch_report.json'}")
                        self.root.after(0, finished, text, YELLOW if warnings or failed or skipped else GREEN)
                except Exception as exc:
                    self.root.after(0, append_log, f"Batch error · {exc}")
                    self.root.after(0, finished, f"Batch could not finish. {exc}", RED)

            threading.Thread(target=worker, daemon=True).start()

        tree.bind("<<TreeviewSelect>>", sync_buttons)
        redraw()

    # ==================================================================
    # Launch actions
    # ==================================================================

    def _build_retraining_step(self, parent) -> None:
        button_fg = "#1e1e2e" if sys.platform == "darwin" else FG
        self._section_header(
            parent, 6,
            "Fine-tune a model with saved mask corrections or completed object reviews.")
        card = self._card(parent, "Training data and model")
        tk.Label(card, text=(
            "Use saved mask corrections or completed cell reviews from Step 4. "
            "Complete individual cell review to train specific labels such as YES or NO."),
            bg=CARD, fg=MUTED, font=FONT, wraplength=720,
            justify="left").pack(anchor="w", pady=(0, 10))
        variants = self._find_variants(SUBFOLDER["seg"])
        if not variants:
            tk.Label(card, text="No segmentation runs found in this project.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w")

        run_var = tk.StringVar(value=(self.config.get("seg_input_subfolder")
                                     if self.config.get("seg_input_subfolder") in variants
                                     else variants[-1] if variants else ""))
        training_backends = ("microsam", "cellpose", "stardist")
        segmentation_backend = self.config.get("seg_backend", "microsam")
        backend_var = tk.StringVar(value=(segmentation_backend
                                         if segmentation_backend in training_backends else "microsam"))
        channels = str(self.config.get("seg_channels") or "1").split(",")
        channel_var = tk.StringVar(value=channels[0].strip())
        model_name_date = datetime.now().strftime("%Y-%m-%d")
        default_model_name = [f"{backend_var.get()} ch{channel_var.get()} {model_name_date}"]
        model_name_var = tk.StringVar(value=default_model_name[0])
        def refresh_default_model_name(*_):
            replacement = f"{backend_var.get()} ch{channel_var.get()} {model_name_date}"
            if model_name_var.get() == default_model_name[0]:
                model_name_var.set(replacement)
            default_model_name[0] = replacement
        backend_var.trace_add("write", refresh_default_model_name)
        channel_var.trace_add("write", refresh_default_model_name)
        epochs_var = tk.StringVar(value="20")
        preprocess_var = tk.StringVar(value=self.config.get("seg_preprocess_mode")
                                     or "local_bg_subtract")
        base_var = tk.StringVar()
        use_var = tk.BooleanVar(value=False)
        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x")
        grid.grid_columnconfigure(1, weight=1)

        def row(number, label, widget):
            tk.Label(grid, text=label, bg=CARD, fg=FG, font=FONT,
                     anchor="w", width=22).grid(row=number, column=0,
                                                 sticky="w", pady=4)
            widget.grid(row=number, column=1, sticky="ew", padx=8, pady=4)

        row(0, "Masks to train from:", ttk.Combobox(
            grid, textvariable=run_var, values=variants, state="readonly"))
        channel_box = ttk.Combobox(grid, textvariable=channel_var,
                                   state="readonly")
        row(1, "Channel to train:", channel_box)
        self._bind_combobox_scroll(channel_box)
        row(2, "Model family to train:", ttk.Combobox(
            grid, textvariable=backend_var,
            values=training_backends, state="readonly"))
        starting_choice = tk.StringVar()
        starting_options = {}
        starting_keys = {"microsam": "seg_custom_model_path",
                         "cellpose": "seg_cellpose_model_path",
                         "stardist": "seg_stardist_model_path"}
        selected_by_backend = {kind: str(self.config.get(key) or "")
                               for kind, key in starting_keys.items()}
        starting_box = ttk.Combobox(grid, textvariable=starting_choice,
                                    state="readonly")
        row(3, "Initialize weights from:", starting_box)
        self._bind_combobox_scroll(starting_box)
        row(4, "Save new model as:", tk.Entry(grid, textvariable=model_name_var,
                                             bg=BG, fg=FG,
                                             insertbackground=FG, relief="flat"))
        tk.Label(card, text=("Saved mask corrections or completed cell reviews supply the teaching examples. "
                             "The masks can come from "
                             "Micro-SAM, micro-SAM2, Cellpose, or StarDist. "
                             "Initialize weights from chooses the starting model within the family "
                             "you will train. Your named result is saved separately in the shared library."),
                 bg=CARD, fg=MUTED, font=FONT, wraplength=750,
                 justify="left").pack(anchor="w", pady=(5, 2))
        from .retraining import model_library_root
        library_row = tk.Frame(card, bg=CARD)
        library_row.pack(fill="x", pady=(3, 5))
        tk.Label(library_row, text=f"Model library: {model_library_root()}",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                 anchor="w").pack(side="left", fill="x", expand=True)
        def open_model_library():
            folder = model_library_root()
            try:
                folder.mkdir(parents=True, exist_ok=True)
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(folder)])
                elif sys.platform == "win32":
                    os.startfile(str(folder))
                else:
                    subprocess.Popen(["xdg-open", str(folder)])
            except (OSError, ValueError) as exc:
                messagebox.showerror("Model library", f"Could not open {folder}: {exc}",
                                     parent=self.root)
        tk.Button(library_row, text="Open Model Library",
                  command=open_model_library, bg=ACTIVE, fg=button_fg,
                  font=FONT, relief="flat", padx=8).pack(side="right")

        def refresh_channels(*_):
            from .retraining import available_channels
            project = self.config.get("project_output_folder", "")
            found = available_channels(os.path.join(project, run_var.get()))
            values = [str(v) for v in found]
            channel_box.configure(values=values)
            if channel_var.get() not in values:
                channel_var.set(values[0] if values else "")
        run_var.trace_add("write", refresh_channels)
        refresh_channels()

        advanced_card = tk.Frame(card, bg=CARD)
        advanced_grid = tk.Frame(advanced_card, bg=CARD)
        advanced_grid.pack(fill="x")
        advanced_grid.grid_columnconfigure(1, weight=1)
        tk.Label(advanced_grid, text="Training epochs:", bg=CARD, fg=FG,
                 font=FONT, anchor="w", width=22).grid(row=0, column=0, sticky="w", pady=3)
        tk.Entry(advanced_grid, textvariable=epochs_var, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat", width=14).grid(
                     row=0, column=1, sticky="w", pady=3)
        tk.Label(advanced_grid, text="Preprocessing:", bg=CARD, fg=FG,
                 font=FONT, anchor="w", width=22).grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(advanced_grid, textvariable=preprocess_var,
                     values=("local_bg_subtract", "local_bg_subtract_mild",
                             "raw_percentile", "tile_clahe_mild", "tile_clahe",
                             "brightfield_invert"), state="readonly", width=24).grid(
                                 row=1, column=1, sticky="w", pady=3)
        advanced_vars = {key: tk.StringVar(value="") for key in
                         ("learning_rate", "batch_size", "patch_size")}
        sampling_focus_var = tk.StringVar(value="none")
        from .retraining import TRAINING_PRESETS
        preset_var = tk.StringVar(value="Default training")
        preset_card = tk.Frame(card, bg=CARD)
        tk.Label(preset_card, text="Training preset", bg=CARD, fg=FG,
                 font=FONT_BOLD).pack(anchor="w", pady=(0, 3))
        for preset_name, recipe in TRAINING_PRESETS.items():
            choice = tk.Frame(preset_card, bg=CARD)
            choice.pack(fill="x", pady=2)
            tk.Radiobutton(choice, text=preset_name, variable=preset_var,
                           value=preset_name, command=lambda: apply_preset(),
                           bg=CARD, fg=FG, selectcolor=BG,
                           activebackground=CARD, font=FONT).pack(anchor="w")
            tk.Label(choice, text=recipe["description"], bg=CARD, fg=MUTED,
                     font=("Segoe UI", 8), wraplength=730,
                     justify="left").pack(anchor="w", padx=(22, 0))
        preset_card.pack(fill="x", pady=(12, 3))

        def apply_preset(*_):
            recipe = TRAINING_PRESETS[preset_var.get()]
            epochs_var.set(str(recipe["epochs"]))
            rate = recipe["learning_rate"]
            if backend_var.get() == "stardist":
                rate = 1e-4 if preset_var.get().endswith("follow-up") else ""
            advanced_vars["learning_rate"].set(str(rate) if rate != "" else "")
            sampling_focus_var.set(recipe["sampling_focus"])

        backend_var.trace_add("write", apply_preset)
        apply_preset()
        advanced_labels = (
            ("Learning rate:", "learning_rate", "default: Cellpose/Micro-SAM 1e-5; StarDist model default"),
            ("Batch size:", "batch_size", "blank = model default"),
            ("Patch size (px):", "patch_size", "blank = model default; multiple of 16"),
        )
        advanced_rows = {}
        for index, (caption, key, hint) in enumerate(advanced_labels):
            label = tk.Label(advanced_grid, text=caption, bg=CARD, fg=FG,
                             font=FONT, anchor="w", width=22)
            label.grid(row=index + 2, column=0, sticky="w", pady=3)
            field = tk.Frame(advanced_grid, bg=CARD)
            field.grid(row=index + 2, column=1, sticky="ew", pady=3)
            advanced_rows[key] = (label, field)
            tk.Entry(field, textvariable=advanced_vars[key], bg=BG, fg=FG,
                     insertbackground=FG, relief="flat", width=14).pack(side="left")
            tk.Label(field, text=hint, bg=CARD, fg=MUTED,
                     font=("Segoe UI", 8)).pack(side="left", padx=8)
        focus_row = tk.Frame(advanced_card, bg=CARD)
        focus_row.pack(fill="x", pady=(5, 0))
        tk.Label(focus_row, text="Training sample focus:", bg=CARD, fg=FG,
                 font=FONT, width=22, anchor="w").pack(side="left")
        ttk.Combobox(focus_row, textvariable=sampling_focus_var,
                     values=("none", "missed_cells", "rejected_cells"),
                     state="readonly", width=18).pack(side="left", padx=8)
        tk.Label(advanced_card,
                 text="Focus repeats only matching training samples; validation samples remain separate.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                 wraplength=720, justify="left").pack(anchor="w", padx=(22, 0))
        manual_path_var = tk.StringVar(value="")
        updating_starting = [False]
        manual_row = tk.Frame(advanced_card, bg=CARD)
        manual_row.pack(fill="x", pady=(8, 2))
        tk.Label(manual_row, text="Other starting model:", bg=CARD, fg=FG,
                 font=FONT, width=22, anchor="w").pack(side="left")
        tk.Entry(manual_row, textvariable=manual_path_var, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", fill="x",
                                                         expand=True, padx=8)
        def browse_base():
            if backend_var.get() == "stardist":
                selected = filedialog.askdirectory(parent=self.root,
                                                   title="Choose a StarDist model folder")
            else:
                selected = filedialog.askopenfilename(parent=self.root,
                                                     title="Choose a model checkpoint")
            if selected:
                manual_path_var.set(selected)
        tk.Button(manual_row, text="Browse…", command=browse_base,
                  bg=ACTIVE, fg=button_fg, font=FONT, relief="flat").pack(side="left", padx=6)
        tk.Label(advanced_card, text=("Choose a model file or folder here only if it is "
                                      "not already in the dropdown above."),
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")

        def open_model_manager():
            from .retraining import delete_managed_model, list_trained_models
            manager = tk.Toplevel(self.root)
            manager.title("CellMatchR Model Manager")
            manager.geometry("1020x480")
            manager.configure(bg=BG)
            tk.Label(manager, text="Saved fine-tuned models", bg=BG, fg=FG,
                     font=FONT_TITLE).pack(anchor="w", padx=18, pady=(16, 4))
            tk.Label(manager, text=("Shared models are available in every project. "
                                    "Older project models are shown when that project is open."),
                     bg=BG, fg=MUTED, font=FONT,
                     wraplength=960, justify="left").pack(anchor="w", padx=18, pady=(0, 10))
            columns = ("name", "backend", "channel", "created", "rounds",
                       "epochs", "location")
            table_frame = tk.Frame(manager, bg=BG)
            table_frame.pack(fill="both", expand=True, padx=18)
            table = ttk.Treeview(table_frame, columns=columns, show="headings",
                                 selectmode="browse")
            headings = {"name": "Model name", "backend": "Model", "channel": "Channel",
                        "created": "Created", "rounds": "Training rounds",
                        "epochs": "Recorded epochs", "location": "Location"}
            widths = {"name": 230, "backend": 90, "channel": 65,
                      "created": 155, "rounds": 110, "epochs": 120,
                      "location": 160}
            for column in columns:
                table.heading(column, text=headings[column])
                table.column(column, width=widths[column], anchor="w")
            table.pack(side="left", fill="both", expand=True)
            scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
            scrollbar.pack(side="right", fill="y")
            table.configure(yscrollcommand=scrollbar.set)
            detail = tk.Label(manager, text="Select a model to see its folder and training history.",
                              bg=BG, fg=MUTED, font=("Segoe UI", 9),
                              wraplength=960, justify="left", anchor="w")
            detail.pack(fill="x", padx=18, pady=(8, 3))
            manager_status = tk.Label(manager, text="", bg=BG, fg=FG,
                                      font=FONT, anchor="w")
            manager_status.pack(fill="x", padx=18)
            buttons = tk.Frame(manager, bg=BG)
            buttons.pack(fill="x", padx=18, pady=(8, 14))
            records = {}
            project = self.config.get("project_output_folder", "")

            def reload_models():
                records.clear()
                for iid in table.get_children():
                    table.delete(iid)
                for index, record in enumerate(list_trained_models(project, include_missing=True)):
                    iid = str(index)
                    records[iid] = record
                    rounds = record.get("training_rounds", 1)
                    epochs = record.get("recorded_epochs", record.get("epochs"))
                    if epochs is None:
                        epochs_text = "Unknown"
                    else:
                        epochs_text = str(epochs)
                        if not record.get("history_complete", True):
                            epochs_text += "+ known"
                    table.insert("", "end", iid=iid, values=(
                        record["name"], record["backend"], record["channel"],
                        str(record.get("completed_at") or "Unknown (older model)").replace("T", " "),
                        rounds, epochs_text,
                        record["location"] + (" (file missing)" if not record["available"] else "")))
                manager_status.config(text=f"{len(records)} model(s) listed.", fg=FG)

            def show_selected(_event=None):
                selected = table.selection()
                if not selected:
                    return
                record = records[selected[0]]
                origin = record.get("starting_model_name") or record.get("base_model") or "Unknown"
                detail.config(text=(f"Folder: {record['record_dir']}\n"
                                    f"Starting model: {origin}"))

            def delete_selected():
                selected = table.selection()
                if not selected:
                    manager_status.config(text="Select a model first.", fg=YELLOW)
                    return
                if self._is_running("Model Retraining"):
                    manager_status.config(text="Finish the current training run before deleting models.",
                                          fg=YELLOW)
                    return
                record = records[selected[0]]
                name = record["name"]
                folder = record["record_dir"]
                if not messagebox.askyesno(
                        "Delete model — first confirmation",
                        f"Permanently delete '{name}' and its saved model folder?\n\n"
                        f"{folder}\n\nProjects using it will need another model. "
                        "Models trained from it remain available.",
                        parent=manager, default=messagebox.NO, icon="warning"):
                    return
                typed = simpledialog.askstring(
                    "Delete model — second confirmation",
                    f"Type the exact model name to permanently delete it:\n{name}",
                    parent=manager)
                if typed != name:
                    manager_status.config(text="Deletion cancelled: model name did not match.",
                                          fg=YELLOW)
                    return
                try:
                    delete_managed_model(folder, project)
                    key = starting_keys[record["backend"]]
                    if (self.config.get(key) == record["model_path"] or
                            self._vars[key].get() == record["model_path"]):
                        self.config[key] = ""
                        self._vars[key].set("")
                        save_config(self.config)
                        save_project_config(self.config)
                    if selected_by_backend[record["backend"]] == record["model_path"]:
                        selected_by_backend[record["backend"]] = ""
                    refresh_starting_models()
                    reload_models()
                    detail.config(text="Select a model to see its folder and training history.")
                    manager_status.config(text=f"Deleted {name}.", fg=GREEN)
                except (OSError, ValueError) as exc:
                    manager_status.config(text=f"Could not delete model: {exc}", fg=RED)

            table.bind("<<TreeviewSelect>>", show_selected)
            tk.Button(buttons, text="Refresh", command=reload_models,
                      bg=ACTIVE, fg=button_fg, font=FONT, relief="flat", padx=12).pack(side="left")
            tk.Button(buttons, text="Delete Selected Model…", command=delete_selected,
                      bg=RED, fg="#1e1e2e", font=FONT_BOLD,
                      relief="flat", padx=12).pack(side="right")
            reload_models()

        tk.Button(advanced_card, text="Model Manager…", command=open_model_manager,
                  bg=ACTIVE, fg=button_fg, font=FONT_BOLD, relief="flat",
                  padx=10, pady=5).pack(anchor="w", pady=(10, 3))

        def manual_path_changed(*_):
            if updating_starting[0]:
                return
            path = manual_path_var.get().strip()
            if path:
                starting_options["Other model from file/folder"] = path
                starting_box.configure(values=list(starting_options))
                starting_choice.set("Other model from file/folder")
                base_var.set(path)
                selected_by_backend[backend_var.get()] = path
            elif starting_choice.get() == "Other model from file/folder":
                starting_options.pop("Other model from file/folder", None)
                starting_box.configure(values=list(starting_options))
                first = next(iter(starting_options), "")
                starting_choice.set(first)
                base_var.set(starting_options.get(first, ""))
                selected_by_backend[backend_var.get()] = base_var.get()
        manual_path_var.trace_add("write", manual_path_changed)
        loss_row = tk.Frame(advanced_card, bg=CARD)
        tk.Label(loss_row, text="StarDist distance loss:", bg=CARD, fg=FG,
                 font=FONT, width=22, anchor="w").pack(side="left")
        loss_var = tk.StringVar(value="Model default")
        ttk.Combobox(loss_row, textvariable=loss_var,
                     values=("Model default", "mae", "mse"),
                     state="readonly", width=18).pack(side="left", padx=8)
        freeze_var = tk.BooleanVar(value=True)
        freeze_row = tk.Checkbutton(
            advanced_card, text="Micro-SAM: keep image encoder frozen (current default)",
            variable=freeze_var, bg=CARD, fg=FG, selectcolor=BG,
            activebackground=CARD, font=FONT)
        tk.Label(advanced_card, text=("Loss selection is available for StarDist distance loss. "
                                      "Cellpose and Micro-SAM use their built-in loss functions."),
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                 wraplength=750, justify="left").pack(anchor="w", pady=(6, 0))
        def show_backend_advanced(*_):
            loss_row.pack_forget()
            freeze_row.pack_forget()
            for widget in advanced_rows["patch_size"]:
                (widget.grid_remove() if backend_var.get() == "cellpose"
                 else widget.grid())
            if backend_var.get() == "stardist":
                loss_row.pack(anchor="w", pady=3)
            elif backend_var.get() == "microsam":
                freeze_row.pack(anchor="w", pady=3)
        backend_var.trace_add("write", show_backend_advanced)
        show_backend_advanced()
        def toggle_advanced():
            if advanced_card.winfo_manager():
                advanced_card.pack_forget()
                advanced_button.config(text="Show advanced settings ▸")
            else:
                advanced_card.pack(fill="x", pady=(4, 10), after=advanced_button)
                advanced_button.config(text="Hide advanced settings ▾")
        advanced_button = tk.Button(card, text="Show advanced settings ▸",
                                    command=toggle_advanced, bg=ACTIVE, fg=button_fg,
                                    font=FONT, relief="flat", padx=10, pady=5)
        advanced_button.pack(anchor="w", pady=(7, 0))

        def refresh_starting_models(*_args, prefer_path=None):
            from .retraining import list_trained_models
            backend = backend_var.get()
            builtins = ({"microsam": ("vit_b_lm", "vit_b_histopathology", "vit_l_lm", "vit_h_lm",
                                     "vit_b", "vit_l", "vit_h"),
                         "cellpose": ("cpsam",),
                         "stardist": ("2D_versatile_fluo",)})[backend]
            starting_options.clear()
            for name in builtins:
                starting_options[f"Built-in: {name}"] = name
            for record in list_trained_models(
                    self.config.get("project_output_folder", ""), backend):
                label = (f"{record['name']}  (ch{record['channel']}, "
                         f"{record['location']}, {record['record_folder'][-15:]})")
                starting_options[label] = record["model_path"]
            default = (self.config.get("seg_model_type") or "vit_b_lm") if backend == "microsam" else builtins[0]
            current = prefer_path or selected_by_backend[backend] or default
            label = next((text for text, path in starting_options.items()
                          if path == current), None)
            if label is None:
                label = "Other model from file/folder"
                starting_options[label] = current
            starting_box.configure(values=list(starting_options))
            starting_choice.set(label)
            base_var.set(current)
            selected_by_backend[backend] = current
            updating_starting[0] = True
            manual_path_var.set(current if label == "Other model from file/folder" else "")
            updating_starting[0] = False

        def choose_starting_model(_event=None):
            choice = starting_choice.get()
            value = starting_options.get(choice, "")
            base_var.set(value)
            selected_by_backend[backend_var.get()] = value
            updating_starting[0] = True
            manual_path_var.set(value if choice == "Other model from file/folder" else "")
            updating_starting[0] = False
        starting_box.bind("<<ComboboxSelected>>", choose_starting_model)
        backend_var.trace_add("write", refresh_starting_models)
        refresh_starting_models()

        def refresh_preprocess(*_):
            from .retraining import PREPROCESS_MODES

            path = os.path.join(self.config.get("project_output_folder", ""),
                                run_var.get(), "_run_progress.json")
            try:
                with open(path) as source:
                    identity = json.load(source).get("run_identity", {})
                if identity.get("preprocess") in PREPROCESS_MODES:
                    preprocess_var.set(identity["preprocess"])
            except (OSError, ValueError, TypeError):
                pass
        run_var.trace_add("write", refresh_preprocess)
        refresh_preprocess()

        tk.Checkbutton(card, text=("Select the new model for this project's next segmentation runs "
                                       "(available in every project)"),
                       variable=use_var, bg=CARD, fg=FG, selectcolor=BG,
                       activebackground=CARD, font=FONT).pack(anchor="w", pady=(8, 4))

        # An imported project is a source of corrected masks, not a place where
        # training writes results. The current project remains the report home.
        sources_card = self._card(parent, "Training projects")
        tk.Label(sources_card, text=("The current project is included automatically. "
                                     "Add other projects with corrected masks or completed cell reviews."),
                 bg=CARD, fg=MUTED, font=FONT, wraplength=730,
                 justify="left").pack(anchor="w")
        imported_rows = []
        imported_frame = tk.Frame(sources_card, bg=CARD)
        imported_frame.pack(fill="x", pady=(6, 2))
        gallery_state = {"examples": [], "sources": "", "generation": 0,
                         "excluded": set(), "photos": [], "choosing": False,
                         "unfinished": [], "unavailable": []}
        start_button_ref = [None]

        def source_specs():
            from .retraining import available_channels
            result = []
            current = self.config.get("project_output_folder", "").strip()
            if current and run_var.get() and channel_var.get():
                folder = os.path.join(current, run_var.get())
                if int(channel_var.get()) in available_channels(folder):
                    result.append({"project": current, "name":
                                   self.config.get("project_name") or os.path.basename(current),
                                   "seg_run": run_var.get(), "channel": int(channel_var.get())})
            for entry in imported_rows:
                file = entry["file"].get().strip()
                if file:
                    if not os.path.isfile(file):
                        raise ValueError(f"Project file not found: {file}")
                    if not entry["run"].get() or not entry["channel"].get():
                        raise ValueError(f"Choose a run and channel for {file}")
                    result.append({"project": os.path.dirname(os.path.abspath(file)),
                                   "name": os.path.basename(os.path.dirname(file)),
                                   "seg_run": entry["run"].get(),
                                   "channel": int(entry["channel"].get())})
            if not result:
                raise ValueError("Choose a segmentation run in this project or add another project")
            return result

        def add_project_row():
            row_frame = tk.Frame(imported_frame, bg=CARD)
            row_frame.pack(fill="x", pady=3)
            file_var = tk.StringVar()
            run_choice = tk.StringVar()
            channel_choice = tk.StringVar()
            entry = {"frame": row_frame, "file": file_var,
                     "run": run_choice, "channel": channel_choice, "runs": []}
            imported_rows.append(entry)
            path_row = tk.Frame(row_frame, bg=CARD)
            path_row.pack(fill="x")
            tk.Entry(path_row, textvariable=file_var, state="readonly",
                     readonlybackground=BG, fg=FG, relief="flat").pack(
                         side="left", fill="x", expand=True)
            choices = tk.Frame(row_frame, bg=CARD)
            choices.pack(fill="x", pady=(3, 0))
            tk.Label(choices, text="Reviewed run:", bg=CARD, fg=MUTED,
                     font=FONT).pack(side="left")
            run_box = ttk.Combobox(choices, textvariable=run_choice,
                                  state="readonly", width=36)
            run_box.pack(side="left", padx=(4, 10))
            tk.Label(choices, text="Channel:", bg=CARD, fg=MUTED,
                     font=FONT).pack(side="left")
            channel_box = ttk.Combobox(choices, textvariable=channel_choice,
                                      state="readonly", width=6)
            channel_box.pack(side="left", padx=4)

            def run_changed(_event=None):
                chosen = next((item for item in entry["runs"]
                               if item["name"] == run_choice.get()), None)
                values = [str(value) for value in chosen["channels"]] if chosen else []
                channel_box.configure(values=values)
                preferred = channel_var.get()
                channel_choice.set(preferred if preferred in values else
                                   values[0] if values else "")
                refresh_gallery()

            def browse_project():
                from .retraining import project_runs
                file = filedialog.askopenfilename(
                    parent=self.root, title="Choose cellmatchr_project.json",
                    filetypes=[("CellMatchR project", "cellmatchr_project.json"),
                               ("JSON files", "*.json")])
                if not file:
                    return
                try:
                    runs = project_runs(file)
                    if not runs:
                        raise ValueError("No segmentation run with predicted channels was found")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    messagebox.showerror("Cannot add project", str(exc), parent=self.root)
                    return
                entry["runs"] = runs
                file_var.set(file)
                run_box.configure(values=[item["name"] for item in runs])
                preferred = run_var.get()
                run_choice.set(preferred if preferred in [item["name"] for item in runs]
                               else runs[-1]["name"])
                run_changed()

            tk.Button(path_row, text="Browse project file…", command=browse_project,
                      bg=ACTIVE, fg=button_fg, font=FONT, relief="flat").pack(side="left", padx=5)
            def remove_project():
                imported_rows.remove(entry)
                row_frame.destroy()
                refresh_gallery()
            tk.Button(path_row, text="Remove", command=remove_project,
                      bg=ACTIVE, fg=button_fg, font=FONT, relief="flat").pack(side="left")
            run_box.bind("<<ComboboxSelected>>", run_changed)
            channel_box.bind("<<ComboboxSelected>>", lambda _event: refresh_gallery())

        tk.Button(sources_card, text="＋ Add another project", command=add_project_row,
                  bg=ACTIVE, fg=button_fg, font=FONT_BOLD, relief="flat",
                  padx=10, pady=5).pack(anchor="w", pady=(4, 0))

        labels_card = self._card(parent, "Cell labels to train")
        tk.Label(labels_card, text=(
            "Leave every label unchecked (default) to train on all cells kept in a saved "
            "corrected mask; individual cell verification is not required. Erased predictions "
            "and ARTIFACT labels are background. Select labels only after completing individual "
            "cell review; other cell types become background in that specialized model."),
            bg=CARD, fg=MUTED, font=FONT, wraplength=740,
            justify="left").pack(anchor="w", pady=(0, 5))
        label_choices_frame = tk.Frame(labels_card, bg=CARD)
        label_choices_frame.pack(fill="x")
        label_vars = {}

        gallery_card = self._card(parent, "Images available for training")
        tk.Label(gallery_card, text=("Uncheck images that should not be used. Previews combine "
                                     "all saved channels without mask outlines. Counts show "
                                     "original proposals, ARTIFACT/erased predictions, training cells, "
                                     "and any completed cell-label counts."),
                 bg=CARD, fg=MUTED, font=FONT, wraplength=740,
                 justify="left").pack(anchor="w", pady=(0, 4))
        gallery_summary = tk.Label(gallery_card, text="Checking corrected and reviewed images…",
                                   bg=CARD, fg=MUTED, font=FONT, anchor="w")
        gallery_summary.pack(anchor="w", pady=(0, 4))
        gallery_canvas = tk.Canvas(gallery_card, bg=CARD, height=390,
                                   highlightthickness=0)
        gallery_scroll = ttk.Scrollbar(gallery_card, orient="vertical",
                                      command=gallery_canvas.yview)
        gallery_canvas.configure(yscrollcommand=gallery_scroll.set)
        gallery_scroll.pack(side="right", fill="y")
        gallery_canvas.pack(side="left", fill="both", expand=True)
        gallery_grid = tk.Frame(gallery_canvas, bg=CARD)
        gallery_window = gallery_canvas.create_window(
            (0, 0), window=gallery_grid, anchor="nw")
        def resize_gallery(event):
            width = int(float(gallery_canvas.itemcget(gallery_window, "width")))
            if width != event.width:
                gallery_canvas.itemconfigure(gallery_window, width=event.width)
        def update_gallery_scroll(_event):
            bounds = gallery_canvas.bbox("all")
            current = tuple(int(float(value)) for value in
                            str(gallery_canvas.cget("scrollregion")).split())
            if bounds and bounds != current:
                gallery_canvas.configure(scrollregion=bounds)
        gallery_canvas.bind("<Configure>", resize_gallery)
        gallery_grid.bind("<Configure>", update_gallery_scroll)

        validation_card = self._card(parent, "Choose how to validate training")
        from .retraining import DEFAULT_PATCH_TARGET_INCORRECT
        tk.Label(validation_card, text="Choose one approach before training starts.",
                 bg=CARD, fg=FG, font=FONT_BOLD).pack(anchor="w", pady=(0, 4))
        validation_mode_var = tk.StringVar(value="")
        validation_key_var = tk.StringVar(value="")
        patch_target_var = tk.StringVar(value=str(DEFAULT_PATCH_TARGET_INCORRECT))
        validation_hint = tk.Label(validation_card, text="No approach selected yet.",
                                   bg=CARD, fg=MUTED, font=FONT, wraplength=730,
                                   justify="left")
        for mode, title, explanation in (
            ("whole_image", "Reserve one whole image (recommended)",
             "Keep a chosen image out of training; compare before and after on it."),
            ("patch", "Reserve a small patch (small dataset)",
             "Choose a random patch around an ARTIFACT or erased prediction; "
             "its pixels stay out of training."),
            ("training_image", "Use all images for training (very small dataset)",
             "Evaluate on a trained image. Results are in-sample, not independent validation."),
        ):
            tk.Radiobutton(validation_card, text=title, variable=validation_mode_var,
                           value=mode, bg=CARD, fg=FG, selectcolor=BG,
                           tristatevalue="__unset__", activebackground=CARD,
                           font=FONT_BOLD).pack(anchor="w")
            tk.Label(validation_card, text=explanation, bg=CARD, fg=MUTED,
                     font=("Segoe UI", 8), wraplength=700,
                     justify="left").pack(anchor="w", padx=(22, 0), pady=(0, 4))
        choose_row = tk.Frame(validation_card, bg=CARD)
        choose_row.pack(anchor="w", pady=(3, 2))
        def choose_holdout():
            validation_mode_var.set("whole_image")
            gallery_state["choosing"] = True
            validation_hint.config(text="Click the preview of the image to reserve below.",
                                   fg=ACCENT)
        tk.Button(choose_row, text="Choose held-out image from library",
                  command=choose_holdout, bg=ACTIVE, fg=button_fg,
                  font=FONT, relief="flat").pack(side="left")
        tk.Label(choose_row, text="Target incorrect predictions per patch (approx.):",
                 bg=CARD, fg=MUTED, font=FONT).pack(side="left", padx=(15, 4))
        tk.Entry(choose_row, textvariable=patch_target_var, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat", width=4).pack(side="left")
        validation_hint.pack(anchor="w", pady=(4, 6))
        actions = tk.Frame(validation_card, bg=CARD)
        actions.pack(anchor="w", pady=6)
        status = tk.Label(validation_card, text="Checking saved masks…",
                          bg=CARD, fg=MUTED, font=FONT, wraplength=720,
                          justify="left")
        status.pack(anchor="w", pady=(3, 0))

        log_card = self._card(parent, "Retraining Log")
        log_box = scrolledtext.ScrolledText(
            log_card, bg="#0d0d15", fg="#a6e3a1", font=FONT_MONO,
            height=16, relief="flat", wrap="word")
        log_box.pack(fill="both", expand=True)
        # Deliberately not packed until training has produced a report.
        result_card = tk.Frame(parent, bg=CARD, padx=14, pady=12)
        result_photos = []

        def show_report(report_path):
            for child in result_card.winfo_children():
                child.destroy()
            result_photos.clear()
            result_card.pack(fill="x", pady=(10, 6))
            tk.Label(result_card, text="Training result", bg=CARD, fg=ACCENT,
                     font=FONT_TITLE).pack(anchor="w", pady=(0, 6))
            try:
                with open(report_path, encoding="utf-8") as source:
                    report = json.load(source)
            except (OSError, ValueError) as exc:
                tk.Label(result_card, text=f"Could not read report: {exc}",
                         bg=CARD, fg=RED, font=FONT).pack(anchor="w")
                return
            if report.get("status") == "report_error":
                tk.Label(result_card, text=report.get("message", "Report generation failed"),
                         bg=CARD, fg=RED, font=FONT, wraplength=720,
                         justify="left").pack(anchor="w")
            else:
                plan = report.get("validation_plan", {})
                label = ("In-sample check — this image was used for training"
                         if plan.get("mode") == "training_image" else
                         "Reserved image/patch — not used for training")
                tk.Label(result_card, text=label, bg=CARD, fg=YELLOW if
                         plan.get("mode") == "training_image" else GREEN,
                         font=FONT_BOLD).pack(anchor="w", pady=(0, 4))
                for entry in report.get("validation", []):
                    before, after = entry["before"], entry["after"]
                    reference = ("corrected mask (individual labels not verified)"
                                 if entry.get("review_level") == "mask_correction" else
                                 "cell-verified mask")
                    caption = (f"{os.path.basename(entry['project'])} / "
                               f"{entry['sample']} — {reference}:  "
                               f"recall {before['recall']:.1%} → "
                               f"{after['recall']:.1%}; missed cells "
                               f"{before['missed_cell']} → {after['missed_cell']}; "
                               f"false positives {before['false_positive']} → "
                               f"{after['false_positive']}")
                    tk.Label(result_card, text=caption, bg=CARD, fg=FG,
                             font=FONT, wraplength=730,
                             justify="left").pack(anchor="w", pady=2)
                if _PIL:
                    names = ["learning_curve.png"]
                    for index in range(len(report.get("validation", []))):
                        names.extend((f"image_{index+1}_reviewed.png",
                                      f"image_{index+1}_before.png",
                                      f"image_{index+1}_after.png"))
                    image_row = tk.Frame(result_card, bg=CARD)
                    image_row.pack(fill="x", pady=(8, 4))
                    for position, name in enumerate(names[:10]):
                        file = os.path.join(os.path.dirname(report_path), name)
                        if not os.path.isfile(file):
                            continue
                        try:
                            with Image.open(file) as original:
                                preview = original.copy()
                            preview.thumbnail((350, 245), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(preview)
                            result_photos.append(photo)
                            cell = tk.Frame(image_row, bg=CARD)
                            cell.grid(row=position // 2, column=position % 2,
                                      sticky="nw", padx=4, pady=4)
                            tk.Label(cell, text=name.replace("_", " ").removesuffix(".png"),
                                     bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
                            tk.Label(cell, image=photo, bg=CARD).pack(anchor="w")
                        except (OSError, ValueError):
                            continue
            tk.Label(result_card, text=f"Saved report: {report_path}", bg=CARD,
                     fg=MUTED, font=FONT, wraplength=730,
                     justify="left").pack(anchor="w", pady=(5, 2))
            def open_report_folder():
                folder = os.path.dirname(report_path)
                try:
                    if sys.platform == "darwin":
                        subprocess.Popen(["open", folder])
                    elif sys.platform == "win32":
                        os.startfile(folder)
                    else:
                        subprocess.Popen(["xdg-open", folder])
                except OSError as exc:
                    messagebox.showerror("Report folder", str(exc), parent=self.root)
            tk.Button(result_card, text="Open report folder", command=open_report_folder,
                      bg=ACTIVE, fg=button_fg, font=FONT, relief="flat").pack(anchor="w")

        def log(line):
            try:
                log_box.insert("end", line + "\n")
                log_box.see("end")
            except tk.TclError:
                pass

        def selected_cell_labels():
            return {word for word, variable in label_vars.items() if variable.get()}

        def matching_examples():
            selected = selected_cell_labels()
            return [item for item in gallery_state["examples"]
                    if item["key"] not in gallery_state["excluded"]
                    and (not selected or selected.intersection(item["label_ids"]))]

        def update_start_state(*_):
            included = matching_examples()
            mode = validation_mode_var.get()
            key = validation_key_var.get()
            ready = bool(gallery_state["sources"] and included and mode)
            if mode == "whole_image":
                ready = ready and len(included) >= 2 and any(
                    item["key"] == key for item in included)
            elif mode == "patch":
                try:
                    valid_target = 1 <= int(patch_target_var.get()) <= 50
                except ValueError:
                    valid_target = False
                ready = (ready and valid_target and any(
                    item.get("rejected_ids") or
                    (item.get("erased_prediction_ids") and item.get("prediction_baseline"))
                    for item in included))
            if start_button_ref[0] is not None:
                start_button_ref[0].config(state="normal" if ready else "disabled")
            if mode == "whole_image":
                selected = next((item for item in included if item["key"] == key), None)
                validation_hint.config(
                    text=("Include at least two training images to reserve one."
                          if len(included) < 2 else
                          f"Reserved: {selected['project_name']} / {selected['sample']}"
                          if selected else "Click ‘Choose held-out image’, then click a preview."),
                    fg=ACCENT if selected else MUTED)
            elif mode == "patch":
                hint = ("Enter a target from 1 to 50." if not valid_target else
                        "No ARTIFACT or erased prediction with a saved original mask "
                        "is available to locate a patch." if not any(
                            item.get("rejected_ids") or
                            (item.get("erased_prediction_ids") and
                             item.get("prediction_baseline")) for item in included) else
                        "An ARTIFACT or erased prediction anchors a random patch. "
                        "The report records its exact location and count.")
                validation_hint.config(text=hint, fg=MUTED)
            elif mode == "training_image":
                validation_hint.config(text="All selected images train the model. The result "
                                       "will be labeled in-sample, not held-out.", fg=YELLOW)
            elif not gallery_state["choosing"]:
                validation_hint.config(text="No approach selected yet.", fg=MUTED)

        validation_mode_var.trace_add("write", update_start_state)
        patch_target_var.trace_add("write", update_start_state)

        def render_label_choices():
            previous = {word: variable.get() for word, variable in label_vars.items()}
            label_vars.clear()
            for child in label_choices_frame.winfo_children():
                child.destroy()
            counts = {}
            for item in gallery_state["examples"]:
                for word, ids in item["label_ids"].items():
                    counts[word] = counts.get(word, 0) + len(ids)
            if not counts:
                tk.Label(label_choices_frame,
                         text="Label choices appear after completed individual cell reviews.",
                         bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
                return
            for word in sorted(counts):
                variable = tk.BooleanVar(value=previous.get(word, False))
                label_vars[word] = variable
                tk.Checkbutton(label_choices_frame,
                               text=f"{word} ({counts[word]} cells)",
                               variable=variable, command=render_gallery,
                               bg=CARD, fg=FG, selectcolor=BG,
                               activebackground=CARD, font=FONT).pack(anchor="w")
            tk.Button(label_choices_frame, text="Use all cell labels",
                      command=lambda: clear_label_choices(), bg=ACTIVE,
                      fg=button_fg, font=FONT, relief="flat", padx=8).pack(
                          anchor="w", pady=(5, 0))

        def clear_label_choices():
            for variable in label_vars.values():
                variable.set(False)
            render_gallery()

        def render_gallery():
            for child in gallery_grid.winfo_children():
                child.destroy()
            gallery_state["photos"] = []
            items = sorted(gallery_state["examples"] + gallery_state["unavailable"],
                           key=lambda entry: (entry["project_name"], entry["seg_run"], entry["sample"]))
            if not items:
                tk.Label(gallery_grid, text="No saved corrected masks or completed cell reviews found.",
                         bg=CARD, fg=YELLOW, font=FONT).grid(row=0, column=0,
                                                              sticky="w", pady=10)
            for index, item in enumerate(items):
                tile = tk.Frame(gallery_grid, bg=ACTIVE, padx=8, pady=7)
                tile.grid(row=index // 2, column=index % 2, sticky="nsew",
                          padx=4, pady=4)
                chosen = selected_cell_labels()
                targets = (sum(len(ids) for word, ids in item["label_ids"].items()
                               if word in chosen) if chosen else item["final_count"])
                unavailable = item.get("unavailable")
                eligible = targets > 0 and not unavailable
                image = item.get("preview")
                if image is not None and _PIL:
                    photo = ImageTk.PhotoImage(image)
                    gallery_state["photos"].append(photo)
                    picture = tk.Label(tile, image=photo, bg=ACTIVE,
                                       cursor="hand2")
                else:
                    picture = tk.Label(tile, text=item.get("preview_error", "Preview unavailable"), bg=BG,
                                       fg=MUTED, width=45, height=10,
                                       wraplength=320,
                                       cursor="hand2")
                picture.pack(anchor="center")
                def reserve(_event=None, selected=item):
                    if not any(entry["key"] == selected["key"]
                               for entry in matching_examples()):
                        return
                    if gallery_state["choosing"] or validation_mode_var.get() == "whole_image":
                        validation_key_var.set(selected["key"])
                        validation_mode_var.set("whole_image")
                        gallery_state["choosing"] = False
                        update_start_state()
                picture.bind("<Button-1>", reserve)
                included = tk.BooleanVar(value=(eligible and item["key"] not in
                                                 gallery_state["excluded"]))
                def toggle(selected=item, variable=included):
                    if variable.get():
                        gallery_state["excluded"].discard(selected["key"])
                    else:
                        gallery_state["excluded"].add(selected["key"])
                    update_start_state()
                    gallery_summary.config(
                        text=f"{len(matching_examples())} of {len(items)} image(s) "
                             "match selected labels and are included")
                tk.Checkbutton(tile, text=f"{item['project_name']} / {item['sample']} "
                               f"(ch{item['channel']})" +
                               (" — unavailable for training" if unavailable else
                                " — no selected cells" if not eligible else ""),
                               variable=included,
                               command=toggle, bg=ACTIVE, fg=FG, selectcolor=BG,
                               activebackground=ACTIVE, font=FONT_BOLD,
                               state="normal" if eligible else "disabled",
                               wraplength=330, justify="left").pack(anchor="w")
                if not unavailable:
                    tk.Label(tile, text=(f"Original proposals: ~{item['original_count']}  ·  "
                                         f"ARTIFACT/erased: {item['incorrect_count']}  ·  "
                                         f"Training targets: {targets}"),
                             bg=ACTIVE, fg=MUTED, font=("Segoe UI", 8),
                             wraplength=330, justify="left").pack(anchor="w")
                label_counts = "  ·  ".join(
                    f"{word}: {len(ids)}" for word, ids in sorted(item["label_ids"].items()))
                if label_counts:
                    tk.Label(tile, text=f"Completed cell labels: {label_counts}",
                             bg=ACTIVE, fg=MUTED, font=("Segoe UI", 8),
                             wraplength=330, justify="left").pack(anchor="w")
                verified = item.get("verified", item.get("review_level") == "cell_labels")
                tk.Label(tile, text="Verified" if verified else "Not verified",
                         bg=ACTIVE, fg=GREEN if verified else MUTED, font=("Segoe UI", 8),
                         wraplength=330, justify="left").pack(anchor="w")
            for column in (0, 1):
                gallery_grid.grid_columnconfigure(column, weight=1)
            gallery_summary.config(
                text=f"{len(matching_examples())} of {len(items)} image(s) "
                     "match selected labels and are included")
            update_start_state()

        def refresh_gallery(*_):
            gallery_state["generation"] += 1
            generation = gallery_state["generation"]
            try:
                specs = source_specs()
            except (ValueError, OSError, KeyError) as exc:
                gallery_state["sources"] = ""
                gallery_state["examples"] = []
                gallery_state["unfinished"] = []
                gallery_state["unavailable"] = []
                render_label_choices()
                render_gallery()
                status.config(text=str(exc), fg=RED)
                return
            gallery_state["sources"] = ""
            update_start_state()
            check_btn.config(state="disabled")
            status.config(text="Checking corrected and reviewed masks…", fg=ACCENT)
            def worker():
                try:
                    from .retraining import collect_source_examples
                    from .training_gallery import (composite_preview, project_preview_settings,
                                                   unavailable_gallery_samples)
                    from .slide_roi_preview import slide_roi_for_sample
                    examples, skipped, unfinished = collect_source_examples(
                        specs, with_review_status=True)
                    unavailable = unavailable_gallery_samples(specs, examples, skipped)
                    display = {(os.path.normcase(os.path.realpath(spec["project"])), spec["seg_run"]):
                               project_preview_settings(
                                   spec["project"],
                                   self.config.get("project_output_folder", ""), self.config,
                                   run=spec["seg_run"])
                               for spec in specs}
                    for item in examples + unavailable:
                        try:
                            colors, brightfield = display[
                                os.path.normcase(os.path.realpath(item["project"])), item["seg_run"]]
                            run = os.path.join(item["project"], item["seg_run"])
                            roi_path = slide_roi_for_sample(item["sample"], run=run)
                            folder = item.get("folder") or os.path.dirname(item["mask"])
                            item["preview"] = composite_preview(
                                folder, color_names=colors, brightfield=brightfield, roi_path=roi_path)
                        except (OSError, ValueError, ImportError) as exc:
                            item["preview"] = None
                            item["preview_error"] = str(exc)
                    def finished():
                        if generation != gallery_state["generation"]:
                            return
                        try:
                            check_btn.config(state="normal")
                            gallery_state["sources"] = json.dumps(specs, sort_keys=True)
                            gallery_state["examples"] = examples
                            gallery_state["unfinished"] = unfinished
                            gallery_state["unavailable"] = unavailable
                            render_label_choices()
                            render_gallery()
                            cell_count = sum(len(item["accepted_ids"]) for item in examples)
                            rejected_count = sum(item["incorrect_count"] for item in examples)
                            mask_only_count = sum(item["review_level"] == "mask_correction"
                                                  for item in examples)
                            status.config(text=(f"{len(examples)} sample(s) ready "
                                                f"({mask_only_count} corrected masks without "
                                                f"cell-label review; {cell_count} training cells, "
                                                f"{rejected_count} incorrect predictions); "
                                                f"{len(unfinished)} not ready, "
                                                f"{len(skipped) - len(unfinished)} other skipped."),
                                          fg=GREEN if examples else YELLOW)
                            for reason in skipped:
                                log("Skipped: " + reason)
                        except tk.TclError:
                            pass
                    self.root.after(0, finished)
                except (ValueError, OSError, RuntimeError, KeyError) as exc:
                    error = str(exc)
                    def failed():
                        if generation != gallery_state["generation"]:
                            return
                        try:
                            check_btn.config(state="normal")
                            status.config(text=error, fg=RED)
                        except tk.TclError:
                            pass
                    self.root.after(0, failed)
            threading.Thread(target=worker, daemon=True).start()

        def check_data():
            refresh_gallery()

        def start_training():
            if self._is_running("Model Retraining"):
                status.config(text="Training is already running.", fg=YELLOW)
                return
            try:
                specs = source_specs()
                if json.dumps(specs, sort_keys=True) != gallery_state["sources"]:
                    raise ValueError("Refresh the image library after changing projects or channels")
                included = matching_examples()
                mode = validation_mode_var.get()
                if not mode:
                    raise ValueError("Choose how to validate training below the image library")
                if not included:
                    raise ValueError("Include at least one corrected or verified image")
                holdout = validation_key_var.get() if mode == "whole_image" else ""
                if mode == "whole_image" and (len(included) < 2 or not any(
                        item["key"] == holdout for item in included)):
                    raise ValueError("Choose a held-out image and keep another image for training")
                patch_target = (int(patch_target_var.get()) if mode == "patch" else
                                DEFAULT_PATCH_TARGET_INCORRECT)
                if mode == "patch" and not 1 <= patch_target <= 50:
                    raise ValueError("Incorrect predictions per patch must be 1–50")
                channel = int(channel_var.get())
                epochs = int(epochs_var.get())
                if channel < 0 or not 1 <= epochs <= 500:
                    raise ValueError("Use a channel ID of 0 or higher and 1–500 epochs")
                model_name = model_name_var.get().strip()
                if not model_name:
                    raise ValueError("Enter a name for the new model")
                if len(model_name) > 60 or any(ch in model_name for ch in "/\\\n\r\t"):
                    raise ValueError("Model name must be under 60 characters with no slashes")
                from .retraining import validate_advanced
                advanced = {}
                for key, var in advanced_vars.items():
                    if var.get().strip():
                        advanced[key] = var.get().strip()
                backend = backend_var.get()
                if starting_choice.get() == "Other model from file/folder":
                    chosen = os.path.expanduser(manual_path_var.get().strip())
                    valid = (os.path.isdir(chosen) if backend == "stardist"
                             else os.path.isfile(chosen))
                    if not valid:
                        raise ValueError("Choose an existing starting model "
                                         + ("folder" if backend == "stardist" else "file"))
                    base_var.set(os.path.abspath(chosen))
                if backend == "cellpose":
                    advanced.pop("patch_size", None)
                if backend == "stardist" and loss_var.get() != "Model default":
                    advanced["loss"] = loss_var.get()
                if backend == "microsam" and not freeze_var.get():
                    advanced["freeze_encoder"] = False
                advanced["sampling_focus"] = sampling_focus_var.get()
                advanced = validate_advanced(backend, advanced)
            except ValueError as exc:
                status.config(text=f"Check training settings: {exc}", fg=RED)
                return
            project = self.config.get("project_output_folder", "").strip()
            if not project:
                status.config(text="Open a project in Step 1 first.", fg=RED)
                return
            if backend == "stardist":
                from importlib.util import find_spec
                if find_spec("stardist") is None or find_spec("tensorflow") is None:
                    status.config(text=("StarDist training needs the optional StarDist and "
                                        "TensorFlow packages in this app's environment."), fg=RED)
                    return
            chosen_preprocess = preprocess_var.get()
            use_trained_model = use_var.get()
            unfinished = list(gallery_state["unfinished"])
            if selected_cell_labels():
                unfinished.extend(
                    f"{item['sample']}: individual cell labels not verified"
                    for item in gallery_state["examples"]
                    if item["review_level"] != "cell_labels")
            allow_completed_only = False
            if unfinished:
                allow_completed_only = messagebox.askyesno(
                    "Some images are not ready",
                    f"{len(unfinished)} image(s) cannot be used with these training "
                    "choices yet.\n\n"
                    f"Train using the {len(included)} included, ready image(s) only?\n\n"
                    "No cancels training so you can correct or review more images.",
                    parent=self.root, default=messagebox.NO)
                if not allow_completed_only:
                    status.config(text="Training canceled; correct or review more images first.",
                                  fg=YELLOW)
                    return
            cmd = [sys.executable, "-m", "cellmatchr.retraining",
                   "--project", project, "--seg-run", run_var.get() or "Segmentation",
                   "--channel", str(channel), "--backend", backend,
                   "--epochs", str(epochs), "--base-model", base_var.get().strip(),
                   "--preprocess", chosen_preprocess, "--model-name", model_name,
                   "--sources-json", json.dumps(specs),
                   "--validation-mode", mode,
                   "--patch-target-incorrect", str(patch_target)]
            if allow_completed_only:
                cmd.append("--allow-completed-only")
            for word in sorted(selected_cell_labels()):
                cmd.extend(("--cell-label", word))
            if holdout:
                cmd.extend(("--validation-key", holdout))
            for key in sorted(gallery_state["excluded"]):
                cmd.extend(("--exclude-key", key))
            for key, value in advanced.items():
                flag = "--" + key.replace("_", "-")
                if key == "freeze_encoder":
                    value = "yes" if value else "no"
                cmd.extend((flag, str(value)))
            start_btn.config(state="disabled", text="Training…")
            status.config(text="Training started. Progress appears below.", fg=ACCENT)
            log("Starting " + backend + " retraining…")

            def worker():
                model_path = ""
                report_path = ""
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=os.path.dirname(os.path.dirname(self._script("retraining.py"))),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
                    self._track_proc(proc, "Model Retraining")
                    for line in proc.stdout:
                        clean = line.rstrip()
                        if clean.startswith("[COMPLETE] Trained model: "):
                            model_path = clean.partition(": ")[2]
                        elif clean.startswith("[REPORT] "):
                            report_path = clean.removeprefix("[REPORT] ")
                        self.root.after(0, log, clean)
                    proc.wait()
                    def finished():
                        resolved_report_path = report_path
                        setup_error = None
                        if proc.returncode == 0 and model_path and use_trained_model:
                            try:
                                key = {"microsam": "seg_custom_model_path",
                                       "cellpose": "seg_cellpose_model_path",
                                       "stardist": "seg_stardist_model_path"}[backend]
                                self.config[key] = model_path
                                self._vars[key].set(model_path)
                                if backend == "cellpose":
                                    self.config["seg_cellpose_variant"] = "cpsam"
                                    self._vars["seg_cellpose_variant"].set("cpsam")
                                self.config["seg_backend"] = backend
                                self._vars["seg_backend"].set(backend)
                                self.config["seg_preprocess_mode"] = chosen_preprocess
                                self._vars["seg_preprocess_mode"].set(chosen_preprocess)
                                save_config(self.config)
                                save_project_config(self.config)
                            except (OSError, ValueError, tk.TclError) as exc:
                                setup_error = str(exc)
                                log(f"Model trained, but automatic selection failed: {exc}")
                        try:
                            start_btn.config(state="normal", text="Start Retraining")
                            if proc.returncode == 0 and model_path:
                                refresh_starting_models(prefer_path=model_path)
                                if not resolved_report_path:
                                    try:
                                        with open(os.path.join(os.path.dirname(model_path),
                                                               "model.json"), encoding="utf-8") as source:
                                            resolved_report_path = json.load(source).get("report_path", "")
                                    except (OSError, ValueError):
                                        pass
                                if resolved_report_path:
                                    show_report(resolved_report_path)
                                message = (f"Training complete: {model_name}. "
                                           + ("Open Step 3 to select it. " if not use_trained_model else
                                              "It is selected for future segmentation. ")
                                           + (f"Automatic selection failed: {setup_error}. "
                                              if setup_error else "")
                                           + f"Saved at {model_path}")
                                status.config(text=message, fg=YELLOW if setup_error else GREEN)
                            else:
                                status.config(text="Training failed; check the log above.", fg=RED)
                        except tk.TclError:
                            pass
                    self.root.after(0, finished)
                except Exception as exc:
                    self.root.after(0, log, f"Training error: {exc}")
                    def failed():
                        try:
                            start_btn.config(state="normal", text="Start Retraining")
                            status.config(text="Training could not start; check the log above.", fg=RED)
                        except tk.TclError:
                            pass
                    self.root.after(0, failed)
            threading.Thread(target=worker, daemon=True).start()

        check_btn = tk.Button(actions, text="Refresh image library", command=check_data,
                              bg=ACTIVE, fg=button_fg, font=FONT_BOLD, relief="flat",
                              padx=12, pady=7,
                              state="normal")
        check_btn.pack(side="left", padx=(0, 8))
        start_btn = tk.Button(actions, text="Start Retraining", command=start_training,
                              bg=ACCENT, fg="#1e1e2e", font=FONT_BOLD,
                              relief="flat", padx=12, pady=7,
                              state="disabled")
        start_btn.pack(side="left")
        start_button_ref[0] = start_btn
        run_var.trace_add("write", lambda *_: refresh_gallery())
        channel_var.trace_add("write", lambda *_: refresh_gallery())
        refresh_gallery()

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
                        self._vsi_auto_label.config(text=self._vsi_auto_text(), fg=FG)
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
        self._archive_model_preview(out, self._log)
        self._run_subprocess_in_log(self._script("segmentation.py"), "Segmentation")

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
        if not self._confirm_legacy_roi_matches(ana_input):
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
            extra_args=("--use-saved-reviews",),
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
            ana_input = os.path.join(proj, seg_sub_for_ana)
            if not run_seg and not self._confirm_legacy_roi_matches(ana_input):
                return
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
        if seg_out:
            self._archive_model_preview(seg_out, self._log)
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
                    if (self.config.get("seg_backend") == "cellpose"
                            and self.config.get("seg_cellpose_variant", "cpsam") != "cpsam"):
                        self.root.after(0, self._log,
                                        "Preparing the selected Cellpose model; first use "
                                        "downloads its weights and separate runtime…")
                    command, environment = segmentation_command(
                        seg_script, CONFIG_PATH, self.config)
                    proc1 = subprocess.Popen(
                        command, env=environment,
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
                    if run_seg:
                        # The new sample names are known only once segmentation completes.
                        decision = threading.Event()
                        accepted = False

                        def confirm_masks():
                            nonlocal accepted
                            try:
                                accepted = self._confirm_legacy_roi_matches(ana_input)
                            finally:
                                decision.set()

                        self.root.after(0, confirm_masks)
                        decision.wait()
                        if not accepted:
                            _restore_previous()
                            self.root.after(0, self._set_run_status,
                                            "Segmentation complete. Analysis canceled.", YELLOW)
                            return
                    self.config["ana_microsam_output_root"] = ana_input
                    self.config["output_root_folder"] = ana_out
                    save_config(self.config)
                    save_project_config(self.config)

                    self.root.after(0, self._log, "\n" + "─"*40)
                    self.root.after(0, self._log,
                                    f"▶  Analyzing {seg_sub_for_ana}…")
                    self.root.after(0, self._set_run_status,
                                    "Analysis running…", ACCENT)

                    proc2 = subprocess.Popen(
                        [sys.executable, ana_script, "--config", CONFIG_PATH, "--use-saved-reviews"],
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
