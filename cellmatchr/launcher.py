"""
CellMatchr Launcher — step-by-step wizard UI.
"""

import glob
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

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


# ---------------------------------------------------------------------------
# Conflict dialog — shown when an output folder already exists
# ---------------------------------------------------------------------------

class _ConflictDialog(tk.Toplevel):
    """Ask user: replace existing folder, or save with a custom suffix."""

    def __init__(self, parent: tk.Tk, folder_path: str) -> None:
        super().__init__(parent)
        self.result = None          # None=cancel | "replace" | "<suffix string>"
        self.transient(parent)
        self.grab_set()
        self.configure(bg=BG)
        self.title("Output already exists")
        self.resizable(False, False)

        folder_name = os.path.basename(folder_path)

        tk.Label(self, text=f"'{folder_name}' already exists.",
                 bg=BG, fg=FG, font=FONT_BOLD).pack(pady=(20, 4), padx=24)
        tk.Label(self, text="Choose what to do:",
                 bg=BG, fg=MUTED, font=FONT).pack(pady=(0, 14), padx=24)

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

        row = tk.Frame(self, bg=BG)
        row.pack(padx=24, pady=4)
        tk.Label(row, text=f"{folder_name}_", bg=BG, fg=MUTED, font=FONT).pack(side="left")
        self._suffix_var = tk.StringVar()
        self._entry = tk.Entry(row, textvariable=self._suffix_var,
                               bg=CARD, fg=FG, insertbackground=FG,
                               relief="flat", font=FONT, width=16)
        self._entry.pack(side="left", padx=4)
        self._entry.focus_set()
        self._entry.bind("<Return>", lambda _: self._new_suffix())

        tk.Button(
            self, text="Save with suffix",
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
        if not s:
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


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------

class CellMatchrLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CellMatchR")
        self.root.configure(bg=BG)
        self.root.geometry("1150x800")
        self.root.minsize(900, 640)

        self.config = DEFAULT_CONFIG.copy()  # always start clean
        self.current_step = 0
        self._spinning = False
        self._spin_idx  = 0

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
            "seg_channels", "seg_model_type", "seg_preprocess_mode",
            "val_mask_pattern", "val_channel_names", "val_channel_colors",
            "ana_red_channel", "ana_pv_channel", "ana_iou_threshold",
            "ana_roi_cell_rule", "ana_roi_mask_folder", "ana_subfolder_name",
        ]
        num_fields = [
            "seg_min_object_size", "seg_max_object_size",
            "seg_tile_h", "seg_tile_w", "seg_halo_h", "seg_halo_w",
            "seg_qc_downsample", "seg_max_ratio", "seg_raw_mean_min",
        ]
        bool_fields = [
            "seg_test_run", "seg_dry_run", "seg_skip_subdirs",
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

        # Title
        tk.Label(self.sidebar, text="CellMatchR",
                 bg=SIDEBAR, fg=ACCENT,
                 font=("Segoe UI", 18, "bold")).pack(pady=(0, 2))
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

    def _show_step(self, idx: int) -> None:
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

    def _browse_dir(self, key: str) -> None:
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

    def _resolve_output_dir(self, base: str) -> str | None:
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
            dlg = _ConflictDialog(self.root, candidate)
            if dlg.result is None:
                return None
            if dlg.result == "replace":
                return candidate
            return os.path.join(proj, f"{base}_{dlg.result}")
        return candidate

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if hasattr(self, "_log_box"):
            self._log_box.insert("end", msg + "\n")
            self._log_box.see("end")

    def _vsi_log_append(self, msg: str) -> None:
        """Append a line to the VSI converter output log box."""
        if not hasattr(self, "_vsi_log"):
            return
        self._vsi_log.config(state="normal")
        self._vsi_log.insert("end", msg + "\n")
        self._vsi_log.see("end")
        self._vsi_log.config(state="disabled")

    def _set_run_status(self, msg: str, color: str = MUTED) -> None:
        if hasattr(self, "_run_status"):
            self._run_status.config(text=msg, fg=color)

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
                                on_done=None) -> None:
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
                for line in proc.stdout:
                    self.root.after(0, self._log, line.rstrip())
                proc.wait()
                ok = proc.returncode == 0
                msg = f"\n✓  {label} complete." if ok \
                    else f"\n✗  {label} failed (code {proc.returncode})."
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
        for val, lbl in [
            ("czi", "CZI or TIF files  (Zeiss / slide scanner — use directly)"),
            ("vsi", "VSI file  (Olympus — convert to TIF first via VSI Converter below)"),
        ]:
            tk.Radiobutton(
                type_card, text=lbl,
                variable=self._vars["input_type"], value=val,
                bg=CARD, fg=FG, selectcolor=CARD,
                activebackground=CARD, font=FONT,
            ).pack(anchor="w", pady=2)

        # ── VSI section (shown when input_type == "vsi") ─────────────
        self._vsi_section = tk.Frame(parent, bg=BG)

        vsi_card = self._card(self._vsi_section, "VSI Converter  (Olympus only)")
        tk.Label(vsi_card,
                 text="Output auto-saves to: Project folder / VSI_Converted_TIFF/",
                 bg=CARD, fg=MUTED, font=FONT).pack(anchor="w", pady=(0, 6))

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
        tk.Button(btn_row, text="Launch VSI Converter →",
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

        # ── CZI/TIF section (shown when input_type == "czi") ─────────
        self._czi_section = tk.Frame(parent, bg=BG)
        czi_card = self._card(self._czi_section, "CZI / TIF Input Folder")
        tk.Label(czi_card,
                 text="Select the root folder. Each subfolder should contain one CZI or TIF file.",
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

        # ── Wire radio toggle ─────────────────────────────────────────
        self._vars["input_type"].trace_add("write", lambda *_: self._on_input_type_changed())
        self._on_input_type_changed()

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
            else:
                self._vsi_section.pack_forget()
                self._czi_section.pack(fill="x")
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
        folder = self.config.get("input_root_folder", "").strip()
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
            bg=ACTIVE, fg=FG, font=FONT, relief="flat", padx=10, pady=4,
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
        ttk.Combobox(
            g, textvariable=self._vars["seg_model_type"],
            values=["vit_b_lm", "vit_l_lm", "vit_h_lm", "vit_b", "vit_l", "vit_h"],
            state="readonly", width=18,
        ).grid(row=2, column=1, sticky="w", pady=3, padx=8)

        tk.Label(g, text="Preprocessing mode:", bg=CARD, fg=FG, font=FONT).grid(
            row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            g, textvariable=self._vars["seg_preprocess_mode"],
            values=["local_bg_subtract", "local_bg_subtract_mild",
                    "raw_percentile", "tile_clahe_mild", "tile_clahe"],
            state="readonly", width=24,
        ).grid(row=3, column=1, sticky="w", pady=3, padx=8)

        filt_card = self._card(parent, "Object Filtering")
        tk.Label(
            filt_card,
            text="Set to 0 to disable a filter.  Values depend on imaging conditions —\n"
                 "open an example image and measure a typical cell to choose appropriate bounds.",
            bg=CARD, fg=MUTED, font=FONT, justify="left",
        ).pack(anchor="w", pady=(0, 8))
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
        for row_i, (lbl, key, hint) in enumerate(_filt_rows):
            tk.Label(filt_grid, text=lbl, bg=CARD, fg=FG, font=FONT).grid(
                row=row_i, column=0, sticky="w", pady=3)
            tk.Entry(filt_grid, textvariable=self._vars[key], bg=BG, fg=FG,
                     insertbackground=FG, relief="flat", font=FONT, width=12).grid(
                row=row_i, column=1, sticky="w", pady=3, padx=8)
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
        self._seg_status = tk.Label(btn_row, text="", bg=CARD, fg=MUTED, font=FONT)
        self._seg_status.pack(side="left", padx=12)

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
        if self._per_ch_filt_enabled.get():
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

    def _find_first_image(self) -> str | None:
        folder = self.config.get("input_root_folder", "").strip()
        if not folder or not os.path.isdir(folder):
            return None
        exts = (".czi", ".tif", ".tiff")
        for item in sorted(os.listdir(folder)):
            sub = os.path.join(folder, item)
            if os.path.isdir(sub):
                for fname in sorted(os.listdir(sub)):
                    if fname.lower().endswith(exts):
                        return os.path.join(sub, fname)
            elif item.lower().endswith(exts):
                return sub
        return None

    def _load_channel_previews(self, callback=None) -> None:
        """Run in daemon thread. Posts result back to main thread via callback."""
        if callback is None:
            callback = self._set_channel_previews
        if not _PIL:
            return
        path = self._find_first_image()
        if not path:
            self.root.after(0, callback, [], "(no image found)")
            return
        try:
            import numpy as np
            fname = os.path.basename(path)
            print(f"\n[Preview] Loading: {fname}")

            if path.lower().endswith(".czi"):
                # Use czifile directly — AICSImage internally hits a tifffile
                # version conflict (_TIFF.RESUNIT) with the pinned env versions.
                from czifile import CziFile
                print(f"[Preview] Reading CZI…")
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
                    print(f"[Preview] No C axis — treated as single channel after MIP")
                else:
                    num_channels = data.shape[c_axis]
                    num_z = data.shape[z_axis] if z_axis != -1 else 1
                    print(f"[Preview] Channels: {num_channels}, Z-slices: {num_z}")
                    if num_z > 1:
                        print(f"[Preview] Z-stack — applying max intensity projection per channel")
                    channels_np = []
                    for c in range(num_channels):
                        ch = np.take(data, c, axis=c_axis)
                        ch = np.squeeze(ch)
                        while ch.ndim > 2:          # MIP over Z and any remaining dims
                            ch = np.max(ch, axis=0)
                        channels_np.append(ch)
                        print(f"[Preview]   Channel {c}: shape={ch.shape}")
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

        # Clear the placeholder and build one entry row per channel
        if self._val_ch_names_frame:
            for w in self._val_ch_names_frame.winfo_children():
                w.destroy()
            self._val_ch_name_vars = []
            self._val_ch_color_vars = []

            _COLOR_OPTIONS = ["Blue", "Green", "Red", "Purple", "Cyan", "Yellow"]
            _COLOR_DEFAULTS = ["Blue", "Green", "Red", "Purple", "Cyan", "Yellow"]

            n = len(photos) if photos else 0
            if n == 0:
                tk.Label(self._val_ch_names_frame,
                         text=fname or "Could not detect channels.",
                         bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
            else:
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
                tk.Label(g, text="Name", bg=CARD, fg=MUTED, font=FONT).grid(
                    row=0, column=1, sticky="w", padx=8, pady=(0, 4))
                tk.Label(g, text="Color", bg=CARD, fg=MUTED, font=FONT).grid(
                    row=0, column=3, sticky="w", pady=(0, 4))
                for i in range(n):
                    def_name  = (saved_names[i]  if i < len(saved_names)  and saved_names[i]
                                 else f"Channel {i}")
                    def_color = (saved_colors[i] if i < len(saved_colors) and saved_colors[i]
                                 else _COLOR_DEFAULTS[i % len(_COLOR_DEFAULTS)])
                    name_var  = tk.StringVar(value=def_name)
                    color_var = tk.StringVar(value=def_color)
                    name_var.trace_add( "write", lambda *_: self._update_val_channel_names_str())
                    color_var.trace_add("write", lambda *_: self._update_val_channel_names_str())
                    self._val_ch_name_vars.append(name_var)
                    self._val_ch_color_vars.append(color_var)
                    tk.Label(g, text=f"Channel {i}:", bg=CARD, fg=FG,
                             font=FONT_BOLD).grid(row=i + 1, column=0, sticky="w", pady=3)
                    tk.Entry(g, textvariable=name_var, bg=BG, fg=FG,
                             insertbackground=FG, relief="flat",
                             font=FONT, width=22).grid(
                        row=i + 1, column=1, sticky="w", pady=3, padx=8)
                    ttk.Combobox(g, textvariable=color_var, values=_COLOR_OPTIONS,
                                 state="readonly", width=10).grid(
                        row=i + 1, column=3, sticky="w", pady=3)
                self._update_val_channel_names_str()

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

    def _run_segmentation_step(self) -> None:
        """Run segmentation from within the Segmentation step card."""
        self._sync_vars_to_config()
        folder = self.config.get("input_root_folder", "").strip()
        channels = self.config.get("seg_channels", "").strip()

        if not folder:
            self._seg_status.config(text="⚠  No input folder set. Go to Step 2.", fg=RED)
            return
        if not channels:
            self._seg_status.config(text="⚠  Channels to predict is required (red ✱).", fg=RED)
            return

        out = self._resolve_output_dir(SUBFOLDER["seg"])
        if out is None:
            return
        self.config["output_root_folder"] = out
        save_config(self.config)
        save_project_config(self.config)

        self._seg_run_btn.config(state="disabled", text="Running…")
        self._seg_progress.start(12)
        self._seg_status.config(text="Starting…", fg=ACCENT)

        # Clear log
        self._seg_log.config(state="normal")
        self._seg_log.delete("1.0", "end")
        self._seg_log.config(state="disabled")

        script = self._script("segmentation.py")

        def _seg_log_line(line: str) -> None:
            if not hasattr(self, "_seg_log"):
                return
            self._seg_log.config(state="normal")
            self._seg_log.insert("end", line + "\n")
            self._seg_log.see("end")
            self._seg_log.config(state="disabled")
            # Also update the one-liner status with the last meaningful line
            if line.strip() and hasattr(self, "_seg_status"):
                self._seg_status.config(text=line.strip()[:120], fg=MUTED)

        def _thread():
            try:
                proc = subprocess.Popen(
                    [sys.executable, script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    self.root.after(0, _seg_log_line, line.rstrip())
                proc.wait()
                ok = proc.returncode == 0
                def _done():
                    self._seg_progress.stop()
                    self._seg_run_btn.config(state="normal", text="▶  Run Segmentation")
                    if ok:
                        self._seg_status.config(text="✔  Segmentation complete.", fg=GREEN)
                        _seg_log_line("\n✔  Segmentation complete.")
                    else:
                        self._seg_status.config(
                            text=f"✗  Failed (exit {proc.returncode}).", fg=RED)
                        _seg_log_line(f"\n✗  Failed (exit code {proc.returncode}).")
                self.root.after(0, _done)
            except Exception as exc:
                def _err():
                    self._seg_progress.stop()
                    self._seg_run_btn.config(state="normal", text="▶  Run Segmentation")
                    self._seg_status.config(text=f"✗  {exc}", fg=RED)
                self.root.after(0, _err)

        threading.Thread(target=_thread, daemon=True).start()

    def _run_validation_step(self) -> None:
        """Launch the validator from the Validation step card."""
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

        # Use the most recent segmentation output folder
        seg_variants = self._find_variants(SUBFOLDER["seg"])
        seg_sub  = seg_variants[-1] if seg_variants else SUBFOLDER["seg"]
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
            subprocess.Popen(
                [sys.executable, self._script("validator.py"), "--config", CONFIG_PATH]
            )
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

        # ── Channel dropdown (derived from seg_channels) ─────────────
        cfg_card = self._card(parent, "Channel to Validate")
        channels_raw = self.config.get("seg_channels", "").strip()
        channel_options = [x.strip() for x in channels_raw.split(",")
                           if x.strip().isdigit()]

        if channel_options:
            tk.Label(cfg_card,
                     text="Select which segmented channel to validate:",
                     bg=CARD, fg=FG, font=FONT).pack(anchor="w", pady=(0, 6))
            self._val_channel_var = tk.StringVar(
                value=f"Channel {channel_options[0]}"
            )
            ttk.Combobox(
                cfg_card, textvariable=self._val_channel_var,
                values=[f"Channel {c}" for c in channel_options],
                state="readonly", width=20,
            ).pack(anchor="w")
            tk.Label(
                cfg_card,
                text="The mask glob pattern is derived automatically from the channel you select.",
                bg=CARD, fg=MUTED, font=FONT,
            ).pack(anchor="w", pady=(6, 0))
        else:
            tk.Label(
                cfg_card,
                text="⚠  No channels configured — go to Step 3 — Segmentation and set channels first.",
                bg=CARD, fg=YELLOW, font=FONT, wraplength=660,
            ).pack(anchor="w")
            self._val_channel_var = None

        # ── Channel display names (one entry per channel) ────────────
        names_card = self._card(parent, "Channel Display Names")
        tk.Label(
            names_card,
            text="Names are shown in the validator for each channel panel. "
                 "They are detected automatically from the first image in your input folder.",
            bg=CARD, fg=MUTED, font=FONT, wraplength=660, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        self._val_ch_names_frame = tk.Frame(names_card, bg=CARD)
        self._val_ch_names_frame.pack(fill="x")
        self._val_ch_name_vars = []
        # Show a placeholder message until the background scan finishes
        tk.Label(
            self._val_ch_names_frame,
            text="Scanning input folder for channel count…" if self.config.get("input_root_folder") else
                 "Set an input folder in Step 2 to detect channels.",
            bg=CARD, fg=MUTED, font=FONT,
        ).pack(anchor="w")

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
            bg=ACTIVE, fg=FG, font=FONT, relief="flat", padx=10, pady=4,
            state="normal" if (has_input and _PIL) else "disabled",
        )
        self._val_load_btn.pack(side="left", padx=(12, 0))

        self._val_preview_img = tk.Label(prev_card, bg=CARD)
        self._val_preview_img.pack(anchor="w", pady=(0, 4))

        if not _PIL:
            tk.Label(prev_card, text="Install Pillow (pip install Pillow) to enable preview.",
                     bg=CARD, fg=YELLOW, font=FONT).pack(anchor="w")

        # ── Validation progress summary ──────────────────────────────
        vprog_card = self._card(parent, "Validation Progress")
        self._val_prog_frame = tk.Frame(vprog_card, bg=CARD)
        self._val_prog_frame.pack(anchor="w", fill="x")
        tk.Button(
            vprog_card, text="↻  Refresh",
            command=self._refresh_val_status,
            bg=BG, fg=MUTED, font=FONT, relief="flat", padx=6, pady=2,
        ).pack(anchor="w", pady=(6, 0))
        self.root.after(120, self._refresh_val_status)

        # ── Run button ───────────────────────────────────────────────
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
        toggle_card = self._card(parent, "Enable")
        tk.Checkbutton(
            toggle_card, text="Include analysis step",
            variable=self._vars["ana_enabled"],
            bg=CARD, fg=FG, selectcolor=CARD,
            activebackground=CARD, font=FONT_BOLD,
        ).pack(anchor="w")

        # ── ROI mask folder ───────────────────────────────────────────
        if self.config.get("input_type") == "vsi":
            # VSI: auto-detect mask folder from VSI Converter output
            if not self.config.get("ana_roi_mask_folder", "").strip():
                proj = self.config.get("project_output_folder", "").strip()
                if proj:
                    default_roi = os.path.join(
                        proj, "VSI_Converted_TIFF", "DONOTDELETE-MASK_FOR_VSI"
                    )
                    self._vars["ana_roi_mask_folder"].set(default_roi)
                    self.config["ana_roi_mask_folder"] = default_roi

            roi_card = self._card(parent, "ROI Mask Folder  (VSI pipeline)")
            tk.Label(
                roi_card,
                text="Auto-detected from VSI Converter output. "
                     "Contains *_mask_overview.tif files.",
                bg=CARD, fg=MUTED, font=FONT,
            ).pack(anchor="w", pady=(0, 4))
            tk.Label(roi_card, textvariable=self._vars["ana_roi_mask_folder"],
                     bg=CARD, fg=FG, font=FONT_MONO, wraplength=700).pack(anchor="w", pady=(0, 4))
            self._ana_roi_override_frame = tk.Frame(roi_card, bg=CARD)
            self._path_row(self._ana_roi_override_frame,
                           "Change ROI mask folder:", "ana_roi_mask_folder")
            ovr_lbl = tk.Label(
                roi_card, text="(Optional) Change ROI mask folder",
                bg=CARD, fg=MUTED, font=(FONT[0], FONT[1], "underline"), cursor="hand2",
            )
            ovr_lbl.pack(anchor="w")
            ovr_lbl.bind("<Button-1>", lambda _: (
                self._ana_roi_override_frame.pack_forget()
                if self._ana_roi_override_frame.winfo_ismapped()
                else self._ana_roi_override_frame.pack(fill="x", pady=(4, 0))
            ))
        else:
            # TIF / CZI: manual ROI mask folder (optional, hidden by default)
            roi_card = self._card(parent, "ROI Mask Folder  (optional, TIF/CZI)")
            tk.Label(
                roi_card,
                text="Optional: provide a folder of ROI mask files to restrict "
                     "cell counts and colocalization to a region of interest.\n\n"
                     "Supported formats:\n"
                     "  • .roi  — Fiji/ImageJ ROI file (polygon drawn at full image resolution)\n"
                     "  • .tif  — binary mask TIF (same pixel dimensions as the original image)\n\n"
                     "Naming rule: each file must contain the full original image filename "
                     "somewhere in its name so CellMatchR can match them.\n"
                     "Example:  original = 'MouseA_slice3.tif'  →  "
                     "mask = 'MouseA_slice3_roi.roi'  or  'MouseA_slice3_mask.tif'",
                bg=CARD, fg=MUTED, font=FONT, wraplength=700, justify="left",
            ).pack(anchor="w", pady=(0, 6))
            self._ana_roi_manual_frame = tk.Frame(roi_card, bg=CARD)
            self._path_row(self._ana_roi_manual_frame,
                           "ROI mask folder:", "ana_roi_mask_folder")
            current_roi = self._vars["ana_roi_mask_folder"].get().strip()
            show_lbl = tk.Label(
                roi_card,
                text="▶  Set ROI mask folder" if not current_roi else "▼  ROI mask folder set (click to change)",
                bg=CARD, fg=MUTED, font=(FONT[0], FONT[1], "underline"), cursor="hand2",
            )
            show_lbl.pack(anchor="w")

            def _toggle_roi_manual(lbl=show_lbl):
                if self._ana_roi_manual_frame.winfo_ismapped():
                    self._ana_roi_manual_frame.pack_forget()
                    lbl.config(text="▶  Set ROI mask folder")
                else:
                    self._ana_roi_manual_frame.pack(fill="x", pady=(4, 0))
                    lbl.config(text="▼  ROI mask folder set (click to change)")

            show_lbl.bind("<Button-1>", lambda _: _toggle_roi_manual())
            # Auto-expand if a folder is already configured
            if current_roi:
                self._ana_roi_manual_frame.pack(fill="x", pady=(4, 0))

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
        channel_options = [x.strip() for x in
                           self.config.get("seg_channels", "").split(",")
                           if x.strip().isdigit()]
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
        ttk.Combobox(g, textvariable=self._ana_coi_chan_var,
                     values=chan_values, state=state, width=18).grid(
            row=0, column=1, sticky="w", pady=3, padx=8)

        tk.Label(g, text="Protein marker channel:", bg=CARD, fg=FG, font=FONT).grid(
            row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(g, textvariable=self._ana_pm_chan_var,
                     values=chan_values, state=state, width=18).grid(
            row=1, column=1, sticky="w", pady=3, padx=8)

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
        ttk.Combobox(g2, textvariable=self._vars["ana_roi_cell_rule"],
                     values=["centroid", "any_pixel"],
                     state="readonly", width=14).grid(
            row=1, column=1, sticky="w", pady=3, padx=8)

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
        """Return 'complete', 'partial', or 'none' for a channel in a project tree."""
        import glob as _glob
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

        coi_ch = self.config.get("ana_red_channel", "")
        pm_ch  = self.config.get("ana_pv_channel",  "")
        channels_to_check = []
        for ch_val in [coi_ch, pm_ch]:
            try:
                channels_to_check.append(int(ch_val))
            except (ValueError, TypeError):
                pass

        # Collect per-channel aggregate status across all project sub-folders
        import glob as _glob
        project_dirs = [
            d for d in sorted(os.listdir(proj))
            if os.path.isdir(os.path.join(proj, d))
            and not d.startswith(".")
            and not d.startswith("DEBUG")
        ]

        any_partial = False
        if not channels_to_check:
            tk.Label(frame, text="Set channel numbers above to see validation status.",
                     bg=CARD, fg=MUTED, font=FONT).pack(anchor="w")
        else:
            for ch in channels_to_check:
                statuses = [self._get_val_status_for_channel(os.path.join(proj, d), ch)
                            for d in project_dirs]
                n_total    = len(statuses)
                n_complete = statuses.count("complete")
                n_partial  = statuses.count("partial")
                n_none     = statuses.count("none")

                if n_total == 0:
                    label_text = f"Channel {ch}: no project folders found"
                    fg = MUTED
                elif n_complete == n_total:
                    label_text = f"Channel {ch}: validation COMPLETE ({n_complete}/{n_total})  ✓  — filter will be applied automatically"
                    fg = GREEN
                elif n_complete + n_partial > 0:
                    label_text = (f"Channel {ch}: PARTIAL — "
                                  f"{n_complete} complete  {n_partial} in-progress  {n_none} not started  ⟳")
                    fg = YELLOW
                    any_partial = True
                else:
                    label_text = f"Channel {ch}: not yet validated  ✗"
                    fg = RED

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
                elif s == "partial":
                    icon, col = "⟳", YELLOW
                    if row_status == "complete":
                        row_status = "partial"
                else:
                    icon, col = "✗", RED
                    row_status = "none"
                ch_parts.append((f"ch{ch}: {icon}", col))

            # Build a single row frame so each channel tag can be its own colour
            row_frame = tk.Frame(frame, bg=CARD)
            row_frame.pack(anchor="w", pady=1, fill="x")

            name_fg = GREEN if row_status == "complete" else (YELLOW if row_status == "partial" else FG)
            tk.Label(row_frame, text=f"{d}  ", bg=CARD, fg=name_fg,
                     font=FONT_BOLD, justify="left").pack(side="left")
            for tag_text, tag_col in ch_parts:
                tk.Label(row_frame, text=f"[{tag_text}]  ", bg=CARD, fg=tag_col,
                         font=FONT, justify="left").pack(side="left")

    def _run_analysis_step(self) -> None:
        """Run analysis from the Analysis step card."""
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

        seg_variants = self._find_variants(SUBFOLDER["seg"])
        seg_sub = seg_variants[-1] if seg_variants else SUBFOLDER["seg"]
        ana_input = os.path.join(proj, seg_sub)

        if not os.path.isdir(ana_input):
            if hasattr(self, "_ana_status"):
                self._ana_status.config(
                    text=f"⚠  Segmentation output not found: {seg_sub}", fg=RED)
            return

        out = self._resolve_output_dir(SUBFOLDER["ana"])
        if out is None:
            return

        self.config["ana_microsam_output_root"] = ana_input
        self.config["output_root_folder"] = out
        save_config(self.config)
        save_project_config(self.config)

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
            if not hasattr(self, "_ana_log"):
                return
            self._ana_log.config(state="normal")
            self._ana_log.insert("end", line + "\n")
            self._ana_log.see("end")
            self._ana_log.config(state="disabled")
            if line.strip() and hasattr(self, "_ana_status"):
                self._ana_status.config(text=line.strip()[:120], fg=MUTED)

        def _thread():
            try:
                proc = subprocess.Popen(
                    [sys.executable, script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    self.root.after(0, _ana_log_line, line.rstrip())
                proc.wait()
                ok = proc.returncode == 0
                def _done():
                    if hasattr(self, "_ana_progress"):
                        self._ana_progress.stop()
                    if ok:
                        if hasattr(self, "_ana_status"):
                            self._ana_status.config(text="✔  Analysis complete.", fg=GREEN)
                        _ana_log_line("\n✔  Analysis complete.")
                    else:
                        if hasattr(self, "_ana_status"):
                            self._ana_status.config(
                                text=f"✗  Failed (exit {proc.returncode}).", fg=RED)
                        _ana_log_line(f"\n✗  Failed (exit code {proc.returncode}).")
                self.root.after(0, _done)
            except Exception as exc:
                def _err():
                    if hasattr(self, "_ana_progress"):
                        self._ana_progress.stop()
                    if hasattr(self, "_ana_status"):
                        self._ana_status.config(text=f"✗  {exc}", fg=RED)
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
            # Find the segmentation output folder
            seg_folder = None
            for v in self._find_variants(SUBFOLDER["seg"]):
                f = os.path.join(proj, v)
                if os.path.isdir(f):
                    seg_folder = f
                    break
            if seg_folder is None:
                return False
            # Find every project subfolder that has a prediction mask
            mask_pat = (self.config.get("val_mask_pattern", "") or "*_predict_original_ch2*.tif").strip()
            projects = [
                os.path.join(seg_folder, d)
                for d in os.listdir(seg_folder)
                if os.path.isdir(os.path.join(seg_folder, d))
                and not d.startswith((".", "DEBUG"))
                and glob.glob(os.path.join(seg_folder, d, mask_pat))
            ]
            if not projects:
                return False
            # Complete only when EVERY project has the validation-complete marker
            return all(
                os.path.exists(os.path.join(p, "ROI Validation Results", "_validation_complete"))
                for p in projects
            )
        if step == "ana":
            # combined_analysis_summary.xlsx is written at the very end of
            # analysis.py to whatever folder was used as the output root.
            # The folder name varies (Analysis, analysis_output, etc.), so
            # search every first-level subdirectory of the project folder.
            if not os.path.isdir(proj):
                return False
            for name in os.listdir(proj):
                folder = os.path.join(proj, name)
                if os.path.isdir(folder) and os.path.exists(
                    os.path.join(folder, "combined_analysis_summary.xlsx")
                ):
                    return True
            return False
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
            f"Model:           {cfg.get('seg_model_type', '?')}  /  {cfg.get('seg_preprocess_mode', '?')}",
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
            text="Runs all enabled steps in order. Steps already marked ✔ are automatically skipped.",
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
        if self._spinning:
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
        out = self._resolve_output_dir(SUBFOLDER["seg"])
        if out is None:
            return
        self._sync_vars_to_config()
        self.config["output_root_folder"] = out
        save_config(self.config)
        self._run_subprocess_in_log(self._script("segmentation.py"), "Segmentation")

    def _run_validation(self) -> None:
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
        subprocess.Popen(
            [sys.executable, self._script("validator.py"), "--config", CONFIG_PATH]
        )

    def _run_analysis(self) -> None:
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set.")
            return

        if self._ana_input_var:
            seg_sub = self._ana_input_var.get()
        else:
            vs = self._find_variants(SUBFOLDER["seg"])
            seg_sub = vs[-1] if vs else SUBFOLDER["seg"]
        ana_input = os.path.join(proj, seg_sub)

        out = self._resolve_output_dir(SUBFOLDER["ana"])
        if out is None:
            return

        self._sync_vars_to_config()
        self.config["ana_microsam_output_root"] = ana_input
        self.config["output_root_folder"] = out
        save_config(self.config)

        self._run_subprocess_in_log(self._script("analysis.py"), "Analysis")

    def _run_all(self) -> None:
        proj = self.config.get("project_output_folder", "").strip()
        if not proj:
            self._log("ERROR: No project output folder set. Go to Step 1 — Create Project.")
            return

        seg_done = self._step_is_done("seg")
        ana_done = self._step_is_done("ana")
        run_seg = not seg_done
        run_ana = bool(self.config.get("ana_enabled")) and not ana_done

        if not run_seg and not run_ana:
            self._log("\n✔  All steps already complete — nothing to run.")
            self._set_run_status("All steps already complete.", GREEN)
            return

        seg_out = None
        if run_seg:
            seg_out = self._resolve_output_dir(SUBFOLDER["seg"])
            if seg_out is None:
                return

        ana_out = None
        if run_ana:
            ana_out = self._resolve_output_dir(SUBFOLDER["ana"])
            if ana_out is None:
                return

        self._sync_vars_to_config()
        if seg_out:
            self.config["output_root_folder"] = seg_out
        save_config(self.config)

        seg_script = self._script("segmentation.py")
        ana_script = self._script("analysis.py")

        self._log(f"\n{'═'*60}")
        self._log("▶▶  Run All")
        self._log(f"{'═'*60}")
        if self.config.get("val_enabled"):
            self._log("ℹ  Validation is interactive — use the Run button on the Validation row.")
        self._set_run_status("Pipeline running…", ACCENT)

        def _thread():
            if run_seg:
                proc1 = subprocess.Popen(
                    [sys.executable, seg_script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc1.stdout:
                    self.root.after(0, self._log, line.rstrip())
                proc1.wait()

                if proc1.returncode != 0:
                    self.root.after(0, self._log, "\n✗  Segmentation failed — stopping.")
                    self.root.after(0, self._set_run_status, "Pipeline failed (segmentation).", RED)
                    return
                self.root.after(0, self._log, "\n✓  Segmentation complete.")
            else:
                self.root.after(0, self._log, "\n✔  Segmentation already complete — skipping.")

            if run_ana and ana_out:
                vs = self._find_variants(SUBFOLDER["seg"])
                seg_sub_for_ana = vs[-1] if vs else SUBFOLDER["seg"]
                self.config["ana_microsam_output_root"] = os.path.join(proj, seg_sub_for_ana)
                self.config["output_root_folder"] = ana_out
                save_config(self.config)

                self.root.after(0, self._log, "\n" + "─"*40)
                self.root.after(0, self._log, "▶  Analysis starting…")
                self.root.after(0, self._set_run_status, "Analysis running…", ACCENT)

                proc2 = subprocess.Popen(
                    [sys.executable, ana_script, "--config", CONFIG_PATH],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc2.stdout:
                    self.root.after(0, self._log, line.rstrip())
                proc2.wait()

                if proc2.returncode == 0:
                    self.root.after(0, self._log, "\n✓  Analysis complete.")
                    self.root.after(0, self._set_run_status, "All steps complete!", GREEN)
                else:
                    self.root.after(0, self._log, "\n✗  Analysis failed.")
                    self.root.after(0, self._set_run_status, "Pipeline failed (analysis).", RED)
            elif self.config.get("ana_enabled") and ana_done:
                self.root.after(0, self._log, "\n✔  Analysis already complete — skipping.")
                self.root.after(0, self._set_run_status, "Pipeline complete.", GREEN)
            else:
                self.root.after(0, self._set_run_status, "Pipeline complete.", GREEN)

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
