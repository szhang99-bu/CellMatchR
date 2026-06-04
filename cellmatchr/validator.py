import os
import re
import json
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, font, filedialog
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table, find_contours
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from glob import glob

# Set Matplotlib to use the TkAgg backend
matplotlib.use('TkAgg')

# =============================================================================
# USER SETTINGS
# =============================================================================

# --- 1. Predicted Mask Configuration ---
# CHANGE THE NAME DEPENDING ON which channel prodiction you want to cellmatchr.
PREDICTED_MASK_GLOB_PATTERN = "*_predict_original_ch2*.tif"
CONFIGURED_VAL_CHANNEL = None   # set by _apply_config when launcher pre-selects a channel

# Where to save the manual validation TSV.
# This folder will be created inside each project folder.
VALIDATION_FOLDER_NAME = "ROI Validation Results"

# --- 2. Channel Configuration ---
# The app now auto-detects how many *_split_ch*.tif files are present in each project folder.
# These templates define display names/colors for common channel numbers.
# ch3 is shown in purple/magenta.
DEFAULT_CHANNEL_TEMPLATES = {
    0: {
        'name': 'DAPI',
        'color': 'PureBlue',
        'rgb_weights': (0.0, 0.0, 1.0),
    },
    1: {
        'name': 'GAD67',
        'color': 'PureGreen',
        'rgb_weights': (0.0, 1.0, 0.0),
    },
    2: {
        'name': 'AAV Cre',
        'color': 'PureRed',
        'rgb_weights': (1.0, 0.0, 0.0),
    },
    3: {
        'name': 'NeuN',
        'color': 'PurePurple',
        'rgb_weights': (1.0, 0.0, 1.0),
    },
}

FALLBACK_CHANNEL_COLORS = [
    ('PureBlue', (0.0, 0.0, 1.0)),
    ('PureGreen', (0.0, 1.0, 0.0)),
    ('PureRed', (1.0, 0.0, 0.0)),
    ('PurePurple', (1.0, 0.0, 1.0)),
    ('Cyan', (0.0, 1.0, 1.0)),
    ('Yellow', (1.0, 1.0, 0.0)),
]

# Names and colors set by the launcher config (keyed by display index)
CHANNEL_NAME_OVERRIDES:  dict = {}
CHANNEL_COLOR_OVERRIDES: dict = {}

# Map user-friendly color labels → internal cmap names
_COLOR_LABEL_TO_INTERNAL = {
    "Blue":   "PureBlue",
    "Green":  "PureGreen",
    "Red":    "PureRed",
    "Purple": "PurePurple",
    "Cyan":   "Cyan",
    "Yellow": "Yellow",
    "White":  "PureWhite",
}

# Map internal cmap name → RGB blend weights used in merged/context views
_COLOR_TO_RGB_WEIGHTS = {
    "PureBlue":   (0.0, 0.0, 1.0),
    "PureGreen":  (0.0, 1.0, 0.0),
    "PureRed":    (1.0, 0.0, 0.0),
    "PurePurple": (1.0, 0.0, 1.0),
    "Cyan":       (0.0, 1.0, 1.0),
    "Yellow":     (1.0, 1.0, 0.0),
    "PureWhite":  (1.0, 1.0, 1.0),
}


# --- 3. Display Settings ---
ROI_PADDING = 50          # "Tight" Zoom
MID_CONTEXT_PADDING = 150 # "2x" Zoom Out
CONTEXT_PADDING = 400     # "Context" Zoom Out
OUTPUT_FILENAME = "roi_verification_results.tsv"
CONFIG_FILE = ".roi_verifier_config"

# --- Customizable decision buttons -------------------------------------------
# Users may rename the decision words and reassign their keyboard shortcuts
# (up to 6).  ARTIFACT is fixed: it is always present, always bound to Shift,
# and nothing else may use Shift.  'Down' is reserved for Save & Next.
VAL_BUTTONS_FILE = os.path.expanduser("~/.cellmatchr/validator_buttons.json")
DEFAULT_DECISIONS = [
    {"word": "YES",   "key": "Left"},
    {"word": "NO",    "key": "Right"},
    {"word": "MAYBE", "key": "Up"},
]
ARTIFACT_WORD = "ARTIFACT"
RESERVED_KEYS = {"Shift", "Shift_L", "Shift_R", "Down"}  # Shift=Artifact, Down=Save&Next
MAX_DECISIONS = 6
DECISION_COLORS = ["#40a02b", "#d20f39", "#fe640b", "#1e88e5", "#8e44ad", "#b8860b"]


def load_decisions():
    """Load the user's custom decision buttons, or the defaults."""
    try:
        if os.path.exists(VAL_BUTTONS_FILE):
            with open(VAL_BUTTONS_FILE) as f:
                cats = json.load(f).get("categories", [])
            out = []
            for c in cats[:MAX_DECISIONS]:
                w = str(c.get("word", "")).strip()
                k = str(c.get("key", "")).strip()
                if w and k and w.upper() != ARTIFACT_WORD and k not in RESERVED_KEYS:
                    out.append({"word": w, "key": k})
            if out:
                for i, c in enumerate(out):
                    c["color"] = DECISION_COLORS[i % len(DECISION_COLORS)]
                return out
    except Exception as e:
        print(f"[validator] Could not read {VAL_BUTTONS_FILE}: {e}")
    out = [dict(d) for d in DEFAULT_DECISIONS]
    for i, c in enumerate(out):
        c["color"] = DECISION_COLORS[i % len(DECISION_COLORS)]
    return out


def save_decisions(cats):
    """Persist the custom decision buttons to ~/.cellmatchr/validator_buttons.json."""
    try:
        os.makedirs(os.path.dirname(VAL_BUTTONS_FILE), exist_ok=True)
        with open(VAL_BUTTONS_FILE, "w") as f:
            json.dump({"categories": [{"word": c["word"], "key": c["key"]} for c in cats]},
                      f, indent=2)
    except Exception as e:
        print(f"[validator] Could not save decision buttons: {e}")


# =============================================================================
# PER-CHANNEL VALIDATION HELPERS
# =============================================================================
def get_output_filename(channel):
    """Return the TSV filename for a given channel number."""
    return f"roi_verification_results_ch{channel}.tsv"


def get_completion_marker(channel):
    """Return the completion-marker filename for a given channel number."""
    return f"_validation_complete_ch{channel}"


def get_predicted_channels(project_folder):
    """Return sorted list of channel numbers that have a predicted mask TIF."""
    seen = set()
    for f in glob(os.path.join(project_folder, "*_predict_original_ch*.tif")):
        m = re.search(r"_predict_original_ch(\d+)", os.path.basename(f))
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def get_channel_validation_status(project_folder, channel):
    """Return 'complete', 'partial', or 'none' for the given channel."""
    val_dir = os.path.join(project_folder, VALIDATION_FOLDER_NAME)
    if os.path.exists(os.path.join(val_dir, get_completion_marker(channel))):
        return "complete"
    if os.path.exists(os.path.join(val_dir, "_validation_complete")):
        return "complete"  # legacy channel-unaware marker
    tsv_path = os.path.join(val_dir, get_output_filename(channel))
    if not os.path.exists(tsv_path):
        tsv_path = os.path.join(val_dir, OUTPUT_FILENAME)  # legacy fallback
    if os.path.exists(tsv_path):
        try:
            n = len(pd.read_csv(tsv_path, sep="\t"))
            return "partial" if n > 0 else "none"
        except Exception:
            pass
    return "none"

# --- 4. Dark Mode Colors ---
DARK_BG = '#2b2b2b'   
PLOT_BG = '#000000'   
FG_COLOR = '#d3d3d3'  
HIGHLIGHT = '#528bff' 
DEFAULT_FONT = ("Arial", 10)

# =============================================================================
# CUSTOM HELPERS
# =============================================================================
def get_custom_cmap(color_name):
    """Creates a LinearSegmentedColormap from Black to the target color."""
    colors = {
        'PureBlue':  ['black', 'blue'],
        'PureGreen': ['black', 'lime'],
        'PureRed':   ['black', 'red'],
        'PurePurple':['black', 'magenta'],
        'Cyan':      ['black', 'cyan'],
        'Magenta':   ['black', 'magenta'],
        'Yellow':    ['black', 'yellow'],
        'PureWhite': ['black', 'white'],
    }
    return LinearSegmentedColormap.from_list(f"BlackTo{color_name}", colors.get(color_name, ['black','white']))



def detect_channel_configs(project_path):
    """
    Auto-detect split channel TIFFs in a project folder and build display configs.

    Looks for files like:
        *_split_ch0.tif
        *_split_ch1.tif
        *_split_ch2.tif
        *_split_ch3.tif

    Returns a list of configs sorted by actual channel number. Each config has:
        channel_index, name, glob_pattern, color, rgb_weights
    """
    split_files = sorted(glob(os.path.join(project_path, "*_split_ch*.tif")))
    detected_channels = []

    for path in split_files:
        name = os.path.basename(path)
        m = re.search(r"_split_ch(\d+)\.tif{1,2}$", name, flags=re.IGNORECASE)
        if m:
            detected_channels.append(int(m.group(1)))

    detected_channels = sorted(set(detected_channels))

    if not detected_channels:
        # Fallback to the historical 3-channel behavior if no split files are found yet.
        detected_channels = [0, 1, 2]

    configs = []

    for display_index, ch in enumerate(detected_channels):
        template = DEFAULT_CHANNEL_TEMPLATES.get(ch)

        if template is None:
            color_name, rgb_weights = FALLBACK_CHANNEL_COLORS[display_index % len(FALLBACK_CHANNEL_COLORS)]
            template = {
                'name': f'Channel {ch}',
                'color': color_name,
                'rgb_weights': rgb_weights,
            }

        # User-configured name and color override the defaults
        name        = CHANNEL_NAME_OVERRIDES.get(display_index, template['name'])
        color_label = CHANNEL_COLOR_OVERRIDES.get(display_index)
        if color_label:
            color       = _COLOR_LABEL_TO_INTERNAL.get(color_label, template['color'])
            rgb_weights = _COLOR_TO_RGB_WEIGHTS.get(color, template['rgb_weights'])
        else:
            color       = template['color']
            rgb_weights = template['rgb_weights']

        configs.append({
            'channel_index': ch,
            'name': name,
            'glob_pattern': f'*_split_ch{ch}.tif',
            'color': color,
            'rgb_weights': rgb_weights,
        })

    print(f"Detected {len(configs)} channel(s) in {os.path.basename(project_path)}: "
          f"{[c['channel_index'] for c in configs]}")

    return configs

def setup_dark_theme(root):
    root.configure(bg=DARK_BG)
    s = ttk.Style()
    s.theme_use('default')
    s.configure('.', background=DARK_BG, foreground=FG_COLOR, font=DEFAULT_FONT, fieldbackground=DARK_BG, borderwidth=0)
    s.configure('TFrame', background=DARK_BG)
    s.configure('TLabel', background=DARK_BG, foreground=FG_COLOR)
    s.configure('TButton', background='#3c3c3c', foreground=FG_COLOR, font=("Arial", 11, "bold"), padding=10)
    s.map('TButton', background=[('active', HIGHLIGHT)], foreground=[('active', 'white')])
    s.configure('TEntry', fieldbackground='#3c3c3c', foreground=FG_COLOR, insertcolor=FG_COLOR)
    s.configure('TLabelFrame', background=DARK_BG, foreground=FG_COLOR)
    s.configure('TLabelFrame.Label', background=DARK_BG, foreground=FG_COLOR, font=("Arial", 11, "bold"))
    root.option_add('*Listbox.background', '#3c3c3c')
    root.option_add('*Listbox.foreground', FG_COLOR)
    root.option_add('*Listbox.selectBackground', HIGHLIGHT)
    root.option_add('*Text.background', '#3c3c3c')
    root.option_add('*Text.foreground', FG_COLOR)
    # Scale widget dark mode colors
    root.option_add('*Scale.background', DARK_BG)
    root.option_add('*Scale.foreground', FG_COLOR)
    root.option_add('*Scale.troughColor', '#404040')
    root.option_add('*Scale.activeBackground', HIGHLIGHT)
    
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = PLOT_BG
    plt.rcParams['figure.facecolor'] = DARK_BG
    plt.rcParams['savefig.facecolor'] = DARK_BG

# =============================================================================
# ROI VERIFIER CLASS
# =============================================================================
class RoiVerifierApp:
    def __init__(self, root, project_path, filtered_mask_path, channel_file_paths, output_file_path, channel_configs=None, on_close_callback=None, val_channel=None):
        self.root = root
        ch_suffix = f" — Channel {val_channel}" if val_channel is not None else ""
        self.root.title(f"ROI Verifier - {os.path.basename(project_path)}{ch_suffix}")
        self.on_close_callback = on_close_callback
        self.val_channel = val_channel
        setup_dark_theme(self.root)

        # Dynamic channel configuration for this project
        self.channel_config = channel_configs if channel_configs is not None else detect_channel_configs(project_path)
        self.n_channels = len(self.channel_config)
        
        # Data
        self.props = []
        self.channel_images = {}
        self.filtered_mask = None
        self.current_roi_index = 0
        
        # Paths
        self.project_path = project_path
        self.filtered_mask_path = filtered_mask_path
        self.channel_file_paths = channel_file_paths
        self.output_file_path = output_file_path
        
        # State for Sliders and Visibility
        self.sliders_initialized = False
        self.slider_widgets = []
        self.slider_vars = []
        self.slider_entries = []            # Entry widgets for typed intensity
        self._slider_entry_syncing = False  # guard against circular updates
        self.is_updating = False
        
        # VISIBILITY FLAGS
        self.show_contours_global = True
        self.show_contours_channel = {i: True for i in range(self.n_channels)}

        # Layout Main
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # --- Plots Container ---
        self.plot_container = ttk.Frame(self.main_frame)
        self.plot_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # 1. Slider Frame (Left)
        self.slider_frame = ttk.Frame(self.plot_container, width=90)
        self.slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # --- GLOBAL HIDE BUTTON ---
        btn_hide_all = ttk.Button(self.slider_frame, text="Hide\nALL", width=4)
        btn_hide_all.bind('<ButtonPress-1>', lambda e: self.toggle_global_roi(False))
        btn_hide_all.bind('<ButtonRelease-1>', lambda e: self.toggle_global_roi(True))
        btn_hide_all.pack(side=tk.TOP, pady=(0, 10))
        
        # Create one vertical slider + typed entry + hide button per detected channel.
        # Slider length is adaptive so all 4 channels fit without being cut off.
        _slider_len = max(55, min(100, 320 // max(self.n_channels, 1)))

        for i in range(self.n_channels):
            # Frame for each slider group
            sf = ttk.Frame(self.slider_frame)
            sf.pack(side=tk.TOP, expand=True, fill=tk.BOTH, pady=2)

            # Channel name label
            name = self.channel_config[i]['name'] if i < len(self.channel_config) else f"Ch{i}"
            ttk.Label(sf, text=f"{name}\nMax", font=("Arial", 7),
                      anchor="center", justify="center").pack(side=tk.TOP, pady=(2, 0))

            # Slider variable
            sv = tk.DoubleVar(value=0)
            self.slider_vars.append(sv)

            # Slider widget — showvalue=0 because the entry below shows the value
            s = tk.Scale(sf, from_=255, to=0, orient=tk.VERTICAL, variable=sv,
                         showvalue=0, resolution=1, length=_slider_len,
                         command=self.on_slider_change)
            s.pack(side=tk.TOP, expand=True, fill=tk.Y)
            self.slider_widgets.append(s)

            # ── Typed-value entry ──────────────────────────────────────────
            # Shows the current slider value and accepts typed input.
            # Press Enter (or Tab away) to apply the typed value.
            entry = tk.Entry(sf, width=7, justify="center",
                             bg="#3c3c3c", fg=FG_COLOR,
                             insertbackground=FG_COLOR,
                             relief="flat", font=("Arial", 8))
            entry.pack(side=tk.TOP, pady=(1, 2))
            entry.insert(0, "0")
            self.slider_entries.append(entry)
            entry.bind("<Return>",   lambda e, idx=i: self._apply_slider_entry(idx))
            entry.bind("<FocusOut>", lambda e, idx=i: self._apply_slider_entry(idx))

            # Channel Hide button
            btn_hide = ttk.Button(sf, text="Hide\nROI", width=4)
            btn_hide.bind("<ButtonPress-1>",   lambda e, ch=i: self.toggle_channel_roi(ch, False))
            btn_hide.bind("<ButtonRelease-1>", lambda e, ch=i: self.toggle_channel_roi(ch, True))
            btn_hide.pack(side=tk.TOP, pady=2)

        # 2. Canvas Frame (Right)
        self.canvas_frame = ttk.Frame(self.plot_container)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Matplotlib Setup ---
        # Use a conservative initial size; the canvas resizes dynamically with the window.
        fig_height = max(6, 2.0 * self.n_channels + 2)
        self.fig = plt.figure(figsize=(10, fig_height))
        self.fig.patch.set_facecolor(DARK_BG)

        gs = GridSpec(
            self.n_channels,
            4,
            figure=self.fig,
            width_ratios=[1, 1, 1, 1.5],
            wspace=0.05,
            hspace=0.08
        )

        # Columns 0-2: one row per detected channel
        self.col_axes = []
        self.gray_axes = []
        self.mid_axes = []

        for i in range(self.n_channels):
            self.col_axes.append(self.fig.add_subplot(gs[i, 0]))
            self.gray_axes.append(self.fig.add_subplot(gs[i, 1]))
            self.mid_axes.append(self.fig.add_subplot(gs[i, 2]))

        # Column 3: merged + context views.
        split_row = max(1, self.n_channels // 2)
        self.ax_merge = self.fig.add_subplot(gs[0:split_row, 3])
        self.ax_context = self.fig.add_subplot(gs[split_row:self.n_channels, 3])

        # Tight fixed margins — avoids large blank gaps around subplots
        self.fig.subplots_adjust(
            left=0.01, right=0.99, top=0.96, bottom=0.01,
            wspace=0.04, hspace=0.06,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        _canvas_widget = self.canvas.get_tk_widget()
        _canvas_widget.configure(background=DARK_BG)
        _canvas_widget.pack(fill=tk.BOTH, expand=True)
        # matplotlib's native <Configure> handler resizes the PhotoImage buffer correctly;
        # do NOT replace it with a custom bind — just let it run.

        # --- Controls Frame (Right Side of GUI) ---
        self.control_frame = ttk.Frame(self.main_frame, padding=10)
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, minsize=220, weight=0)
        
        # Configure a slightly larger button style for the control panel
        _s = ttk.Style()
        _s.configure("Ctrl.TButton",
                      font=("Arial", 12),
                      anchor="center",
                      padding=(10, 10))
        _s.configure("CtrlNav.TButton",
                      font=("Arial", 11),
                      anchor="center",
                      padding=(10, 8))

        self.control_frame.grid_columnconfigure(0, weight=1)

        # Widgets
        row = 0
        self.info_label = ttk.Label(self.control_frame, text="Loading...", font=("Arial", 14, "bold"), anchor="center")
        self.info_label.grid(row=row, column=0, pady=10, sticky="ew"); row+=1
        ttk.Separator(self.control_frame).grid(row=row, column=0, sticky="ew", pady=10); row+=1

        ttk.Label(self.control_frame, text="Decision:", font=("Arial", 12)).grid(row=row, column=0, sticky="w", pady=(0, 5)); row+=1
        self.decision_var = tk.StringVar()

        # Decision buttons live in their own frame so they can be rebuilt when
        # the user customizes the words / keys.
        self.decisions = load_decisions()
        self._bound_keys = []
        self._decision_box = ttk.Frame(self.control_frame)
        self._decision_box.grid(row=row, column=0, sticky="ew"); row+=1
        self._decision_box.grid_columnconfigure(0, weight=1)
        self._build_decision_buttons()

        # Hidden-ish "customize" link (change words / keys)
        self._customize_lbl = ttk.Label(
            self.control_frame, text="⚙  Customize buttons…",
            font=("Arial", 9, "underline"), foreground=HIGHLIGHT, cursor="hand2")
        self._customize_lbl.grid(row=row, column=0, sticky="e", pady=(2, 0)); row+=1
        self._customize_lbl.bind("<Button-1>", lambda e: self._open_customize_dialog())

        self.decision_status_label = ttk.Label(self.control_frame, text="No decision made", font=("Arial", 10, "italic"), anchor="center")
        self.decision_status_label.grid(row=row, column=0, sticky="ew", pady=8); row+=1
        ttk.Separator(self.control_frame).grid(row=row, column=0, sticky="ew", pady=8); row+=1

        ttk.Label(self.control_frame, text="Notes:", font=("Arial", 12)).grid(row=row, column=0, sticky="w", pady=(0, 5)); row+=1
        self.notes_text = tk.Text(self.control_frame, height=8, width=20, wrap="word")
        self.notes_text.grid(row=row, column=0, sticky="ew"); row+=1

        self.nav_frame = ttk.Frame(self.control_frame)
        self.nav_frame.grid(row=row, column=0, sticky="ew", pady=(16, 0))
        self.nav_frame.grid_columnconfigure(0, weight=1)

        self.btn_back = ttk.Button(self.nav_frame, text="Go Back",
                                   style="CtrlNav.TButton", command=self.go_back)
        self.btn_back.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.save_btn = ttk.Button(self.nav_frame, text="Save & Next (Down)",
                                   style="CtrlNav.TButton", command=self.save_and_next)
        self.save_btn.grid(row=1, column=0, sticky="ew")
        
        # Keys — decision shortcuts are ignored while the Notes box has focus so
        # typing notes never triggers an accidental decision.  Bindings are built
        # from the (customizable) decision list; Artifact is always Shift and
        # Save & Next is always Down.
        self._rebind_decision_keys()

        self.load_data()

    def _typing_in_notes(self) -> bool:
        """True when keyboard focus is inside the Notes text box."""
        try:
            return self.root.focus_get() is self.notes_text
        except Exception:
            return False

    def _key_decision(self, decision):
        """Keyboard shortcut wrapper — ignored while the user is typing notes."""
        if self._typing_in_notes():
            return
        self.make_decision(decision)

    def _key_save_and_next(self):
        """Keyboard wrapper for Save & Next — ignored while typing notes."""
        if self._typing_in_notes():
            return
        self.save_and_next()

    def make_decision(self, decision):
        self.decision_var.set(decision)
        cmap = {c["word"]: c["color"] for c in self.decisions}
        cmap[ARTIFACT_WORD] = "#8e44ad"
        self.decision_status_label.config(
            text=f"Marked as {decision}", foreground=cmap.get(decision, FG_COLOR))

    # ── Customizable decision buttons ────────────────────────────────────────
    def _build_decision_buttons(self):
        """(Re)create the decision buttons from self.decisions plus the fixed
        ARTIFACT button, and (re)bind their keyboard shortcuts."""
        for w in self._decision_box.winfo_children():
            w.destroy()
        self._decision_btns = {}
        r = 0
        for cat in self.decisions:
            word, key = cat["word"], cat["key"]
            b = ttk.Button(self._decision_box, text=f"{word}  ({key})",
                           style="Ctrl.TButton",
                           command=lambda w=word: self.make_decision(w))
            b.grid(row=r, column=0, sticky="ew", pady=3); r += 1
            self._decision_btns[word] = b
        # Fixed ARTIFACT button — always present, always Shift.
        b = ttk.Button(self._decision_box, text="Artifact  (Shift)",
                       style="Ctrl.TButton",
                       command=lambda: self.make_decision(ARTIFACT_WORD))
        b.grid(row=r, column=0, sticky="ew", pady=3)
        self._decision_btns[ARTIFACT_WORD] = b
        if hasattr(self, "load_data"):   # only rebind after root exists
            self._rebind_decision_keys()

    def _rebind_decision_keys(self):
        """Bind each decision's key; ARTIFACT→Shift and Save&Next→Down are fixed."""
        for k in getattr(self, "_bound_keys", []):
            try:
                self.root.unbind(f"<{k}>")
            except Exception:
                pass
        self._bound_keys = []
        for cat in self.decisions:
            key, word = cat["key"], cat["word"]
            if key in RESERVED_KEYS:
                continue
            try:
                self.root.bind(f"<{key}>", lambda e, w=word: self._key_decision(w))
                self._bound_keys.append(key)
            except Exception as ex:
                print(f"[validator] Could not bind key '{key}': {ex}")
        # Fixed bindings
        self.root.bind('<Shift_L>', lambda e: self._key_decision(ARTIFACT_WORD))
        self.root.bind('<Shift_R>', lambda e: self._key_decision(ARTIFACT_WORD))
        self.root.bind('<Down>',    lambda e: self._key_save_and_next())

    def _open_customize_dialog(self):
        """Editor to rename decision words and reassign keys (max 6)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Customize decision buttons")
        setup_dark_theme(dlg)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(
            dlg, justify="left", font=("Arial", 10),
            text=("Rename the decision words and set each shortcut key (up to 6).\n"
                  "Keys: arrow names (Left / Right / Up) or a single letter (a, n, …).\n\n"
                  "Reserved — cannot be reused:  Shift = Artifact   ·   Down = Save & Next.\n"
                  "The Artifact button is always present and always bound to Shift."),
        ).grid(row=0, column=0, columnspan=3, padx=14, pady=(14, 8), sticky="w")

        hdr = ttk.Frame(dlg); hdr.grid(row=1, column=0, columnspan=3, padx=14, sticky="w")
        ttk.Label(hdr, text="Word", font=("Arial", 10, "bold"), width=18).pack(side=tk.LEFT)
        ttk.Label(hdr, text="Key",  font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT)

        rows_frame = ttk.Frame(dlg); rows_frame.grid(row=2, column=0, columnspan=3, padx=14, sticky="ew")
        cust_rows = []

        def add_row(word="", key=""):
            if len(cust_rows) >= MAX_DECISIONS:
                return
            rf = ttk.Frame(rows_frame); rf.pack(fill="x", pady=2)
            wv, kv = tk.StringVar(value=word), tk.StringVar(value=key)
            ttk.Entry(rf, textvariable=wv, width=18).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Entry(rf, textvariable=kv, width=12).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Button(rf, text="✕", width=3,
                       command=lambda: remove_row(rf)).pack(side=tk.LEFT)
            cust_rows.append((wv, kv, rf))
            _update_add_state()

        def remove_row(rf):
            if len(cust_rows) <= 1:
                messagebox.showwarning("Keep one", "You need at least one decision button.", parent=dlg)
                return
            cust_rows[:] = [t for t in cust_rows if t[2] is not rf]
            rf.destroy()
            _update_add_state()

        add_btn = ttk.Button(dlg, text="＋  Add button", command=lambda: add_row())
        def _update_add_state():
            add_btn.config(state=(tk.NORMAL if len(cust_rows) < MAX_DECISIONS else tk.DISABLED))

        for c in self.decisions:
            add_row(c["word"], c["key"])

        # Locked Artifact row (display only)
        artf = ttk.Frame(dlg); artf.grid(row=3, column=0, columnspan=3, padx=14, pady=(6, 0), sticky="w")
        ttk.Label(artf, text="ARTIFACT", width=18, foreground="#8e44ad",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(artf, text="Shift  (fixed)", width=12, foreground="#888888").pack(side=tk.LEFT)

        add_btn.grid(row=4, column=0, columnspan=3, padx=14, pady=(10, 6), sticky="w")
        _update_add_state()

        def do_save():
            cats, seen_w, seen_k = [], set(), set()
            for wv, kv, _ in cust_rows:
                w, k = wv.get().strip(), kv.get().strip()
                if not w or not k:
                    messagebox.showwarning("Incomplete", "Every button needs a word and a key.", parent=dlg); return
                if w.upper() == ARTIFACT_WORD:
                    messagebox.showwarning("Reserved", "'ARTIFACT' is reserved — it always exists.", parent=dlg); return
                if k in RESERVED_KEYS:
                    messagebox.showwarning("Reserved key", f"'{k}' is reserved (Shift = Artifact, Down = Save & Next).", parent=dlg); return
                if w.upper() in seen_w:
                    messagebox.showwarning("Duplicate word", f"'{w}' is used more than once.", parent=dlg); return
                if k in seen_k:
                    messagebox.showwarning("Duplicate key", f"Key '{k}' is used more than once.", parent=dlg); return
                seen_w.add(w.upper()); seen_k.add(k)
                cats.append({"word": w, "key": k})
            if not (1 <= len(cats) <= MAX_DECISIONS):
                messagebox.showwarning("Count", f"Use between 1 and {MAX_DECISIONS} buttons.", parent=dlg); return
            for i, c in enumerate(cats):
                c["color"] = DECISION_COLORS[i % len(DECISION_COLORS)]
            self.decisions = cats
            save_decisions(cats)
            self._build_decision_buttons()
            dlg.destroy()

        btns = ttk.Frame(dlg); btns.grid(row=5, column=0, columnspan=3, padx=14, pady=(8, 14), sticky="e")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Save", command=do_save).pack(side=tk.LEFT)

    def on_slider_change(self, val):
        if not self.is_updating and not self._slider_entry_syncing:
            self._sync_entries_from_sliders()
            self.update_roi_display(recalc_roi=False)

    def _sync_entries_from_sliders(self) -> None:
        """Update all entry widgets to reflect current slider variable values."""
        for var, entry in zip(self.slider_vars, self.slider_entries):
            try:
                entry.delete(0, tk.END)
                entry.insert(0, str(int(var.get())))
            except tk.TclError:
                pass

    def _apply_slider_entry(self, idx: int, event=None) -> None:
        """
        User pressed Enter (or tabbed away) from a slider entry box.
        Parse the typed value, clamp to the slider's valid range, and apply.
        Works for any dynamic range — slide-scanner images with values < 100
        are handled exactly since the user can type the precise number.
        """
        if self._slider_entry_syncing or idx >= len(self.slider_entries):
            return
        entry  = self.slider_entries[idx]
        var    = self.slider_vars[idx]
        slider = self.slider_widgets[idx]
        try:
            val = int(entry.get().strip())
            lo  = int(slider.cget("to"))      # minimum (always 0)
            hi  = int(slider.cget("from"))    # maximum (255 / 4095 / 65535)
            val = max(lo, min(hi, val))
        except (ValueError, tk.TclError):
            val = int(var.get())              # revert to current on bad input
        self._slider_entry_syncing = True
        var.set(val)
        try:
            entry.delete(0, tk.END)
            entry.insert(0, str(val))
        except tk.TclError:
            pass
        self._slider_entry_syncing = False
        if not self.is_updating:
            self.update_roi_display(recalc_roi=False)
            
    def toggle_channel_roi(self, channel_idx, state):
        if self.show_contours_channel[channel_idx] != state:
            self.show_contours_channel[channel_idx] = state
            self.update_roi_display(recalc_roi=False)

    def toggle_global_roi(self, state):
        if self.show_contours_global != state:
            self.show_contours_global = state
            self.update_roi_display(recalc_roi=False)

    def finish_session(self):
        """Called when all ROIs are done."""
        try:
            val_dir = os.path.join(self.project_path, VALIDATION_FOLDER_NAME)
            os.makedirs(val_dir, exist_ok=True)
            marker = (
                get_completion_marker(self.val_channel)
                if self.val_channel is not None
                else "_validation_complete"
            )
            with open(os.path.join(val_dir, marker), "w") as f:
                f.write("done")
        except Exception:
            pass
        messagebox.showinfo("Complete", "All ROIs verified.")
        if self.on_close_callback:
            self.on_close_callback()
        else:
            self.root.destroy()

    def load_data(self):
        try:
            print(f"Loading mask: {self.filtered_mask_path}")
            self.filtered_mask = tifffile.imread(self.filtered_mask_path)
            
            print("Calculating ROI properties...")
            props_df = pd.DataFrame(regionprops_table(self.filtered_mask, properties=('label', 'bbox')))
            if props_df.empty: raise ValueError("No ROIs found.")
            self.props = props_df.to_dict('records')
            self.total_rois = len(self.props)
            
            print("Loading channel images...")
            global_max = 0
            for i, config in enumerate(self.channel_config):
                img_path = self.channel_file_paths.get(i)
                if img_path and os.path.exists(img_path):
                    img = tifffile.imread(img_path)
                    self.channel_images[i] = img
                    if img.max() > global_max: global_max = img.max()
                else:
                    self.channel_images[i] = np.zeros_like(self.filtered_mask)

            # Determine Bit Depth for Slider Range
            if global_max > 4095:
                slider_max = 65535 # 16-bit
            elif global_max > 255:
                slider_max = 4095  # 12-bit
            else:
                slider_max = 255   # 8-bit
            
            for s in self.slider_widgets:
                s.config(from_=slider_max, to=0)

            if not os.path.exists(self.output_file_path):
                with open(self.output_file_path, 'w') as f: f.write("ROI_ID\tDecision\tNote\n")
            else:
                try:
                    df_results = pd.read_csv(self.output_file_path, sep='\t')
                    self.current_roi_index = len(df_results)
                except: self.current_roi_index = 0

            if self.current_roi_index >= self.total_rois:
                self.finish_session()
                return
            
            self.update_roi_display(recalc_roi=True)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            if self.on_close_callback: self.on_close_callback()

    def update_roi_display(self, recalc_roi=True):
        if self.current_roi_index >= self.total_rois:
            self.finish_session()
            return
        
        self.is_updating = True 
        self.info_label.config(text=f"ROI {self.current_roi_index + 1} / {self.total_rois}")
        
        roi = self.props[self.current_roi_index]
        roi_label = roi['label']
        min_r, min_c, max_r, max_c = roi['bbox-0'], roi['bbox-1'], roi['bbox-2'], roi['bbox-3']
        h, w = self.filtered_mask.shape
        
        # Define Slices
        r1_z, r2_z = max(0, min_r - ROI_PADDING), min(h, max_r + ROI_PADDING)
        c1_z, c2_z = max(0, min_c - ROI_PADDING), min(w, max_c + ROI_PADDING)
        slice_zoom = np.s_[r1_z:r2_z, c1_z:c2_z]
        extent_zoom = [c1_z, c2_z, r2_z, r1_z]
        
        r1_m, r2_m = max(0, min_r - MID_CONTEXT_PADDING), min(h, max_r + MID_CONTEXT_PADDING)
        c1_m, c2_m = max(0, min_c - MID_CONTEXT_PADDING), min(w, max_c + MID_CONTEXT_PADDING)
        slice_mid = np.s_[r1_m:r2_m, c1_m:c2_m]
        extent_mid = [c1_m, c2_m, r2_m, r1_m]
        
        r1_c, r2_c = max(0, min_r - CONTEXT_PADDING), min(h, max_r + CONTEXT_PADDING)
        c1_c, c2_c = max(0, min_c - CONTEXT_PADDING), min(w, max_c + CONTEXT_PADDING)
        slice_ctx = np.s_[r1_c:r2_c, c1_c:c2_c]
        extent_ctx = [c1_c, c2_c, r2_c, r1_c]

        # Contour
        roi_mask = (self.filtered_mask == roi_label)
        mask_z = roi_mask[slice_zoom]
        ct_z = find_contours(mask_z, 0.5)
        contour_z = ct_z[0] if ct_z else np.array([[0,0]])
        
        # Additive Buffers
        h_z, w_z = (r2_z - r1_z), (c2_z - c1_z)
        h_c, w_c = (r2_c - r1_c), (c2_c - c1_c)
        merged_zoom_rgb = np.zeros((h_z, w_z, 3), dtype=float)
        merged_ctx_rgb = np.zeros((h_c, w_c, 3), dtype=float)

        # Clear Axes
        all_axes = self.col_axes + self.gray_axes + self.mid_axes + [self.ax_merge, self.ax_context]
        for ax in all_axes:
            ax.clear()
            ax.set_facecolor(PLOT_BG)
        
        # Loop Channels
        for i, ax_col, ax_gray, ax_mid in zip(range(self.n_channels), self.col_axes, self.gray_axes, self.mid_axes):
            if i < len(self.channel_config):
                cfg = self.channel_config[i]
                img_full = self.channel_images[i]
                img_m = img_full[slice_mid]
                img_z = img_full[slice_zoom]
                
                vmin = img_m.min()
                if not self.sliders_initialized:
                    auto_max = img_m.max()
                    self.slider_vars[i].set(auto_max)
                vmax = self.slider_vars[i].get()
                if vmax <= vmin: vmax = vmin + 1
                
                norm = plt.Normalize(vmin, vmax) 
                cmap = get_custom_cmap(cfg['color'])
                
                # Plot 1x Colored
                ax_col.imshow(img_z, cmap=cmap, norm=norm, extent=extent_zoom)
                if self.show_contours_global and self.show_contours_channel[i]:
                    ax_col.plot(contour_z[:, 1] + c1_z, contour_z[:, 0] + r1_z, c='white', linewidth=1, linestyle=':')
                
                ax_col.set_title(cfg['name'])
                ax_col.axis('off')
                
                # Plot 1x Gray
                ax_gray.imshow(img_z, cmap='gray', norm=norm, extent=extent_zoom)
                if self.show_contours_global and self.show_contours_channel[i]:
                    ax_gray.plot(contour_z[:, 1] + c1_z, contour_z[:, 0] + r1_z, c='yellow', linewidth=1, linestyle=':')
                
                ax_gray.set_title(f"{cfg['name']} (Gray)")
                ax_gray.axis('off')
                
                # Plot 2x Gray
                ax_mid.imshow(img_m, cmap='gray', norm=norm, extent=extent_mid)
                if self.show_contours_global and self.show_contours_channel[i]:
                    ax_mid.plot(contour_z[:, 1] + c1_z, contour_z[:, 0] + r1_z, c='yellow', linewidth=1.5)
                
                ax_mid.set_title(f"{cfg['name']} (2x)")
                ax_mid.axis('off')
                
                # Merging
                norm_val_z = np.clip((img_z.astype(float) - vmin) / (vmax - vmin + 1e-9), 0, 1)
                img_c = img_full[slice_ctx]
                vmin_c = img_c.min() 
                norm_val_c = np.clip((img_c.astype(float) - vmin_c) / (vmax - vmin_c + 1e-9), 0, 1)
                rgb_weights = cfg.get('rgb_weights', (1.0, 1.0, 1.0))
                for rgb_idx, weight in enumerate(rgb_weights):
                    if weight != 0:
                        merged_zoom_rgb[..., rgb_idx] += norm_val_z * weight
                        merged_ctx_rgb[..., rgb_idx] += norm_val_c * weight
        
        self.sliders_initialized = True
        # Populate entry boxes with the auto-detected max values on first load
        self._sync_entries_from_sliders()

        # Display Merged
        merged_zoom_rgb = np.clip(merged_zoom_rgb, 0, 1)
        merged_ctx_rgb = np.clip(merged_ctx_rgb, 0, 1)

        self.ax_merge.set_title("Merged View (Additive)")
        self.ax_merge.imshow(merged_zoom_rgb, extent=extent_zoom)
        if self.show_contours_global:
            self.ax_merge.plot(contour_z[:, 1] + c1_z, contour_z[:, 0] + r1_z, c='yellow', linewidth=1.5)
        self.ax_merge.set_xlim(c1_z, c2_z); self.ax_merge.set_ylim(r2_z, r1_z)
        self.ax_merge.axis('off')

        self.ax_context.set_title("Context View")
        self.ax_context.imshow(merged_ctx_rgb, extent=extent_ctx)
        if self.show_contours_global:
            self.ax_context.plot(contour_z[:, 1] + c1_z, contour_z[:, 0] + r1_z, c='yellow', linewidth=2)
        self.ax_context.set_xlim(c1_c, c2_c); self.ax_context.set_ylim(r2_c, r1_c)
        self.ax_context.axis('off')

        self.canvas.draw()
        self.is_updating = False
        
        if recalc_roi:
            self.decision_var.set("")
            self.decision_status_label.config(text="No decision made", foreground=FG_COLOR)
            self.notes_text.delete("1.0", tk.END)

    def save_and_next(self, event=None):
        decision = self.decision_var.get()
        if not decision:
            messagebox.showwarning("No Decision", "Make a decision first.")
            return
        roi_id = self.props[self.current_roi_index]['label']
        notes = self.notes_text.get("1.0", tk.END).strip().replace("\n", " ").replace("\t", " ")
        try:
            with open(self.output_file_path, 'a') as f: f.write(f"{roi_id}\t{decision}\t{notes}\n")
        except Exception as e: messagebox.showerror("Error", str(e)); return
        self.current_roi_index += 1
        self.update_roi_display(recalc_roi=True)

    def go_back(self, event=None):
        if self.current_roi_index > 0:
            self.current_roi_index -= 1
            
            try:
                df = pd.read_csv(self.output_file_path, sep='\t')
                target_label = self.props[self.current_roi_index]['label']
                row = df[df['ROI_ID'] == target_label]
                if not row.empty:
                    prev_decision = str(row.iloc[0]['Decision'])
                    prev_note = row.iloc[0]['Note']
                    self.make_decision(prev_decision)
                    self.notes_text.delete("1.0", tk.END)
                    if pd.notna(prev_note) and str(prev_note).strip() != "":
                        self.notes_text.insert("1.0", str(prev_note))
                
                df = df[df['ROI_ID'] != target_label]
                df.to_csv(self.output_file_path, sep='\t', index=False)
                
            except Exception as e:
                print(f"Error restoring state: {e}")

            self.update_roi_display(recalc_roi=False)

# =============================================================================
# PROJECT CHOOSER
# =============================================================================
class ProjectChooser:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Chooser")
        setup_dark_theme(self.root)
        self.main_folder = ""
        self.projects = []
        
        # --- BIND CLOSE EVENT TO CLEAN EXIT ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.main_frame = ttk.Frame(root, padding=10); self.main_frame.pack(fill=tk.BOTH, expand=True)
        f_frame = ttk.Frame(self.main_frame); f_frame.pack(fill=tk.X, pady=5)
        ttk.Label(f_frame, text="Main Folder:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="Loading...")
        ttk.Entry(f_frame, textvariable=self.path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(f_frame, text="Browse", command=self.browse).pack(side=tk.LEFT)
        self.listbox = tk.Listbox(self.main_frame, height=15); self.listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox.bind('<Double-1>', self.launch)
        self.btn = ttk.Button(self.main_frame, text="Launch", command=self.launch, state=tk.DISABLED); self.btn.pack(fill=tk.X)
        self.root.geometry("800x500"); self.load_last()

        # Pop in front of the CellMatchR launcher (which spawned this as a
        # subprocess) so the window is not hidden behind the main menu.
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.root.after(600, lambda: self.root.attributes("-topmost", False))

    def on_close(self):
        """Forces the python process to stop the mainloop."""
        self.root.quit()
        self.root.destroy()

    def load_last(self):
        if os.path.exists(CONFIG_FILE):
            try: 
                with open(CONFIG_FILE) as f: self.main_folder = f.read().strip()
                self.path_var.set(self.main_folder); self.find()
            except: pass

    def browse(self):
        self.root.lift()
        self.root.attributes("-topmost", True)
        f = filedialog.askdirectory(parent=self.root)
        self.root.attributes("-topmost", False)
        if f: self.main_folder = f; self.path_var.set(f); self.find(); open(CONFIG_FILE, 'w').write(f)

    def find(self):
        self.listbox.delete(0, tk.END)
        self.projects = []

        if not os.path.isdir(self.main_folder):
            return

        for d in sorted(os.listdir(self.main_folder)):
            p = os.path.join(self.main_folder, d)

            if not os.path.isdir(p):
                continue

            if d.startswith("DEBUG") or d.startswith("."):
                continue

            # Try multi-channel discovery first
            predicted_chs = get_predicted_channels(p)

            if predicted_chs:
                validation_dir = os.path.join(p, VALIDATION_FOLDER_NAME)
                os.makedirs(validation_dir, exist_ok=True)

                status_parts = []
                for ch in predicted_chs:
                    s = get_channel_validation_status(p, ch)
                    icon = "✓" if s == "complete" else ("⟳" if s == "partial" else "✗")
                    status_parts.append(f"ch{ch}:{icon}")
                status_str = "  ".join(status_parts)

                self.projects.append({"path": p, "channels": predicted_chs})
                self.listbox.insert(tk.END, f"{d}  [{status_str}]")

            else:
                # Fall back to PREDICTED_MASK_GLOB_PATTERN (legacy / config-overridden pattern)
                mask_candidates = sorted(glob(os.path.join(p, PREDICTED_MASK_GLOB_PATTERN)))
                mask_candidates = [
                    m for m in mask_candidates
                    if "coverage" not in os.path.basename(m).lower()
                    and "combined" not in os.path.basename(m).lower()
                    and "debug" not in os.path.basename(m).lower()
                    and "unfiltered" not in os.path.basename(m).lower()
                ]
                if not mask_candidates:
                    continue

                predicted_mask_path = mask_candidates[-1]
                validation_dir = os.path.join(p, VALIDATION_FOLDER_NAME)
                os.makedirs(validation_dir, exist_ok=True)
                out = os.path.join(validation_dir, OUTPUT_FILENAME)
                stat = "Pending"
                if os.path.exists(out):
                    try:
                        stat = f"In Progress ({len(pd.read_csv(out, sep=chr(9)))})"
                    except Exception:
                        pass
                self.projects.append({"path": p, "mask": predicted_mask_path, "out": out, "channels": None})
                self.listbox.insert(tk.END, f"{d}  [{stat}]")

        if self.projects:
            self.btn.config(state=tk.NORMAL)
        else:
            self.btn.config(state=tk.DISABLED)

    def _select_channel_dialog(self, project_path, channels):
        """Prompt user to pick which channel to validate. Returns channel number or None."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Select Channel to Validate")
        setup_dark_theme(dlg)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        result = [None]

        ttk.Label(dlg, text="Select a channel to validate:", font=("Arial", 12, "bold")).pack(padx=20, pady=(15, 8))

        for ch in channels:
            status = get_channel_validation_status(project_path, ch)
            if status == "complete":
                icon, fg = "✓  Complete", "green"
            elif status == "partial":
                icon, fg = "⟳  In Progress", "orange"
            else:
                icon, fg = "✗  Not Started", FG_COLOR

            row = ttk.Frame(dlg)
            row.pack(fill=tk.X, padx=20, pady=3)

            def _pick(c=ch):
                result[0] = c
                dlg.destroy()

            ttk.Button(row, text=f"Channel {ch}", command=_pick, width=12).pack(side=tk.LEFT)
            tk.Label(row, text=f"  {icon}", fg=fg, bg=DARK_BG, font=("Arial", 11)).pack(side=tk.LEFT)

        ttk.Button(dlg, text="Cancel", command=dlg.destroy).pack(padx=20, pady=(10, 15), fill=tk.X)
        dlg.wait_window()
        return result[0]

    def launch(self, e=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        proj = self.projects[sel[0]]

        channel_configs = detect_channel_configs(proj["path"])

        if proj.get("channels") is None:
            # Legacy single-mask mode (from PREDICTED_MASK_GLOB_PATTERN fallback)
            val_channel = None
            mask_path = proj["mask"]
            out_path = proj["out"]
        else:
            predicted_chs = proj["channels"]
            if len(predicted_chs) == 1:
                val_channel = predicted_chs[0]
            elif CONFIGURED_VAL_CHANNEL is not None and CONFIGURED_VAL_CHANNEL in predicted_chs:
                # Launcher already chose a channel — skip the redundant dialog
                val_channel = CONFIGURED_VAL_CHANNEL
            else:
                val_channel = self._select_channel_dialog(proj["path"], predicted_chs)
                if val_channel is None:
                    return  # user cancelled

            # Find the correct mask for this channel
            mask_candidates = sorted(glob(os.path.join(proj["path"], f"*_predict_original_ch{val_channel}*.tif")))
            mask_candidates = [
                m for m in mask_candidates
                if "coverage" not in os.path.basename(m).lower()
                and "combined" not in os.path.basename(m).lower()
                and "debug" not in os.path.basename(m).lower()
                and "unfiltered" not in os.path.basename(m).lower()
            ]
            if not mask_candidates:
                messagebox.showerror("Error", f"No predicted mask found for channel {val_channel}.")
                return
            mask_path = mask_candidates[-1]

            validation_dir = os.path.join(proj["path"], VALIDATION_FOLDER_NAME)
            os.makedirs(validation_dir, exist_ok=True)
            out_path = os.path.join(validation_dir, get_output_filename(val_channel))

        ch_paths = {}
        for i, cfg in enumerate(channel_configs):
            fs = glob(os.path.join(proj["path"], cfg['glob_pattern']))
            if fs:
                ch_paths[i] = fs[0]
            else:
                print(f"  [WARNING] Missing channel file for {cfg['name']} using pattern {cfg['glob_pattern']}")

        self.root.withdraw()
        top = tk.Toplevel(self.root)
        top.lift()
        top.attributes("-topmost", True)
        top.focus_force()
        top.after(600, lambda: top.attributes("-topmost", False))

        def on_app_close():
            top.destroy()
            self.root.deiconify()
            self.find()

        top.protocol("WM_DELETE_WINDOW", on_app_close)

        RoiVerifierApp(
            top, proj["path"], mask_path, ch_paths, out_path,
            channel_configs=channel_configs,
            on_close_callback=on_app_close,
            val_channel=val_channel,
        )

def _apply_config(config_path: str) -> None:
    """Pre-populate main folder, mask pattern, channel names and colors from CellMatchr config."""
    import json
    global PREDICTED_MASK_GLOB_PATTERN, CHANNEL_NAME_OVERRIDES, CHANNEL_COLOR_OVERRIDES, CONFIGURED_VAL_CHANNEL
    try:
        with open(config_path) as f:
            cfg = json.load(f)

        pattern = cfg.get("val_mask_pattern", "").strip()
        if pattern:
            PREDICTED_MASK_GLOB_PATTERN = pattern
            # Extract the pre-selected channel number so the project chooser can skip
            # the redundant "Select Channel to Validate" dialog.
            m = re.search(r"_predict_original_ch(\d+)", pattern)
            if m:
                CONFIGURED_VAL_CHANNEL = int(m.group(1))

        names_str = cfg.get("val_channel_names", "").strip()
        if names_str:
            CHANNEL_NAME_OVERRIDES = {
                i: name.strip()
                for i, name in enumerate(names_str.split(","))
                if name.strip()
            }

        colors_str = cfg.get("val_channel_colors", "").strip()
        if colors_str:
            CHANNEL_COLOR_OVERRIDES = {
                i: color.strip()
                for i, color in enumerate(colors_str.split(","))
                if color.strip()
            }

        main_folder = (
            cfg.get("val_main_folder", "").strip()
            or cfg.get("output_root_folder", "").strip()
        )
        if main_folder and os.path.isdir(main_folder):
            with open(CONFIG_FILE, "w") as f:
                f.write(main_folder)
    except Exception as e:
        print(f"[config] Could not apply {config_path}: {e}")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="CellMatchr Validator")
    _parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.cellmatchr/config.json"),
        help="Path to CellMatchr config JSON",
    )
    _args, _ = _parser.parse_known_args()
    if os.path.exists(_args.config):
        _apply_config(_args.config)

    root = tk.Tk()
    app = ProjectChooser(root)
    root.mainloop()