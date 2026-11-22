"""Step 3 dialog for previewing models on the same small patch."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from PIL import Image, ImageTk

from .model_download import (DOWNLOADS_TITLE, built_in_model_id, model_choice_label,
                             model_is_downloaded, model_stage)
from .model_wizard_compute import crop_box, display_image, patch_pixels, write_json


def open_model_downloads(parent, models, track_process):
    """Download built-in models one at a time and keep their output available."""
    existing = getattr(parent, "_cellmatchr_download_window", None)
    if existing is not None and existing.winfo_exists():
        existing.deiconify()
        existing.lift()
        existing.grab_set()
        return existing
    built_in = {name: settings for name, settings in models.items()
                if not any(value for key, value in settings.items() if key.endswith("_model_path"))}
    available = {}
    previous_grab = parent.grab_current()
    window = tk.Toplevel(parent)
    parent._cellmatchr_download_window = window
    window._download_active = False
    window.transient(parent)
    window.title(DOWNLOADS_TITLE)
    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Approximate weight sizes. First use may also install the runtime.").pack(anchor="w")
    choice = ttk.Combobox(frame, state="readonly", width=55)
    choice.pack(fill="x", pady=8)
    ttk.Label(frame, text="Downloaded models:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    inventory = ttk.Frame(frame)
    inventory.pack(fill="x", pady=(4, 8))
    inventory.columnconfigure(0, weight=1)
    downloaded_list = ttk.Treeview(inventory, show="tree", selectmode="none", height=1)
    downloaded_list.grid(row=0, column=0, sticky="ew")
    inventory_scroll = ttk.Scrollbar(inventory, orient="vertical", command=downloaded_list.yview)
    inventory_scroll.grid(row=0, column=1, sticky="ns")
    downloaded_list.configure(yscrollcommand=inventory_scroll.set)
    status = ttk.Label(frame, wraplength=480)
    status.pack(anchor="w")
    output = scrolledtext.ScrolledText(frame, height=8, width=67, wrap="word", state="disabled")
    output.pack(fill="both", expand=True, pady=8)
    temp = TemporaryDirectory(prefix="cellmatchr-model-download-")
    log = Path(temp.name) / "download.log"
    process = None
    last_size = -1

    def poll():
        nonlocal process, last_size
        if not window.winfo_exists():
            return
        try:
            size = log.stat().st_size
            if size != last_size and window.state() != "withdrawn":
                with log.open("rb") as source:
                    source.seek(max(0, size - 24000))
                    lines = source.read().decode("utf-8", errors="replace")
                output.configure(state="normal")
                output.delete("1.0", "end")
                output.insert("1.0", lines)
                output.see("end")
                output.configure(state="disabled")
                last_size = size
        except OSError:
            pass
        detail = model_stage(log)
        if process.poll() is None:
            if detail:
                status.configure(text=detail)
            window.after(400, poll)
            return
        status.configure(text=("Model ready." if process.returncode == 0
                               else "Could not prepare this model. The log below shows the reason."))
        process = None
        window._download_active = False
        if window.state() == "withdrawn":
            close()
            return
        refresh_models()

    def launch(args):
        nonlocal process, last_size
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "cellmatchr.model_download", *args],
                    cwd=Path(__file__).resolve().parent.parent, stdout=stream, stderr=subprocess.STDOUT)
        except OSError as exc:
            status.configure(text=str(exc))
            return
        track_process(process, DOWNLOADS_TITLE)
        window._download_active = True
        last_size = -1
        status.configure(text=f"Preparing {choice.get()}…")
        update_actions()
        poll()

    def start():
        settings = available.get(choice.get())
        if settings:
            launch([settings["seg_backend"], built_in_model_id(settings)])

    def import_weights():
        model = available[choice.get()]["seg_model_type"]
        paths = filedialog.askopenfilenames(
            parent=window, title="Select the original model weights and decoder")
        if paths:
            launch(["--import-microsam", model, *paths])

    def update_actions(_event=None):
        settings = available.get(choice.get())
        enabled = process is None and settings is not None
        choice.configure(state="readonly" if process is None and available else "disabled")
        download.configure(state="normal" if enabled else "disabled")
        import_button.configure(state="normal" if enabled and settings["seg_backend"] == "microsam"
                                else "disabled")

    def refresh_models():
        downloaded = []
        available.clear()
        for name, settings in built_in.items():
            if model_is_downloaded(settings):
                downloaded.append(name)
            else:
                available[model_choice_label(name, settings)] = settings
        for item in downloaded_list.get_children():
            downloaded_list.delete(item)
        for name in downloaded or ("None yet.",):
            downloaded_list.insert("", "end", text=name)
        downloaded_list.configure(height=max(1, min(len(downloaded), 6)))
        if len(downloaded) > 6:
            inventory_scroll.grid()
        else:
            inventory_scroll.grid_remove()
        selection = choice.get()
        choice.configure(values=tuple(available))
        choice.set(selection if selection in available else
                   next(iter(available), "All pretrained models are downloaded"))
        update_actions()

    def close():
        if window.grab_current() is window:
            window.grab_release()
            if previous_grab is not None and previous_grab.winfo_exists():
                previous_grab.grab_set()
        if process is not None:
            window.withdraw()
            return
        window.destroy()
        temp.cleanup()

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    download = ttk.Button(buttons, text="Download selected model", command=start)
    download.pack(side="right")
    ttk.Button(buttons, text="Close", command=close).pack(side="right", padx=5)
    import_button = ttk.Button(buttons, text="Import Micro-SAM weights…", command=import_weights)
    import_button.pack(side="left")
    choice.bind("<<ComboboxSelected>>", update_actions)
    refresh_models()
    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Escape>", lambda _event: close())
    window.grab_set()
    return window


class ModelWizard:
    def __init__(self, root, config, models, defaults, apply_model, open_filters, track_process,
                 saved=None, can_reuse=False):
        self.config = config
        self.models = models
        self.selected = list(defaults)
        self.apply_model = apply_model
        self.open_filters = open_filters
        self.track_process = track_process
        self.preview = None
        self.center = None
        self.box = None
        self.process = None
        self.job = None
        self.chosen_model = None
        self.applied_model = None
        self.model_stage = ("", "")
        self.can_use_filters = False
        self.stop_requested = False
        self.closing = False
        self.applying = False
        self.images = {}
        self._patch_redraw_job = None
        self._rendered_canvas_sizes = None
        self._overview_preview = (None, None)
        self._patch_preview = (None, None)
        self.temp = TemporaryDirectory(prefix="cellmatchr-model-wizard-")
        self.scratch = Path(self.temp.name)

        self.window = tk.Toplevel(root)
        self.window.title("Model Selection Wizard")
        self.window.transient(root)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.geometry(f"{min(1140, root.winfo_screenwidth() - 65)}x"
                             f"{min(810, root.winfo_screenheight() - 100)}")
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.bind("<Escape>", lambda _event: self.close())

        self.heading = ttk.Label(self.window, text="Choose a patch", font=("TkDefaultFont", 16, "bold"))
        self.heading.grid(row=0, column=0, sticky="w", padx=18, pady=(14, 7))
        self.body = ttk.Frame(self.window, padding=(18, 0))
        self.body.grid(row=1, column=0, sticky="nsew")
        self.footer = ttk.Frame(self.window, padding=(18, 8))
        self.footer.grid(row=2, column=0, sticky="ew")
        self.footer.columnconfigure(0, weight=1)
        self.status = ttk.Label(self.footer, text="Opening the first image…", wraplength=670)
        self.status.grid(row=0, column=0, columnspan=3, sticky="w")
        self.progress = ttk.Progressbar(self.footer, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 9))
        self.primary = ttk.Button(self.footer, text="Compare models", command=self._compare, state="disabled")
        self.primary.grid(row=2, column=2, sticky="e", padx=(8, 0))
        self.close_button = ttk.Button(self.footer, text="Close", command=self.close)
        self.close_button.grid(row=2, column=1, sticky="e")
        self.download_button = ttk.Button(self.footer, text=f"{DOWNLOADS_TITLE}…",
                                          command=self._open_downloads)
        self.download_button.grid(row=2, column=0, sticky="w")
        self.window.grab_set()

        if saved:
            self.preview = saved["preview"]
            self.job = saved["folder"]
            self.box = saved["box"]
            self.result_models = saved["models"]
            self.chosen_model = saved["chosen"]
            self.applied_model = saved["chosen"] if can_reuse else None
            self.can_use_filters = can_reuse
            self.channel = tk.StringVar(self.window, value=str(saved["channel"]))
            self.size = tk.StringVar(self.window, value="1024")
            self.progress.grid_remove()
            self.window.update_idletasks()
            self._show_results(saved["results"])
        else:
            self._prepare()

    def _open_downloads(self):
        if self.process is None:
            open_model_downloads(self.window, self.models, self.track_process)

    def _prepare(self):
        if self.applying:
            return
        self._clear()
        self.box = self.center = None
        self.heading.configure(text="Choose a patch")
        self.status.configure(text="Opening the first image…")
        self.primary.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.grid()
        self.progress.start(50)
        settings = self.scratch / "settings.json"
        try:
            write_json(settings, self.config)
            self._start("prepare", settings)
        except (OSError, ValueError) as exc:
            self._show_error(f"Could not open the first image. {exc}")

    def _start(self, action, settings):
        self.job = settings.parent
        log = self.job / "wizard.log"
        with log.open("wb") as stream:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "cellmatchr.model_wizard_compute", action, str(settings)],
                cwd=Path(__file__).resolve().parents[1], stdout=stream, stderr=subprocess.STDOUT)
        self.track_process(self.process, "Model Selection Wizard")
        self.download_button.configure(state="disabled")
        self.window.after(120, self._poll, action, log)

    def _poll(self, action, log):
        progress = self.job / "progress.json"
        data = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else None
        if action == "compare" and data:
            count = len(data["results"])
            self.progress.configure(value=count)
            current = data["current"]
            if current:
                info = f"Comparing {count + 1} of {data['total']} · {current}"
                detail = model_stage(data["current_log"]) if data.get("current_log") else ""
                if detail:
                    self.model_stage = (current, detail)
                elif current == self.model_stage[0]:
                    detail = self.model_stage[1]
                if detail:
                    info += f"\n{detail}"
                if self.stop_requested:
                    info += " · Stopping after this model"
                self.status.configure(text=info)
        exit_code = self.process.poll()
        if exit_code is None:
            self.window.after(350, self._poll, action, log)
            return
        self.process = None
        self.download_button.configure(state="normal")
        if self.closing:
            self.close()
            return
        self.close_button.configure(text="Close", state="normal")
        self.progress.stop()
        self.progress.grid_remove()
        if exit_code:
            if action == "compare":
                self.primary.configure(text="Compare models", command=self._compare, state="normal")
                self._set_picker_enabled(True)
            self._show_error("The wizard could not finish this preview.", log)
        elif action == "prepare":
            self.preview = json.loads((self.scratch / "preview.json").read_text(encoding="utf-8"))
            self._show_picker()
        else:
            # Read the final result after the child exits, even if it wrote progress during this poll.
            data = json.loads(progress.read_text(encoding="utf-8"))
            self._show_results(data["results"])

    def _show_error(self, text, log=None):
        self.progress.stop()
        self.progress.grid_remove()
        self.status.configure(text=text)
        if log:
            self._show_log(log)

    def _show_log(self, path):
        dialog = tk.Toplevel(self.window)
        dialog.title("Model preview log")
        dialog.transient(self.window)
        output = scrolledtext.ScrolledText(dialog, width=96, height=23, wrap="word")
        output.pack(fill="both", expand=True, padx=12, pady=12)
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - 24_000))
            output.insert("1.0", stream.read().decode("utf-8", errors="replace"))
        output.configure(state="disabled")
        output.see("end")
        dialog.focus_set()
        dialog.grab_set()

        def dismiss():
            dialog.destroy()
            self.window.grab_set()

        ttk.Button(dialog, text="Close", command=dismiss).pack(pady=(0, 10))
        dialog.protocol("WM_DELETE_WINDOW", dismiss)
        dialog.bind("<Escape>", lambda _event: dismiss())

    def _clear(self):
        self._cancel_patch_redraw()
        self._rendered_canvas_sizes = None
        for child in self.body.winfo_children():
            child.destroy()
        self.images.clear()

    def _show_picker(self):
        if self.applying:
            return
        self._clear()
        self.close_button.configure(text="Close")
        self.heading.configure(text="Choose a patch")
        ttk.Label(self.body, text=self.preview["sample"], font=("TkDefaultFont", 11, "bold"),
                  wraplength=890).pack(anchor="w")
        ttk.Label(self.body, text="Click where you can see several cells. Include some dim cells or uneven background.").pack(
            anchor="w", pady=(2, 6))

        channels = self.preview["channels"]
        if not hasattr(self, "channel"):
            requested = [part.strip() for part in self.config.get("seg_channels", "").split(",")]
            first = next((part for part in requested if part in channels), next(iter(channels)))
            self.channel = tk.StringVar(self.window, value=first)
            self.size = tk.StringVar(self.window, value="1024")
        elif self.channel.get() not in channels:
            self.channel.set(next(iter(channels)))
        controls = ttk.Frame(self.body)
        controls.pack(fill="x", pady=(0, 7))
        ttk.Label(controls, text="Channel").pack(side="left")
        self.channel_select = ttk.Combobox(controls, textvariable=self.channel,
                                           values=list(channels), width=5, state="readonly")
        self.channel_select.pack(side="left", padx=(6, 18))
        self.channel_select.bind("<<ComboboxSelected>>", lambda _event: self._redraw_patch())
        ttk.Label(controls, text="Patch size (pixels)").pack(side="left")
        self.size_select = ttk.Combobox(controls, textvariable=self.size,
                                        values=("512", "768", "1024"), width=6, state="readonly")
        self.size_select.pack(side="left", padx=6)
        self.size_select.bind("<<ComboboxSelected>>", lambda _event: self._redraw_patch())
        self.patch_size = ttk.Label(controls, text="")
        self.patch_size.pack(side="left", padx=5)

        images = ttk.Frame(self.body)
        images.pack(fill="both", expand=True)
        images.columnconfigure((0, 1), weight=1, uniform="picker")
        images.rowconfigure(1, weight=1)
        ttk.Label(images, text="First image").grid(row=0, column=0, sticky="w")
        ttk.Label(images, text="Selected patch · click to refine").grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.overview = tk.Canvas(images, background="#151722", highlightthickness=0, width=240, height=240)
        self.overview.grid(row=1, column=0, sticky="nsew", pady=5)
        self.overview.bind("<Button-1>", self._choose_center)
        self.overview.bind("<Configure>", self._schedule_patch_redraw)
        self.detail = tk.Canvas(images, background="#151722", highlightthickness=0, width=240, height=240)
        self.detail.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=5)
        self.detail.bind("<Button-1>", self._refine_center)
        self.detail.bind("<Configure>", self._schedule_patch_redraw)
        selection = ttk.Frame(self.body)
        selection.pack(fill="x", pady=(7, 1))
        selection.columnconfigure(0, weight=1)
        selection.columnconfigure(2, weight=1)
        ttk.Label(selection, text="Models to compare", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(selection, text="Add another model").grid(row=0, column=2, sticky="w", padx=(14, 0))
        self.model_list = tk.Listbox(selection, height=4, width=32, exportselection=False)
        self.model_list.grid(row=1, column=0, rowspan=2, sticky="nsew")
        bar = ttk.Scrollbar(selection, orient="vertical", command=self.model_list.yview)
        bar.grid(row=1, column=1, rowspan=2, sticky="ns")
        self.model_list.configure(yscrollcommand=bar.set)
        self.add_choice = ttk.Combobox(selection, state="readonly", width=28)
        self.add_choice.grid(row=1, column=2, sticky="ew", padx=(14, 6), pady=(0, 3))
        self.add_button = ttk.Button(selection, text="Add", command=self._add_model)
        self.add_button.grid(row=1, column=3, sticky="w")
        self.remove_button = ttk.Button(selection, text="Remove selected", command=self._remove_model)
        self.remove_button.grid(row=2, column=2, sticky="w", padx=(14, 0))

        self.primary.configure(text="Compare models", command=self._compare)
        self.status.configure(text="First use may download model weights. The preview shows proposals before object filters.")
        self._refresh_models()
        self._redraw_patch()

    def _source(self):
        return self.preview["channels"][self.channel.get()]

    def _canvas_sizes(self):
        return tuple((canvas.winfo_width(), canvas.winfo_height())
                     for canvas in (self.overview, self.detail))

    def _cancel_patch_redraw(self):
        if self._patch_redraw_job is not None:
            self.window.after_cancel(self._patch_redraw_job)
            self._patch_redraw_job = None

    def _schedule_patch_redraw(self, _event=None):
        self._cancel_patch_redraw()
        if self._canvas_sizes() != self._rendered_canvas_sizes:
            self._patch_redraw_job = self.window.after(75, self._run_patch_redraw)

    def _run_patch_redraw(self):
        self._patch_redraw_job = None
        self._redraw_patch()

    def _draw_image(self, canvas, image):
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if min(width, height) < 2:
            return None
        scale = min(width / image.width, height / image.height)
        image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                             Image.Resampling.LANCZOS)
        self.images[canvas] = ImageTk.PhotoImage(image, master=canvas)
        x, y = (width - image.width) // 2, (height - image.height) // 2
        canvas.create_image(x, y, image=self.images[canvas], anchor="nw")
        return x, y, image.width, image.height

    def _redraw_patch(self):
        self._cancel_patch_redraw()
        self._rendered_canvas_sizes = self._canvas_sizes()
        source = self._source()
        width, height = source["width"], source["height"]
        overview_path = source["overview"]
        if overview_path != self._overview_preview[0]:
            with Image.open(overview_path) as overview:
                self._overview_preview = (overview_path, overview.copy())
        self.overview_rect = self._draw_image(self.overview, self._overview_preview[1])
        if self.center is None:
            self.detail.delete("all")
            self.detail.create_text(self.detail.winfo_width() / 2, self.detail.winfo_height() / 2,
                                    text="Click the first image to preview a patch", fill="#cdd6f4")
            return
        self.box = crop_box(*self.center, int(self.size.get()), width, height)
        left, top, right, bottom = self.box
        self.patch_size.configure(text=f"Selected {right-left} × {bottom-top}")
        if self.overview_rect:
            x, y, w, h = self.overview_rect
            self.overview.create_rectangle(x + left*w/width, y + top*h/height,
                                           x + right*w/width, y + bottom*h/height,
                                           outline="#ffd15c", width=2)
        patch_key = (overview_path, self.box)
        if patch_key != self._patch_preview[0]:
            patch = patch_pixels(source, self.box, self.preview["brightfield"])
            self._patch_preview = (patch_key, display_image(patch))
        self.detail_rect = self._draw_image(self.detail, self._patch_preview[1])
        self._update_compare_button()

    @staticmethod
    def _relative_click(event, rect):
        if rect is None:
            return None
        left, top, width, height = rect
        if left <= event.x < left + width and top <= event.y < top + height:
            return (event.x-left) / width, (event.y-top) / height
        return None

    def _choose_center(self, event):
        point = self._relative_click(event, getattr(self, "overview_rect", None))
        if point is not None and self.process is None:
            source = self._source()
            self.center = (point[0] * source["width"], point[1] * source["height"])
            self._redraw_patch()

    def _refine_center(self, event):
        point = self._relative_click(event, getattr(self, "detail_rect", None))
        if point is not None and self.box is not None and self.process is None:
            left, top, right, bottom = self.box
            self.center = (left + point[0]*(right-left), top + point[1]*(bottom-top))
            self._redraw_patch()

    def _set_picker_enabled(self, enabled):
        for control in (self.channel_select, self.size_select, self.add_choice):
            control.configure(state="readonly" if enabled else "disabled")
        for control in (self.model_list, self.add_button, self.remove_button):
            control.configure(state="normal" if enabled else "disabled")
        if enabled:
            self._refresh_models()

    def _refresh_models(self):
        self.model_list.delete(0, "end")
        for model in self.selected:
            self.model_list.insert("end", model)
        self._additional_models = {
            model_choice_label(name, settings): name for name, settings in self.models.items()
            if name not in self.selected
        }
        additional = list(self._additional_models)
        self.add_choice.configure(values=additional)
        self.add_choice.set(additional[0] if additional else "")
        self.add_button.configure(state="normal" if additional else "disabled")
        self._update_compare_button()

    def _update_compare_button(self):
        self.primary.configure(state="normal" if self.box and len(self.selected) >= 2 and
                               self.process is None else "disabled")

    def _add_model(self):
        name = self._additional_models.get(self.add_choice.get())
        if name in self.models and name not in self.selected:
            self.selected.append(name)
            self._refresh_models()
            self.model_list.see("end")

    def _remove_model(self):
        selection = self.model_list.curselection()
        if selection:
            self.selected.pop(selection[0])
            self._refresh_models()

    def _compare(self):
        if self.process or self.box is None or len(self.selected) < 2:
            return
        downloads = getattr(self.window, "_cellmatchr_download_window", None)
        if getattr(downloads, "_download_active", False):
            self.status.configure(text="Wait for the model download to finish before comparing.")
            return
        try:
            job = Path(mkdtemp(prefix="comparison-", dir=self.scratch))
            settings = job / "settings.json"
            write_json(settings, {"config": self.config, "source": self._source(), "box": self.box,
                                  "brightfield": self.preview["brightfield"],
                                  "models": [(name, self.models[name]) for name in self.selected]})
            self._start("compare", settings)
        except (OSError, ValueError) as exc:
            self._show_error(f"Could not start the comparison. {exc}")
            return
        self.stop_requested = False
        self.chosen_model = None
        self.applied_model = None
        self.can_use_filters = False
        self.result_models = {name: self.models[name] for name in self.selected}
        self._set_picker_enabled(False)
        self.primary.configure(state="disabled")
        self.close_button.configure(text="Stop after current model")
        self.status.configure(text=f"Preparing {len(self.selected)} model previews…")
        self.progress.configure(mode="determinate", maximum=len(self.selected), value=0)
        self.progress.grid()

    def _show_results(self, results, zoom=None):
        if self.applying:
            return
        self.comparison_results = results
        self._clear()
        self.heading.configure(text=f"Compare predictions · Channel {self.channel.get()}")
        ttk.Label(self.body, text="Look for missed cells first. Click a preview, then choose Use selected model.").pack(
            anchor="w", pady=(1, 8))
        # The same zoom window is applied to every model and the original image.
        controls = ttk.Frame(self.body)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Full patch", command=lambda: self._show_results(results)).pack(side="left")
        patch_side = self.box[2] - self.box[0]
        center_crop = crop_box(patch_side // 2, patch_side // 2, patch_side // 2, patch_side, patch_side)
        ttk.Button(controls, text="2× center", command=lambda: self._show_results(results, center_crop)).pack(
            side="left", padx=6)
        ttk.Label(controls, text="Click the original image to zoom on a specific area.").pack(side="left", padx=6)
        canvas = tk.Canvas(self.body, highlightthickness=0, background=self.window.cget("background"))
        canvas.pack(fill="both", expand=True)
        scroll_x = ttk.Scrollbar(self.body, orient="horizontal", command=canvas.xview)
        scroll_x.pack(fill="x")
        canvas.configure(xscrollcommand=scroll_x.set)
        strip = ttk.Frame(canvas)
        canvas.create_window(0, 0, window=strip, anchor="nw")
        strip.bind("<Configure>", lambda event: canvas.configure(scrollregion=(0, 0, event.width, event.height)))
        display = [("Original patch", self.job / "original.png", "Source", None)]
        for result in results:
            overlay = result.get("overlay")
            note = f"{result['count']} proposals" if overlay else result["error"]
            display.append((result["name"], overlay, note, result))
        side = max(130, min(255, (self.window.winfo_width() - 46) // 5 - 18))
        selected = {}
        for i, (name, overlay, note, result) in enumerate(display):
            card = ttk.Frame(strip, padding=(0, 0, 10, 0))
            card.grid(row=0, column=i, sticky="n", pady=4)
            tk.Label(card, text=name, height=2, anchor="sw", justify="left", wraplength=side,
                     font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            if overlay:
                with Image.open(overlay) as image:
                    region = image.crop(zoom) if zoom else image
                    thumb = region.resize((side, side), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(thumb, master=card)
                self.images[card] = photo
                tile = tk.Canvas(card, width=side+8, height=side+8, highlightthickness=0,
                                 background=self.window.cget("background"), cursor="hand2")
                tile.create_image(4, 4, image=photo, anchor="nw")
                tile.pack(anchor="w", pady=(4, 2))
                if result is None:
                    def magnify(event):
                        left, top, right, bottom = zoom or (0, 0, patch_side, patch_side)
                        x = left + (event.x - 4) * (right - left) / side
                        y = top + (event.y - 4) * (bottom - top) / side
                        self._show_results(results, crop_box(x, y, patch_side // 2, patch_side, patch_side))
                    tile.bind("<Button-1>", magnify)
                else:
                    selected[name] = tile
                    tile.bind("<Button-1>", lambda _event, item=result: self._use_model(item, selected))
                    tile.bind("<Return>", lambda _event, item=result: self._use_model(item, selected))
                    tile.bind("<space>", lambda _event, item=result: self._use_model(item, selected))
                    tile.configure(takefocus=True)
            else:
                placeholder = tk.Canvas(card, width=side+8, height=side+8, background="#252736", highlightthickness=0)
                placeholder.create_text(side / 2, side / 2, text="No preview generated", fill="#cdd6f4")
                placeholder.pack(anchor="w", pady=(4, 2))
            ttk.Label(card, text=note, wraplength=side).pack(anchor="w", pady=(3, 7))
            if result and not overlay:
                ttk.Button(card, text="View log", command=lambda path=result["log"]: self._show_log(path)).pack(anchor="w")

        successful = sum("overlay" in result for result in results)
        self.status.configure(text=f"{successful} of {len(self.result_models)} model previews ready."
                              + (" Scroll sideways to see more models." if len(display) > 5 else ""))
        self.filter_button = ttk.Button(controls, text="Open Parameter Wizard", command=self._open_filters,
                                        state="disabled")
        self.filter_button.pack(side="right")
        picker = self._show_picker if "channels" in self.preview else self._prepare
        ttk.Button(controls, text="Change patch or models", command=picker).pack(side="right", padx=6)
        self.primary.configure(text="Use selected model",
                               command=lambda: self._apply_selection(selected))
        self._mark_selection(selected)

    def _mark_selection(self, tiles):
        for name, tile in tiles.items():
            tile.delete("selection")
            if self.chosen_model == name:
                side = int(tile.cget("width"))
                tile.create_rectangle(2, 2, side-2, side-2, width=4, outline="#39b967", tags="selection")
        if self.chosen_model:
            if self.chosen_model != self.applied_model:
                message = "Click Use selected model to apply it to Step 3."
            elif self.can_use_filters:
                message = "Open the Parameter Wizard to reuse this patch."
            elif self.channel.get() not in {part.strip() for part in self.config["seg_channels"].split(",")}:
                message = f"Add channel {self.channel.get()} in Step 3 to reuse this patch."
            else:
                message = "The comparison uses different input or model settings."
            self.status.configure(text=f"Selected {self.chosen_model}. {message}")
        pending = bool(self.chosen_model and self.chosen_model != self.applied_model)
        self.primary.configure(state="normal" if pending else "disabled")
        self.filter_button.configure(state="normal" if self.can_use_filters and not pending else "disabled")
        self.close_button.configure(text="Done" if self.applied_model and not pending else "Close")

    def _use_model(self, result, tiles):
        if self.applying:
            return
        self.chosen_model = result["name"]
        self._mark_selection(tiles)

    def _apply_selection(self, tiles):
        name = self.chosen_model
        if self.applying or not name or name == self.applied_model:
            return
        result = next(item for item in self.comparison_results if item["name"] == name)
        comparison = {
            "original": str(self.job / "original.png"),
            "results": self.comparison_results, "models": self.result_models,
            "box": list(self.box), "channel": int(self.channel.get()),
            "preview": self.preview,
        }
        self.applying = True
        self.primary.configure(state="disabled")
        self.filter_button.configure(state="disabled")
        self.close_button.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.status.configure(text=f"Applying {name} and saving its preview…")

        def finished(error=None):
            self.applying = False
            if error is None:
                self.applied_model = name
                self.close()
                return
            self.close_button.configure(state="normal")
            self.download_button.configure(state="normal")
            self._mark_selection(tiles)
            self.status.configure(text=f"Could not use the selected model. {error}")

        try:
            self.apply_model(self.result_models[name], result, comparison, finished)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            finished(error)

    def _open_filters(self):
        self.close()
        self.open_filters()

    def close(self):
        if self.applying:
            return
        if self.process is not None:
            if self.job != self.scratch:
                (self.job / "stop").touch()
                self.stop_requested = True
                self.close_button.configure(state="disabled")
                self.status.configure(text="Stopping after the current model…")
                return
            self.closing = True
            if self.process.poll() is None:
                self.process.terminate()
            self.close_button.configure(state="disabled")
            return
        self._cancel_patch_redraw()
        self.window.destroy()
        self.images.clear()
        self._overview_preview = (None, None)
        self._patch_preview = (None, None)
        self.temp.cleanup()
