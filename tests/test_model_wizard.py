"""Check that model selection and filter calibration use the same original pixels."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
import tifffile

from cellmatchr import model_wizard_compute as wizard
from cellmatchr import model_wizard_archive as archives
from cellmatchr import parameter_wizard_compute as parameters
from cellmatchr.config import DEFAULT_CONFIG
from cellmatchr.launcher import CellMatchrLauncher, _model_wizard_options
from cellmatchr.model_wizard import ModelWizard


class ModelWizardTests(unittest.TestCase):
    def test_parameter_wizard_no_pick_leaves_launcher_filter_controls_alone(self):
        launcher = CellMatchrLauncher.__new__(CellMatchrLauncher)
        launcher.config = dict(self.cfg, project_output_folder=str(self.root),
                               seg_min_object_size=70, seg_per_channel_filters_enabled=False)
        launcher.root = Mock()
        launcher.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        launcher._is_running = Mock(return_value=False)
        launcher._sync_vars_to_config = Mock()
        launcher._matching_model_preview = Mock(return_value={"channel": 1, "manifest": "cached.json"})
        launcher._track_proc = Mock()
        launcher._wizard_status = Mock()
        launcher._wizard_btn = Mock()
        launcher._wizard_progress = Mock()
        launcher._per_ch_filt_enabled = Mock()
        launcher._toggle_per_ch_filters = Mock()
        process = Mock(stdout=[], returncode=0)
        with (patch("cellmatchr.launcher.save_config"), patch("cellmatchr.launcher.save_project_config"),
              patch("cellmatchr.launcher.load_config", return_value=dict(launcher.config)),
              patch("cellmatchr.launcher.subprocess.Popen", return_value=process),
              patch("cellmatchr.launcher.threading.Thread",
                    side_effect=lambda target, daemon: SimpleNamespace(start=target))):
            launcher._launch_parameter_wizard()
        launcher._per_ch_filt_enabled.set.assert_not_called()
        launcher._toggle_per_ch_filters.assert_not_called()
        self.assertEqual(launcher.config["seg_min_object_size"], 70)
        self.assertEqual(launcher._wizard_status.config.call_args.kwargs["text"], "Filter settings unchanged.")

    def test_using_selected_model_closes_after_save_and_can_retry_failure(self):
        dialog = ModelWizard.__new__(ModelWizard)
        result = {"name": "Micro-SAM"}
        settings = {"seg_backend": "microsam", "seg_model_type": "vit_b_lm"}
        dialog.applying = False
        dialog.chosen_model = result["name"]
        dialog.applied_model = None
        dialog.comparison_results = [result]
        dialog.result_models = {result["name"]: settings}
        dialog.job = Path("comparison")
        dialog.box = (0, 0, 32, 32)
        dialog.channel = SimpleNamespace(get=lambda: "0")
        dialog.preview = {}
        for control in ("primary", "filter_button", "close_button", "download_button", "status"):
            setattr(dialog, control, Mock())
        dialog.apply_model = Mock()
        dialog._mark_selection = Mock()
        dialog.close = Mock()

        dialog._apply_selection({})
        self.assertTrue(dialog.applying)
        dialog.close.assert_not_called()
        dialog.apply_model.call_args.args[-1]("Preview could not be written")
        self.assertFalse(dialog.applying)
        dialog.close.assert_not_called()
        self.assertIn("Preview could not be written", dialog.status.configure.call_args.kwargs["text"])

        dialog._apply_selection({})
        dialog.apply_model.call_args.args[-1]()
        self.assertEqual(dialog.applied_model, result["name"])
        dialog.close.assert_called_once()

    def test_patch_move_skips_redraw_and_resize_coalesces_events(self):
        dialog = ModelWizard.__new__(ModelWizard)
        dialog.window = Mock()
        dialog.window.after.side_effect = ("first", "last")
        dialog._patch_redraw_job = None
        dialog._rendered_canvas_sizes = ((400, 400), (400, 400))
        dialog._canvas_sizes = Mock(return_value=dialog._rendered_canvas_sizes)
        dialog._redraw_patch = Mock()

        dialog._schedule_patch_redraw()
        dialog.window.after.assert_not_called()
        dialog._canvas_sizes.return_value = ((410, 410), (410, 410))
        dialog._schedule_patch_redraw()
        dialog._canvas_sizes.return_value = ((420, 420), (420, 420))
        dialog._schedule_patch_redraw()
        dialog.window.after_cancel.assert_called_once_with("first")
        dialog.window.after.call_args.args[-1]()
        dialog._redraw_patch.assert_called_once()

    def test_scroll_from_download_toplevel_leaves_main_canvas_unchanged(self):
        app = CellMatchrLauncher.__new__(CellMatchrLauncher)
        app.root = object()
        app._canvas = Mock()
        for window in (object(), app.root):
            widget = SimpleNamespace(winfo_toplevel=lambda: window)
            app._on_scroll(SimpleNamespace(widget=widget, num=5, delta=0))
        app._canvas.yview_scroll.assert_called_once_with(2, "units")

    def test_download_size_and_file_stages_are_labelled(self):
        from cellmatchr.model_download import model_choice_label, model_stage

        self.assertIn("626 MB", model_choice_label(
            "micro-SAM2 · hvit_b_cells",
            {"seg_backend": "microsam2", "seg_microsam2_model": "hvit_b_cells"}))
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "download.log"
            log.write_text("Downloading file 'hvit_b_cells'\n100%|####\n"
                           "Downloading file 'hvit_b_cells_decoder'\n 3%|#\n")
            self.assertIn("decoder (file 2 of 2) (3%)", model_stage(log))
            log.write_text("[MODEL] Downloading hvit_b_cells encoder (file 1 of 2) — 60% (0.8 MB/s)\n")
            self.assertEqual(model_stage(log), "Downloading hvit_b_cells encoder (file 1 of 2) — 60% (0.8 MB/s)")

    def test_model_status_and_verified_offline_import(self):
        import hashlib
        import tempfile
        from cellmatchr.model_download import import_microsam_weights, model_stage
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            decoder = root / "decoder.pt"
            weights.write_bytes(b"upstream weight")
            decoder.write_bytes(b"upstream decoder")
            names = ["vit_t_lm", "vit_t_lm_decoder"]
            registry = SimpleNamespace(abspath=root / "cache", registry={
                name: hashlib.sha256(file.read_bytes()).hexdigest()
                for name, file in zip(names, (weights, decoder))})
            with patch("cellmatchr.model_download.microsam_weights", return_value=(registry, names)):
                with self.assertRaisesRegex(ValueError, "published checksum"):
                    import_microsam_weights(names[0], [weights])
                self.assertFalse((registry.abspath / names[0]).exists())
                import_microsam_weights(names[0], [decoder, weights])
            self.assertEqual((registry.abspath / names[0]).read_bytes(), weights.read_bytes())
            log = root / "run.log"
            log.write_text("[MODEL] Downloading weights\n 42%|####\n")
            self.assertEqual(model_stage(log), "Downloading weights (42%)")

    def test_download_inventory_uses_each_cache_and_requires_all_weight_files(self):
        from cellmatchr.model_download import model_is_downloaded

        cases = [
            ("microsam", "seg_model_type", "vit_b_lm", "sam/models/vit_b_lm", "sam/models/vit_b_lm_decoder"),
            ("microsam2", "seg_microsam2_model", "hvit_b_cells",
             "sam2/model-cache/models/hvit_b_cells", "sam2/model-cache/models/hvit_b_cells_decoder"),
            ("microsam2", "seg_microsam2_model", "hvit_b", "sam2/model-cache/models/hvit_b", ""),
            ("cellpose", "seg_cellpose_variant", "cpsam", "cellpose/models/cpsam", ""),
            ("cellpose", "seg_cellpose_variant", "cyto2", "legacy/models/cyto2torch_0", ""),
        ]
        env = {"MICROSAM_CACHEDIR": str(self.root / "sam"),
               "CELLMATCHR_MICROSAM2_HOME": str(self.root / "sam2"),
               "CELLPOSE_LOCAL_MODELS_PATH": str(self.root / "cellpose" / "models"),
               "CELLMATCHR_LEGACY_CELLPOSE_HOME": str(self.root / "legacy"),
               "KERAS_HOME": str(self.root / "keras")}
        with patch.dict(os.environ, env):
            for backend, key, model, checkpoint, decoder in cases:
                with self.subTest(model=model):
                    settings = {"seg_backend": backend, key: model}
                    self.assertFalse(model_is_downloaded(settings))
                    path = self.root / checkpoint
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"checkpoint")
                    if decoder:
                        self.assertFalse(model_is_downloaded(settings))
                        (self.root / decoder).write_bytes(b"")
                        self.assertFalse(model_is_downloaded(settings))
                        (self.root / decoder).write_bytes(b"decoder")
                    self.assertTrue(model_is_downloaded(settings))

            stardist = {"seg_backend": "stardist"}
            archive = self.root / "keras/models/StarDist2D/2D_versatile_fluo/2D_versatile_fluo.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"incomplete archive")
            self.assertFalse(model_is_downloaded(stardist))
            with ZipFile(archive, "w") as complete:
                complete.writestr("weights_best.h5", b"weights")
            self.assertTrue(model_is_downloaded(stardist))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.crop = self.root / "crop"
        self.crop.mkdir()
        self.cfg = dict(DEFAULT_CONFIG, input_type="vsi", input_root_folder=str(self.inputs),
                        project_output_folder=str(self.root), seg_channels="1")
        self.source = self.inputs / "slide.tif"
        self.pixels = np.arange(2*51*60, dtype=np.uint16).reshape((2, 51, 60))
        tifffile.imwrite(self.source, self.pixels, photometric="minisblack",
                         metadata={"axes": "CYX"})

    def test_crop_maps_original_channel_and_clamps_at_edge(self):
        wizard.prepare(self.cfg, self.crop)
        data = json.loads((self.crop / "preview.json").read_text())
        source = data["channels"]["1"]
        box = wizard.crop_box(59, 2, 32, source["width"], source["height"])
        self.assertEqual(box, (28, 0, 60, 32))
        np.testing.assert_array_equal(wizard.patch_pixels(source, box, False),
                                      self.pixels[1, :32, 28:60])
        self.assertFalse(list(self.crop.glob("*.tif")))

    def test_brightfield_crop_keeps_original_planar_rgb(self):
        self.cfg["slide_signal_type"] = "brightfield"
        stained = np.stack([self.pixels[0] + delta for delta in (0, 500, 1000)])
        tifffile.imwrite(self.source, stained, photometric="minisblack",
                         metadata={"axes": "CYX"}, compression="zlib")
        wizard.prepare(self.cfg, self.crop)
        data = json.loads((self.crop / "preview.json").read_text())
        preview = wizard.patch_pixels(data["channels"]["0"], (12, 11, 38, 37), True)
        np.testing.assert_array_equal(preview, np.moveaxis(stained[:, 11:37, 12:38], 0, -1))

    def test_model_failure_preserves_success_and_cache_paths(self):
        wizard.prepare(self.cfg, self.crop)
        preview = json.loads((self.crop / "preview.json").read_text())
        run = self.root / "run"
        run.mkdir()
        choices, defaults = _model_wizard_options(self.cfg)
        request = {"config": self.cfg, "source": preview["channels"]["1"], "box": [8, 6, 40, 38],
                   "brightfield": False, "models": [(label, choices[label]) for label in defaults]}
        seen = []

        def fake_run(command, stdout, **_kwargs):
            config = json.loads(Path(command[-1]).read_text())
            seen.append(config["seg_backend"])
            if config["seg_backend"] == "stardist":
                stdout.write(b"No module named 'stardist'")
                return subprocess.CompletedProcess(command, 1)
            self.assertEqual(config["seg_min_object_size"], 0)
            self.assertFalse(config["seg_per_channel_filters_enabled"])
            raw = tifffile.imread(Path(config["input_root_folder"]) / "patch.tif")
            np.testing.assert_array_equal(raw, self.pixels[1, 6:38, 8:40])
            out = Path(config["output_root_folder"]) / "patch"
            out.mkdir(parents=True)
            labels = np.zeros_like(raw)
            labels[5:12, 6:13] = 1
            tifffile.imwrite(out / "patch_split_ch0.tif", raw)
            tifffile.imwrite(out / "patch_predict_original_ch0_model.tif", labels)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(wizard.subprocess, "run", side_effect=fake_run):
            wizard.compare(request, run)
        progress = json.loads((run / "progress.json").read_text())
        self.assertEqual(seen, ["microsam", "microsam2", "cellpose", "stardist"])
        self.assertEqual(progress["total"], 4)
        self.assertEqual(progress["current"], "")
        first, *_, last = progress["results"]
        self.assertEqual(first["count"], 1)
        self.assertIn("Open the log", last["error"])
        self.assertIn("No module named 'stardist'", Path(last["log"]).read_text())

        launcher = object.__new__(CellMatchrLauncher)
        launcher.config = self.cfg
        comparison = {"original": str(run / "original.png"), "results": progress["results"],
                      "models": choices, "box": request["box"], "channel": 1, "preview": preview}
        launcher._cache_selected_model_preview(choices[defaults[0]], first, 1, preview, comparison)
        self.addCleanup(launcher._model_preview_cache["directory"].cleanup)
        self.assertIsNone(launcher._matching_model_preview(self.cfg, {0}))
        self.assertIsNone(launcher._matching_model_preview(dict(self.cfg, seg_model_type="vit_l"), {1}))
        self.assertIsNotNone(launcher._matching_model_preview(dict(self.cfg, seg_min_object_size=500), {1}))

        # Other configured channels can still be calibrated on a fresh prediction.
        supplied = launcher._model_preview_cache["manifest"]
        target = self.root / "calibration"
        config = self.root / "multi.json"
        config.write_text(json.dumps(dict(self.cfg, seg_channels="1,0")))
        seg = parameters.seg
        with patch.multiple(seg, SKIP_SUBDIRS=False, PREPROCESS_MODE="local_bg_subtract",
                            SLIDE_SIGNAL_TYPE="fluorescence", SEG_BACKEND="microsam",
                            CELLPOSE_VARIANT="cpsam", CELLPOSE_DIAMETER=0, CELLPOSE_FLOW_THRESHOLD=0.4,
                            CELLPOSE_CELLPROB_THRESHOLD=0, TILE_SHAPE_2D=(2048, 2048), HALO_2D=(256, 256),
                            MIN_OBJECT_SIZE=0, MAX_OBJECT_SIZE=0, MAX_MAJOR_MINOR_RATIO=0,
                            RAW_MEAN_MIN=0, RAW_MEAN_MAX=0), \
                patch.object(parameters, "_setup_backend", return_value=(None,) * 5), \
                patch.object(seg, "_free_memory"), \
                patch.object(seg, "segment_by_tiling", return_value=np.zeros_like(self.pixels[0])) as predict, \
                patch("sys.argv", ["wizard", "--config", str(config), "--out-dir", str(target),
                                   "--reuse-manifest", supplied]):
            parameters.main()
        predict.assert_called_once()
        np.testing.assert_array_equal(predict.call_args.args[2], self.pixels[0])
        manifest = json.loads((target / "parameter_wizard_manifest.json").read_text())
        self.assertEqual([row["channel"] for row in manifest["channels"]], [1, 0])
        self.assertTrue(manifest["channels"][0]["model_preview"])

        logs = []
        output = self.root / "Segmentation"
        self.cfg["seg_test_run"] = True
        launcher._archive_model_preview(output, logs.append)
        self.assertFalse((self.root / archives.FOLDER).exists())
        self.cfg["seg_test_run"] = False
        launcher._archive_model_preview(output, logs.append)
        launcher._archive_model_preview(output, logs.append)
        saved, = archives.saved(self.root)
        self.assertEqual(saved["chosen"], defaults[0])
        self.assertEqual(len(saved["results"]), 4)
        self.assertFalse(output.exists())
        relocated = self.root / "moved-comparison"
        shutil.copytree(saved["folder"], relocated)
        copied = archives.load(relocated)
        self.assertEqual(Path(copied["original"]).parent, relocated.resolve())

        reopened = object.__new__(CellMatchrLauncher)
        reopened.config = self.cfg
        self.assertIsNone(reopened._matching_model_preview(self.cfg, {0}))
        self.assertIsNone(reopened._matching_model_preview(dict(self.cfg, seg_model_type="vit_l"), {1}))
        restored = reopened._matching_model_preview(self.cfg, {1})
        self.addCleanup(restored["directory"].cleanup)
        raw = json.loads(Path(restored["manifest"]).read_text())["channels"][0]
        np.testing.assert_array_equal(tifffile.imread(raw["labels"]), tifffile.imread(first["labels"]))

        self.source.unlink()
        self.assertIsNone(launcher._matching_model_preview(self.cfg, {1}))


if __name__ == "__main__":
    unittest.main()
