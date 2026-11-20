"""Check that model selection and filter calibration use the same original pixels."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr import model_wizard_compute as wizard
from cellmatchr import model_wizard_archive as archives
from cellmatchr import parameter_wizard_compute as parameters
from cellmatchr.config import DEFAULT_CONFIG
from cellmatchr.launcher import CellMatchrLauncher, _model_wizard_options


class ModelWizardTests(unittest.TestCase):
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
        self.assertIn("not installed", last["error"])

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
