"""Compatibility checks for brightfield display, filtering, and training presets."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr import segmentation as seg
from cellmatchr.config import load_project_config
from cellmatchr.launcher import (_brightfield_preprocess_for_model,
                                 _registered_cellpose_inference)
from cellmatchr.parameter_wizard import (ParameterWizardApp, _apply_margin,
                                         _original_color_view)
from cellmatchr.retraining import (TRAINING_PRESETS, _focus_training_pairs,
                                   _accepted_manual_additions,
                                   registered_microsam_foreground_threshold,
                                   validate_advanced)


class BrightfieldCompatibilityTests(unittest.TestCase):
    def test_old_project_keeps_fluorescence_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "cellmatchr_project.json").write_text(json.dumps({
                "project_name": "older", "input_type": "vsi",
                "seg_preprocess_mode": "local_bg_subtract"}))
            cfg = load_project_config(str(project))
            self.assertEqual(cfg["slide_signal_type"], "fluorescence")
            self.assertEqual(cfg["seg_raw_mean_max"], 0)
            self.assertEqual(cfg["seg_preprocess_mode"], "local_bg_subtract")

    def test_brightfield_working_gray_and_inversion_leave_color_source_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stain.tif"
            rgb = np.full((48, 64, 3), 235, dtype=np.uint8)
            rgb[8:24, 10:30] = [45, 30, 135]
            tifffile.imwrite(source, rgb, photometric="rgb")
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            paths = seg.save_raw_split_tiffs(str(source), str(root), "stain",
                                             brightfield=True)
            native = tifffile.imread(paths[0])
            self.assertEqual(native.shape, rgb.shape[:2])
            self.assertLess(float(native[12, 12]), float(native[35, 35]))
            model_input, _ = seg.preprocess_tile_for_sam(
                native, mode="brightfield_invert")
            self.assertGreater(float(model_input[12, 12]),
                               float(model_input[35, 35]))
            shown = _original_color_view(str(source), native.shape)
            np.testing.assert_array_equal(shown[12, 12], rgb[12, 12])
            del shown  # Release the memory map before Windows removes the TIFF.
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_brightfield_max_filter_keeps_dark_cells(self):
        raw = np.full((24, 24), 230, dtype=np.uint16)
        labels = np.zeros_like(raw, dtype=np.uint32)
        labels[2:9, 2:9] = 1
        raw[2:9, 2:9] = 45
        labels[14:21, 14:21] = 2
        raw[14:21, 14:21] = 190
        with patch.object(seg, "MIN_OBJECT_SIZE", 0), \
             patch.object(seg, "MAX_OBJECT_SIZE", 0), \
             patch.object(seg, "MAX_MAJOR_MINOR_RATIO", 0), \
             patch.object(seg, "RAW_MEAN_MIN", 0), \
             patch.object(seg, "RAW_MEAN_MAX", 150):
            filtered = seg.filter_labeled_tile_by_raw_mean_and_shape(labels, raw)
        self.assertEqual(int(filtered[4, 4]), 1)
        self.assertEqual(int(filtered[16, 16]), 0)
        self.assertGreater(_apply_margin("max_intensity", 100), 100)

    def test_known_nissl_model_retains_its_training_preprocess(self):
        model_path = "/tmp/example-nissl-model"
        record = {"model_path": model_path, "preprocess": "raw_percentile",
                  "suggested_inference": {"cellprob_threshold": -2.0,
                                          "flow_threshold": 1.0}}
        with patch("cellmatchr.retraining.list_trained_models", return_value=[record]):
            self.assertEqual(_brightfield_preprocess_for_model(
                "cellpose", "cpsam", model_path, "/tmp/example"), "raw_percentile")
            self.assertEqual(_registered_cellpose_inference(
                model_path, "/tmp/example"), record["suggested_inference"])
        self.assertEqual(_brightfield_preprocess_for_model(
            "cellpose", "cpsam", "", ""), "brightfield_invert")
        self.assertEqual(_brightfield_preprocess_for_model(
            "cellpose", "rat_nissl_v2", "", ""), "raw_percentile")

    def test_microsam_recall_setting_is_bound_to_its_registered_model(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            model_dir = library / "lm_recall"
            model_dir.mkdir()
            weights = model_dir / "model.pth"
            weights.touch()
            record = {"backend": "microsam", "model_path": str(weights),
                      "suggested_inference": {"foreground_threshold": 0.2}}
            (model_dir / "model.json").write_text(json.dumps(record))
            with patch.dict(os.environ, {"CELLMATCHR_MODEL_LIBRARY": directory}):
                self.assertEqual(registered_microsam_foreground_threshold(
                    str(weights)), 0.2)
                self.assertIsNone(registered_microsam_foreground_threshold(
                    str(library / "older_model.pth")))
                record["suggested_inference"]["foreground_threshold"] = 1.5
                (model_dir / "model.json").write_text(json.dumps(record))
                self.assertIsNone(registered_microsam_foreground_threshold(
                    str(weights)))

    def test_brightfield_wizard_writes_visible_global_brightness_filter(self):
        wizard = object.__new__(ParameterWizardApp)
        wizard.cfg = {"seg_per_channel_filters": "{}"}
        wizard.brightfield = True
        wizard.channels = [0]
        wizard.pick_order = ["min_size", "max_size", "max_ratio", "max_intensity"]
        wizard.results = {0: {"min_size": 40, "max_size": None,
                              "max_ratio": None, "max_intensity": 100}}
        wizard._finished = False
        wizard.viewer = None
        with patch("cellmatchr.parameter_wizard.save_config"), \
             patch("cellmatchr.parameter_wizard.save_project_config"):
            wizard._finish()
        self.assertFalse(wizard.cfg["seg_per_channel_filters_enabled"])
        self.assertGreater(wizard.cfg["seg_raw_mean_max"], 100)
        self.assertEqual(wizard.cfg["seg_raw_mean_min"], 0)
        self.assertEqual(wizard.cfg["seg_per_channel_filters"], "{}")

    def test_closing_filter_wizard_without_a_pick_preserves_existing_settings(self):
        for brightfield in (False, True):
            with self.subTest(brightfield=brightfield):
                settings = {"seg_min_object_size": 80, "seg_max_ratio": 3.5,
                            "seg_raw_mean_max": 175, "seg_per_channel_filters_enabled": not brightfield,
                            "seg_per_channel_filters": '{"0": {"min_size": 65}}'}
                wizard = ParameterWizardApp(dict(settings), {"channels": [{"channel": 0}],
                                                             "slide_signal_type": "brightfield" if brightfield
                                                             else "fluorescence"})
                wizard._current_picks = {key: None for key in wizard.pick_order}
                with (patch("cellmatchr.parameter_wizard.save_config") as save,
                      patch("cellmatchr.parameter_wizard.save_project_config") as save_project):
                    wizard._on_window_closed()
                self.assertEqual(wizard.cfg, settings)
                save.assert_not_called()
                save_project.assert_not_called()

    def test_wizard_applies_picked_channel_without_clearing_an_empty_channel(self):
        existing = {"1": {"min_size": 35, "max_ratio": 4}, "2": {"min_size": 90}}
        wizard = ParameterWizardApp({"seg_per_channel_filters": json.dumps(existing)},
                                    {"channels": [{"channel": 0}, {"channel": 1}]})
        wizard.results[0] = {"min_size": 60}
        wizard._ch_idx = 1
        wizard._current_picks = {key: None for key in wizard.pick_order}
        with patch("cellmatchr.parameter_wizard.save_config"), \
             patch("cellmatchr.parameter_wizard.save_project_config"):
            wizard._on_window_closed()
        settings = json.loads(wizard.cfg["seg_per_channel_filters"])
        self.assertGreater(settings["0"]["min_size"], 0)
        self.assertEqual(settings["1"], existing["1"])
        self.assertEqual(settings["2"], existing["2"])


class TrainingPresetTests(unittest.TestCase):
    def test_presets_have_real_training_settings_and_short_run(self):
        self.assertEqual(TRAINING_PRESETS["High recall"]["sampling_focus"],
                         "missed_cells")
        self.assertEqual(TRAINING_PRESETS["High precision"]["sampling_focus"],
                         "rejected_cells")
        self.assertLess(TRAINING_PRESETS["Gentle follow-up"]["learning_rate"],
                        TRAINING_PRESETS["Default training"]["learning_rate"])
        self.assertLess(TRAINING_PRESETS["Short trial"]["epochs"],
                        TRAINING_PRESETS["Default training"]["epochs"])
        for name, focus in (("Recall follow-up", "missed_cells"),
                            ("Precision follow-up", "rejected_cells")):
            self.assertEqual(TRAINING_PRESETS[name]["sampling_focus"], focus)
            self.assertEqual(TRAINING_PRESETS[name]["epochs"],
                             TRAINING_PRESETS["Gentle follow-up"]["epochs"])
            self.assertEqual(TRAINING_PRESETS[name]["learning_rate"],
                             TRAINING_PRESETS["Gentle follow-up"]["learning_rate"])

    def test_focus_repeats_only_matching_training_pairs(self):
        pairs = [("a", "A"), ("b", "B"), ("c", "C")]
        examples = [{"manual_added_count": 2, "rejected_count": 0},
                    {"manual_added_count": 0, "rejected_count": 4},
                    {"manual_added_count": 0, "rejected_count": 0}]
        self.assertEqual(_focus_training_pairs(pairs, examples, "missed_cells"),
                         [pairs[0], pairs[0], pairs[1], pairs[2]])
        self.assertEqual(_focus_training_pairs(pairs, examples, "rejected_cells"),
                         [pairs[0], pairs[1], pairs[1], pairs[2]])
        self.assertEqual(validate_advanced("cellpose", {
            "sampling_focus": "missed_cells"})["sampling_focus"], "missed_cells")
        with self.assertRaisesRegex(ValueError, "Sampling focus"):
            validate_advanced("cellpose", {"sampling_focus": "unsupported"})

    def test_manual_focus_uses_only_current_accepted_additions(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory)
            (sample / "_manual_additions_ch0.tsv").write_text(
                "label_id\tsource\n3\tmanual_missed_cell\n4\tmanual_missed_cell\n"
                "3\tmanual_missed_cell\n")
            self.assertEqual(_accepted_manual_additions(sample, 0, {1, 3}), 1)


if __name__ == "__main__":
    unittest.main()
