"""Compatibility checks for brightfield display, filtering, and training presets."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr import segmentation as seg
from cellmatchr.config import load_project_config
from cellmatchr.launcher import _brightfield_preprocess_for_model
from cellmatchr.parameter_wizard import _apply_margin, _original_color_view
from cellmatchr.retraining import (TRAINING_PRESETS, _focus_training_pairs,
                                   _accepted_manual_additions, validate_advanced)


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
        record = {"model_path": model_path, "preprocess": "raw_percentile"}
        with patch("cellmatchr.retraining.list_trained_models", return_value=[record]):
            self.assertEqual(_brightfield_preprocess_for_model(
                "cellpose", "cpsam", model_path, "/tmp/example"), "raw_percentile")
        self.assertEqual(_brightfield_preprocess_for_model(
            "cellpose", "cpsam", "", ""), "brightfield_invert")
        self.assertEqual(_brightfield_preprocess_for_model(
            "cellpose", "rat_nissl_v2", "", ""), "raw_percentile")


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
