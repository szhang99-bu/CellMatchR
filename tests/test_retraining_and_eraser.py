"""Focused checks for the reviewed-data and whole-object erase workflows."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tifffile

from cellmatchr.roi_editor import RoiEditorApp, _remove_touched_objects
from cellmatchr.retraining import collect_examples, train_model
from cellmatchr.validation_state import get_change_state


class WholeObjectEraserTests(unittest.TestCase):
    def test_touching_one_pixel_removes_the_whole_prediction(self):
        predicted = np.zeros((8, 8), dtype=np.uint32)
        predicted[1:4, 1:4] = 1
        predicted[5:7, 5:7] = 2
        strokes = np.zeros_like(predicted)
        strokes[2, 2] = 1
        edited, removed = _remove_touched_objects(predicted, predicted, strokes)
        self.assertEqual(removed, {1})
        self.assertFalse(np.any(edited == 1))
        self.assertEqual(int(np.sum(edited == 2)), 4)
        self.assertEqual(int(np.sum(predicted == 1)), 9)

    def test_erase_undo_restores_preview_before_save(self):
        app = object.__new__(RoiEditorApp)
        original = np.zeros((5, 5), dtype=np.int32)
        original[1:4, 1:4] = 3
        app._base_display_mask = original
        stroke = np.zeros((5, 5), dtype=np.uint8)
        stroke[2, 2] = 1
        app._erase_lyr = Mock(data=stroke)
        app._mask_lyr = Mock()
        app._set_status = Mock()
        app._update_erasure_preview()
        self.assertFalse(np.any(app._mask_lyr.data == 3))
        def undo():
            app._erase_lyr.data[2, 2] = 0
        app._erase_lyr.undo.side_effect = undo
        app._undo_eraser()
        np.testing.assert_array_equal(app._mask_lyr.data, original)


class StarDistRoutingTests(unittest.TestCase):
    def test_stardist_labels_use_existing_tile_filter_and_merge(self):
        from cellmatchr import segmentation as seg

        image = np.full((8, 8), 100, dtype=np.uint16)
        predicted = np.zeros((8, 8), dtype=np.uint16)
        predicted[2:6, 2:6] = 1
        model = Mock()
        model.predict_instances.return_value = (predicted, {})
        with patch.object(seg, "preprocess_tile_for_sam", return_value=(
                np.full((8, 8), 255, dtype=np.uint8),
                np.zeros((8, 8, 3), dtype=np.uint8))), \
             patch.object(seg, "MIN_OBJECT_SIZE", 0), \
             patch.object(seg, "MAX_OBJECT_SIZE", 0), \
             patch.object(seg, "MAX_MAJOR_MINOR_RATIO", 0), \
             patch.object(seg, "RAW_MEAN_MIN", 0), \
             patch.object(seg, "SKIP_EMPTY_TILES", False), \
             patch.object(seg, "TEST_RUN", False):
            result = seg.segment_by_tiling(None, None, image, (8, 8), (0, 0), 2,
                                           stardist_model=model)
        np.testing.assert_array_equal(result, predicted)
        model.predict_instances.assert_called_once()
        self.assertEqual(float(model.predict_instances.call_args.args[0].max()), 1.0)


class ReviewedTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.run = self.project / "Segmentation"
        self.run.mkdir()

    def sample(self, name, *, complete=True, decisions="1\tYES\n2\tARTIFACT\n"):
        sample = self.run / name
        sample.mkdir()
        raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[1:3, 1:3] = 1
        labels[5:7, 5:7] = 2
        tifffile.imwrite(sample / f"{name}_split_ch1.tif", raw)
        tifffile.imwrite(sample / f"{name}_predict_original_ch1_method.tif", labels)
        validation = sample / "ROI Validation Results"
        validation.mkdir()
        (validation / "roi_verification_results_ch1.tsv").write_text(
            "ROI_ID\tDecision\n" + decisions)
        if complete:
            (validation / "_validation_complete_ch1").touch()
        return sample

    def test_reviewed_yes_cells_only_are_used(self):
        self.sample("sample_a")
        self.sample("unfinished", complete=False)
        examples, skipped = collect_examples(str(self.run), 1)
        self.assertEqual([e["sample"] for e in examples], ["sample_a"])
        self.assertEqual(examples[0]["accepted_ids"], [1])
        self.assertEqual(examples[0]["rejected_count"], 1)
        self.assertTrue(any("unfinished" in reason for reason in skipped))

    def test_erasing_rejected_cell_preserves_completed_legacy_review(self):
        sample = self.sample("a")
        mask_path = sample / "a_predict_original_ch1_method.tif"
        original = tifffile.imread(mask_path)
        stroke = np.zeros_like(original)
        stroke[5, 5] = 1
        edited, removed = _remove_touched_objects(original, original, stroke)
        self.assertEqual(removed, {2})
        app = object.__new__(RoiEditorApp)
        app.channel = 1
        app._scale = 1
        app._mask_lyr = Mock()
        app._add_lyr = Mock()
        app._erase_lyr = Mock()
        app._set_status = Mock()
        app._normal_save({"mask": str(mask_path)}, original, edited, 0, 1)
        self.assertEqual(get_change_state(str(sample), 1, str(mask_path))["status"],
                         "current")
        self.assertTrue((sample / "ROI Validation Results" /
                         "_validation_complete_ch1").exists())
        examples, skipped = collect_examples(str(self.run), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(examples[0]["accepted_ids"], [1])

    def test_training_creates_new_model_folder_and_removes_staged_images(self):
        self.sample("a")
        self.sample("b")
        def prepare(example, stage, index, mode):
            image = stage / f"image_{index}.tif"
            label = stage / f"label_{index}.tif"
            tifffile.imwrite(image, np.zeros((8, 8), dtype=np.uint8))
            tifffile.imwrite(label, np.ones((8, 8), dtype=np.uint32))
            return str(image), str(label)
        def trainer(train, val, output, epochs, base):
            self.assertEqual((len(train), len(val), epochs), (1, 1, 2))
            model = output / "trained_model"
            model.touch()
            return str(model)
        with patch("cellmatchr.retraining._prepare_pair", side_effect=prepare), \
             patch.dict("cellmatchr.retraining.TRAINERS", {"cellpose": trainer}):
            result = train_model(str(self.project), "Segmentation", 1,
                                 "cellpose", 2, "cpsam")
        folder = Path(result["model_path"]).parent
        self.assertTrue((folder / "model.json").is_file())
        self.assertEqual(json.loads((folder / "model.json").read_text())["backend"],
                         "cellpose")
        self.assertFalse(list(folder.glob(".training_data_*")))


if __name__ == "__main__":
    unittest.main()
