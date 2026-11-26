"""Regression checks for the one-channel object-review layout."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from matplotlib.figure import Figure
import numpy as np
import tifffile

from cellmatchr.cell_mask_editor import merge_cell_crop
from cellmatchr.roi_editor import _prediction_baseline_path
from cellmatchr.validation_state import get_change_state, get_review_decision, save_review_decision
from cellmatchr.validator import RoiVerifierApp, _build_review_axes


class ValidatorLayoutTests(unittest.TestCase):
    def test_image_and_context_axes_exist_with_one_or_many_channels(self):
        for count in (1, 2, 4):
            with self.subTest(channels=count):
                figure = Figure()
                color, gray, nearby, merge, context = _build_review_axes(figure, count)
                self.assertEqual([len(color), len(gray), len(nearby)], [count] * 3)
                self.assertIsNot(merge, context)
                self.assertGreater(context.get_position().height, 0)
                self.assertEqual(len(figure.axes), count * 3 + 2)

    def test_editing_one_label_preserves_neighbors_and_saved_decisions(self):
        with TemporaryDirectory() as folder:
            sample = Path(folder)
            mask_path = sample / "sample_predict_original_ch0.tif"
            original = np.zeros((10, 12), dtype=np.uint16)
            original[1:3, 1:3] = 11
            original[5:7, 6:8] = 900
            original[5:7, 8:10] = 901
            tifffile.imwrite(mask_path, original)
            val_dir = sample / "ROI Validation Results"
            val_dir.mkdir()
            decisions = val_dir / "roi_verification_results_ch0.tsv"
            decisions.write_text("ROI_ID\tDecision\tNote\n11\tYES\tprior decision\n")

            reviewer = RoiVerifierApp.__new__(RoiVerifierApp)
            reviewer.filtered_mask_path = str(mask_path)
            reviewer.filtered_mask = reviewer._read_prediction()
            reviewer.props = [{"label": 900}]
            reviewer.val_channel = 0
            reviewer.update_roi_display = Mock()
            region = np.s_[4:9, 5:11]
            edited = original[region] == 900
            edited[1, 1] = False  # erase one pixel of the target
            edited[3, 1] = True   # add one pixel of background
            edited[1, 3] = True   # brush crossed a neighboring prediction

            self.assertTrue(reviewer._apply_mask_edit(0, region, edited))
            expected = original.copy()
            expected[5, 6] = 0
            expected[7, 6] = 900
            np.testing.assert_array_equal(tifffile.imread(mask_path), expected)
            np.testing.assert_array_equal(tifffile.imread(_prediction_baseline_path(str(mask_path), 0)), original)
            self.assertEqual(reviewer.props[0], {"label": 900, "bbox-0": 5, "bbox-1": 6,
                                                "bbox-2": 8, "bbox-3": 8})
            reviewer.update_roi_display.assert_called_once_with(recalc_roi=False)
            self.assertTrue(reviewer.focused_review)
            self.assertEqual(get_change_state(folder, 0, str(mask_path))["pending_ids"], [900, 901])
            self.assertEqual(get_review_decision(folder, 0, 11)["Note"], "prior decision")
            self.assertFalse(reviewer._apply_mask_edit(0, region, edited))

            save_review_decision(folder, 0, str(mask_path), 900, "YES", "reshaped")
            save_review_decision(folder, 0, str(mask_path), 901, "NO", "neighbor")
            self.assertEqual(get_change_state(folder, 0, str(mask_path))["status"], "current")
            self.assertIn("900\tYES\treshaped", decisions.read_text())

            # An edit opened before the TIFF changed must not overwrite the other writer's mask.
            tifffile.imwrite(mask_path, original.astype(np.uint32), description="Another edit")
            with self.assertRaisesRegex(RuntimeError, "mask changed"):
                reviewer._apply_mask_edit(0, region, edited)
            np.testing.assert_array_equal(tifffile.imread(mask_path), original)

    def test_erasing_entire_cell_requires_a_review_decision_instead(self):
        crop = np.array([[7, 8], [0, 0]], dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "Artifact"):
            merge_cell_crop(crop, 7, crop == 8)
