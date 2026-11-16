import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from cellmatchr import analysis, segmentation
from cellmatchr.file_paths import filesystem_path, read_tiff
from cellmatchr.launcher import CellMatchrLauncher
from cellmatchr.roi_editor import _find_mask, _find_raw


class WindowsWorkflowTests(unittest.TestCase):
    def test_pending_status_callbacks_ignore_closed_panels(self):
        app = object.__new__(CellMatchrLauncher)
        app._ana_val_status_frame = Mock()
        app._val_prog_frame = Mock()
        for frame in (app._ana_val_status_frame, app._val_prog_frame):
            frame.winfo_exists.return_value = False
        app._refresh_ana_val_status()
        app._refresh_val_status()
        app._ana_val_status_frame.winfo_children.assert_not_called()
        app._val_prog_frame.winfo_children.assert_not_called()

    def test_file_lookup_uses_literal_folder_and_full_channel_number(self):
        with tempfile.TemporaryDirectory() as root:
            sample = Path(root) / "field [1]"
            sample.mkdir()
            raw = sample / "field_split_ch1.tif"
            mask = sample / "field_predict_original_ch1_test.tif"
            other = sample / "field_predict_original_ch10_test.tif"
            for file in (raw, mask, other):
                file.touch()
            self.assertEqual(_find_raw(str(sample), 1), str(raw))
            self.assertEqual(_find_mask(str(sample), 1), str(mask))
            self.assertEqual(_find_mask(str(sample), 10), str(other))
            pattern = analysis.PRED_TIFF_PATTERN.format(ch=1)
            self.assertEqual(analysis._matching_files(str(sample), pattern), [str(mask)])
            mask.unlink()
            self.assertIsNone(_find_mask(str(sample), 1))
            self.assertFalse(analysis._matching_files(str(sample), pattern))

    def test_prediction_tiff_and_png_can_use_deep_windows_paths(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / ("a" * 67 + "[1]") / ("b" * 70)
            folder.mkdir(parents=True)
            mask_path = folder / ("labels-" + "c" * 90 + ".tif")
            qc_path = folder / ("overlay-" + "d" * 90 + ".png")
            mask = np.array([[0, 0, 2], [0, 2, 2]], dtype=np.uint16)
            try:
                segmentation.save_segmentation_tiff(mask_path, mask)
                segmentation.make_final_qc_mask_png(mask, qc_path, downsample=1)
                np.testing.assert_array_equal(read_tiff(mask_path), mask)
                with Image.open(filesystem_path(qc_path)) as preview:
                    self.assertEqual(preview.format, "PNG")
                    self.assertTrue(np.asarray(preview).any())
                if os.name == "nt":
                    self.assertGreaterEqual(len(str(mask_path)), 260)
            finally:
                for file in (mask_path, qc_path):
                    Path(filesystem_path(file)).unlink(missing_ok=True)

    def test_dry_run_plans_without_opening_images_or_writing_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            inputs = Path(root) / "input"
            inputs.mkdir()
            (inputs / "field.tif").touch()
            destination = Path(root) / "output"
            with (patch.multiple(segmentation, ROOT_INPUT_FOLDER=str(inputs),
                                 MAIN_OUTPUT_FOLDER=str(destination), DRY_RUN=True, TEST_RUN=False),
                  patch.object(segmentation, "save_raw_split_tiffs") as split,
                  redirect_stdout(io.StringIO()) as log):
                self.assertEqual(segmentation.main(), 0)
            split.assert_not_called()
            self.assertIn("[PLAN] field", log.getvalue())
            self.assertFalse(destination.exists())

    def test_decode_error_returns_failure_and_leaves_run_resumable(self):
        with tempfile.TemporaryDirectory() as root:
            inputs = Path(root) / "input"
            inputs.mkdir()
            (inputs / "field.tif").touch()
            destination = Path(root) / "output"
            with (patch.multiple(segmentation, ROOT_INPUT_FOLDER=str(inputs),
                                 MAIN_OUTPUT_FOLDER=str(destination), SEG_BACKEND="cellpose",
                                 CELLPOSE_VARIANT="cpsam", DRY_RUN=False, TEST_RUN=False,
                                 RESUME=False),
                  patch("cellpose.models.CellposeModel"),
                  patch.object(segmentation, "save_raw_split_tiffs",
                               side_effect=OSError("fixture decode failed")),
                  patch.object(segmentation, "_free_memory"),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                self.assertEqual(segmentation.main(), 1)
            state = json.loads((destination / "_run_progress.json").read_text())
            self.assertFalse(state["finished"])
            self.assertEqual(state["n_completed"], 0)
