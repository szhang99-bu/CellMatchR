"""Regression checks for native color input and separate segmentation identities."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tifffile

from cellmatchr import segmentation as seg
from cellmatchr.launcher import CellMatchrLauncher


class MicroSam2Tests(unittest.TestCase):
    def test_converter_tiff_feeds_native_rgb_to_model_and_gray_to_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "nissl.tif"
            rgb = np.full((32, 48, 3), (220, 230, 240), dtype=np.uint8)
            rgb[5:12, 5:12] = (12, 20, 65)
            tifffile.imwrite(source, np.moveaxis(rgb, -1, 0), imagej=True,
                             photometric="minisblack", metadata={"axes": "CYX"})
            paths, native = seg.prepare_prediction_channels(
                str(source), directory, "nissl", lif_scene=None, brightfield=True, use_microsam2=True)
            gray = tifffile.imread(paths[0])
            self.assertEqual(list(paths), [0])
            self.assertEqual(gray.shape, rgb.shape[:2])
            self.assertGreater(gray[19, 19], gray[7, 7])

            proposals = np.zeros(gray.shape, dtype=np.uint32)
            proposals[5:12, 5:12] = 1
            proposals[17:24, 17:24] = 2
            model = Mock()
            model.predict.return_value = proposals
            with patch.multiple(seg, SLIDE_SIGNAL_TYPE="brightfield", MIN_OBJECT_SIZE=0,
                                MAX_OBJECT_SIZE=0, MAX_MAJOR_MINOR_RATIO=0,
                                RAW_MEAN_MIN=0, RAW_MEAN_MAX=150):
                labels = seg.segment_by_tiling(
                    None, None, gray, (64, 64), (0, 0), 2,
                    microsam2_model=model, native_brightfield=native)
            np.testing.assert_array_equal(model.predict.call_args.args[0], rgb)
            self.assertGreater(labels[7, 7], 0)
            self.assertEqual(labels[19, 19], 0)

    def test_resume_distinguishes_microsam2_from_microsam(self):
        launcher = object.__new__(CellMatchrLauncher)
        launcher.config = {
            "input_type": "vsi", "slide_signal_type": "brightfield",
            "seg_backend": "microsam2", "seg_channels": "0",
            "seg_microsam2_model": "hvit_t_cells", "seg_model_type": "vit_b_lm",
            "seg_preprocess_mode": "brightfield_invert",
        }
        with patch.multiple(seg, SEG_BACKEND="microsam2", MICROSAM2_MODEL="hvit_t_cells",
                            CHANNELS_TO_PREDICT=[0], SLIDE_SIGNAL_TYPE="brightfield"):
            identity = seg._run_identity()
        self.assertEqual(identity, launcher._segmentation_identity_from_config())
        self.assertEqual(identity["preprocess"], "native_rgb")
        launcher.config["seg_backend"] = "microsam"
        original = launcher._segmentation_identity_from_config()
        self.assertEqual(original["model"], "vit_b_lm")
        self.assertEqual(original["preprocess"], "brightfield_invert")
        self.assertNotEqual(identity, original)
