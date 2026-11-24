"""The launcher executes segmentation.py as a file, outside package mode."""

import importlib.util
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile


@unittest.skipUnless(importlib.util.find_spec("micro_sam"),
                     "Micro-SAM is required for the segmentation launch check")
class SegmentationScriptLaunchTests(unittest.TestCase):
    def test_resume_checkpoint_survives_interrupted_write_and_reconnect(self):
        from cellmatchr import segmentation
        from cellmatchr.launcher import CellMatchrLauncher

        def interrupt_write(_data, stream, **_kwargs):
            stream.write('{"completed": ')
            raise PermissionError("Shared drive disconnected during write")

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run = Path(directory) / "Segmentation_microsam"
            segmentation._write_run_progress(run, 8, {"sample1"}, False)
            with patch("cellmatchr.file_paths.json.dump", side_effect=interrupt_write):
                segmentation._write_run_progress(run, 8, {"sample1", "sample2"}, False)

            launcher = CellMatchrLauncher.__new__(CellMatchrLauncher)
            launcher.config = {"project_output_folder": directory}
            with patch.object(launcher, "_find_variants", side_effect=PermissionError):
                self.assertIsNone(launcher._seg_run_progress())

            restored = launcher._seg_run_progress()
            self.assertEqual(restored["folder"], str(run))
            self.assertEqual(restored["completed"], ["sample1"])
            checkpoint = run / "_run_progress.json"
            self.assertEqual(json.loads(checkpoint.read_text())["n_completed"], 1)
            self.assertEqual(list(run.iterdir()), [checkpoint])

    def test_tiff_time_channel_and_depth_axes_produce_legacy_split_files(self):
        from cellmatchr.segmentation import save_raw_split_tiffs

        data = np.arange(2 * 3 * 2 * 8 * 9, dtype=np.uint16).reshape(2, 3, 2, 8, 9)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            for axes, pixels, expected in (
                ("TZCYX", data, data[0].max(axis=0)),
                ("TZYX", data[:, :, 0], data[0, :, 0].max(axis=0)[None]),
                ("QYX", data[0, 0], data[0, 0]),
            ):
                with self.subTest(axes=axes):
                    source = folder / f"{axes}.tif"
                    tifffile.imwrite(source, pixels, photometric="minisblack", metadata={"axes": axes})
                    result = save_raw_split_tiffs(str(source), directory, axes)
                    self.assertEqual(list(result), list(range(len(expected))))
                    for index, channel in enumerate(expected):
                        self.assertEqual(Path(result[index]).name, f"{axes}_split_ch{index}.tif")
                        np.testing.assert_array_equal(tifffile.imread(result[index]), channel)

    def test_tile_stitch_keeps_cell_ids_across_anchored_seams(self):
        from cellmatchr import segmentation

        image = np.zeros((8, 13), dtype=np.uint8)
        image[1:3, 8:12] = 1    # Crosses the actual join with the last anchored tile.
        image[5:7, 4:6] = 2    # Adjacent cells touching at the first seam.
        image[5:7, 6:8] = 3
        model = SimpleNamespace(predict_instances=lambda pixels:
                                (np.rint(pixels * 255).astype(np.uint32), None))
        settings = dict(SKIP_EMPTY_TILES=False, MIN_OBJECT_SIZE=0, MAX_OBJECT_SIZE=0,
                        RAW_MEAN_MIN=0, RAW_MEAN_MAX=0, MAX_MAJOR_MINOR_RATIO=0)
        with patch.multiple(segmentation, **settings), \
                patch.object(segmentation, "preprocess_tile_for_sam",
                             side_effect=lambda tile, **kwargs: (tile, None)), \
                patch.object(segmentation, "should_save_debug_images", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            for pixels in (image, image.T):
                mask = segmentation.segment_by_tiling(
                    None, None, pixels, (8, 8), (2, 2), 2, stardist_model=model)
                self.assertEqual(mask.dtype, np.uint32)
                np.testing.assert_array_equal(mask > 0, pixels > 0)
                ids = []
                for value in (1, 2, 3):
                    ids.append(set(mask[pixels == value]))
                    self.assertEqual(len(ids[-1]), 1)
                self.assertEqual(len(set.union(*ids)), 3)

    def test_microsam_without_decoder_gets_amg_size_argument(self):
        from cellmatchr import segmentation
        from micro_sam.instance_segmentation import AutomaticMaskGenerator
        from unittest.mock import Mock

        prediction = np.zeros((16, 16), dtype=np.uint32)
        prediction[5:9, 5:9] = 1
        segmenter = Mock(spec=AutomaticMaskGenerator)
        segmenter.generate.return_value = prediction
        predictor = Mock(original_size=prediction.shape, input_size=prediction.shape)
        with patch.object(segmentation, "should_save_debug_images", return_value=False), \
                patch.multiple(segmentation, MIN_OBJECT_SIZE=0, MAX_OBJECT_SIZE=0,
                               RAW_MEAN_MIN=0, RAW_MEAN_MAX=0), \
                contextlib.redirect_stdout(io.StringIO()):
            mask = segmentation.segment_by_tiling(
                predictor, segmenter, prediction.astype(np.uint8) * 200, (16, 16), (0, 0),
                2, min_size=7)
        segmenter.generate.assert_called_once_with(min_mask_region_area=7)
        np.testing.assert_array_equal(mask, prediction)

    def test_direct_script_reaches_microsam_model_loader(self):
        script = Path(__file__).resolve().parents[1] / "cellmatchr" / "segmentation.py"
        code = """
import os, runpy, sys
from pathlib import Path
import numpy as np
import tifffile

script = Path(sys.argv[1])
root = Path.cwd()
inputs = root / 'input'
inputs.mkdir()
tifffile.imwrite(inputs / 'example.tif', np.zeros((16, 16), dtype=np.uint16))
sys.path.insert(0, str(script.parent))
namespace = runpy.run_path(str(script), run_name='segmentation')
state = namespace['main'].__globals__
state.update(ROOT_INPUT_FOLDER=str(inputs), MAIN_OUTPUT_FOLDER=str(root / 'output'),
             PROJECT_OUTPUT_FOLDER=str(root), INPUT_TYPE='tif', SEG_BACKEND='microsam',
             CUSTOM_MODEL_PATH='', _restore_cached_microsam_decoder=lambda model: None)

class ReachedModelLoad(Exception):
    pass

def stop_before_model_download(**kwargs):
    if kwargs['model_type'] != 'vit_b_lm':
        raise AssertionError(kwargs)
    raise ReachedModelLoad()

state['micro_sam'].automatic_segmentation.get_predictor_and_segmenter = stop_before_model_download
try:
    state['main']()
except ReachedModelLoad:
    print('Reached Micro-SAM model load from standalone script')
else:
    raise AssertionError('The model loader was never reached')
"""
        with tempfile.TemporaryDirectory(prefix="cellmatchr_script_launch_") as work:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = ""
            environment["MPLCONFIGDIR"] = work
            environment["NUMBA_CACHE_DIR"] = work
            result = subprocess.run(
                [sys.executable, "-c", code, str(script)], cwd=work,
                env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Reached Micro-SAM model load", result.stdout)

    def test_restore_decoder_only_from_registry_verified_backup(self):
        import pooch
        from micro_sam import util
        from cellmatchr.segmentation import _restore_cached_microsam_decoder

        with tempfile.TemporaryDirectory() as work:
            name = "vit_b_lm_decoder"
            decoder = Path(work) / name
            backup = decoder.with_suffix(".bak")
            original = b"official decoder bytes"
            backup.write_bytes(original)
            registry = pooch.create(
                path=work, base_url="", registry={name: f"xxh128:{pooch.file_hash(backup, alg='xxh128')}"}
            )
            with patch.object(util, "models", return_value=registry):
                decoder.write_bytes(b"previously rewritten decoder")
                _restore_cached_microsam_decoder("vit_b_lm")
                self.assertEqual(decoder.read_bytes(), original)
                self.assertEqual(backup.read_bytes(), original)

                backup.write_bytes(b"unverified backup")
                _restore_cached_microsam_decoder("vit_b_lm")
                self.assertEqual(decoder.read_bytes(), original)

                decoder.write_bytes(b"unverified active file")
                _restore_cached_microsam_decoder("vit_b_lm")
                self.assertEqual(decoder.read_bytes(), b"unverified active file")
                self.assertEqual(set(Path(work).iterdir()), {decoder, backup})


if __name__ == "__main__":
    unittest.main()
