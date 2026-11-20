"""The launcher executes segmentation.py as a file, outside package mode."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("micro_sam"),
                     "Micro-SAM is required for the segmentation launch check")
class SegmentationScriptLaunchTests(unittest.TestCase):
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
             CUSTOM_MODEL_PATH='', _patch_decoder_checkpoints=lambda model: None)

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


if __name__ == "__main__":
    unittest.main()
