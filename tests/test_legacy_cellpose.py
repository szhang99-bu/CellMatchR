"""Keep the newer Cellpose default and the isolated legacy choices separate."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cellmatchr import segmentation
from cellmatchr.launcher import CellMatchrLauncher
from cellmatchr.legacy_cellpose import segmentation_command


class LegacyCellposeTests(unittest.TestCase):
    def test_existing_project_defaults_to_cellpose_sam(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "old_project.json"
            config.write_text(json.dumps({"seg_backend": "cellpose", "seg_channels": "0"}))
            with patch.multiple(segmentation, SEG_BACKEND="microsam",
                                CELLPOSE_VARIANT="cpsam", CHANNELS_TO_PREDICT=[2]):
                segmentation._apply_config(str(config))
                self.assertEqual(segmentation.CELLPOSE_VARIANT, "cpsam")
            launcher = CellMatchrLauncher.__new__(CellMatchrLauncher)
            launcher.config = {"seg_backend": "cellpose", "seg_channels": "0"}
            self.assertEqual(launcher._segmentation_identity_from_config()["model"],
                             "cpsam")

    def test_legacy_choice_uses_separate_runtime_only_for_segmentation(self):
        settings = {"seg_backend": "cellpose", "seg_cellpose_variant": "cyto2"}
        with patch("cellmatchr.legacy_cellpose.ensure_runtime",
                   return_value=("/separate/python", "/models/cyto2")) as prepare:
            command, environment = segmentation_command(
                "/app/segmentation.py", "/config.json", settings)
            self.assertEqual(command[0], "/separate/python")
            self.assertEqual(environment["CELLMATCHR_LEGACY_CELLPOSE_MODEL_PATH"],
                             "/models/cyto2")
            ordinary, ordinary_env = segmentation_command(
                "/app/analysis.py", "/config.json", settings)
            self.assertNotEqual(ordinary[0], "/separate/python")
            self.assertNotIn("CELLMATCHR_LEGACY_CELLPOSE_MODEL_PATH", ordinary_env)
            prepare.assert_called_once_with("cyto2")

    def test_unknown_legacy_model_is_rejected_before_run(self):
        with self.assertRaises(ValueError):
            from cellmatchr.legacy_cellpose import ensure_runtime
            ensure_runtime("unknown")

    def test_new_run_name_identifies_legacy_model(self):
        with tempfile.TemporaryDirectory() as folder:
            launcher = CellMatchrLauncher.__new__(CellMatchrLauncher)
            launcher.config = {"seg_backend": "cellpose",
                               "seg_cellpose_variant": "rat_nissl_v2",
                               "project_output_folder": folder}
            self.assertTrue(launcher._suggest_segmentation_suffix().startswith(
                "cellpose_rat_nissl_v2_"))


if __name__ == "__main__":
    unittest.main()
