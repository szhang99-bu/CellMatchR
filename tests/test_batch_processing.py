import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cellmatchr.batch_processing import (
    automatic_batch_masks, batch_input_folder, batch_settings_issue,
    batch_target_problem, create_batch_folder, run_batch,
)
from cellmatchr.config import DEFAULT_CONFIG, PROJECT_CONFIG_FILE


class BatchProcessingTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.template = self.root / "template"
        self.template.mkdir()
        reference = self.template / "reference"
        reference.mkdir()
        (reference / "image.tif").touch()
        self.config_path = self.template / PROJECT_CONFIG_FILE
        self.config = dict(DEFAULT_CONFIG, project_name="Reference",
                           project_output_folder=str(self.template), input_root_folder=str(reference),
                           seg_channels="1,3", seg_min_object_size=41, ana_iou_threshold="0.42")
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_setup_points_to_the_incomplete_step(self):
        self.assertIsNone(batch_settings_issue(self.config))
        self.assertEqual(batch_settings_issue(dict(self.config, project_output_folder=""))[0], 0)
        self.assertEqual(batch_settings_issue(dict(self.config, input_root_folder=""))[0], 1)
        self.assertEqual(batch_settings_issue(dict(self.config, seg_channels=""))[0], 2)
        self.assertEqual(batch_settings_issue(dict(self.config, ana_pv_channel="9"))[0], 4)
        self.assertEqual(batch_settings_issue(dict(self.config, ana_subfolder_name="../other"))[0], 4)

    def test_converted_scans_use_exports_and_their_own_roi_masks(self):
        parent = self.root / "VSI_Converted_TIFF_second"
        exports = parent / "exports"
        masks = parent / "DONOTDELETE-MASK_FOR_VSI"
        exports.mkdir(parents=True)
        masks.mkdir()
        (exports / "slide.tif").touch()
        (masks / "slide_mask_overview.tif").touch()
        config = dict(self.config, input_type="vsi", slide_signal_type="brightfield")
        self.assertEqual(batch_input_folder(parent, config), exports.resolve())
        self.assertEqual(automatic_batch_masks(exports), masks)
        self.assertFalse(batch_target_problem(exports, masks, config))
        self.assertTrue(batch_target_problem(masks, None, config))

    def _target(self, parent):
        folder = self.root / parent / "same_name"
        folder.mkdir(parents=True)
        (folder / "image.tif").touch()
        return folder

    @staticmethod
    def _make_analysis_outputs(cfg):
        output = Path(cfg["output_root_folder"])
        output.mkdir()
        (output / "combined_analysis_summary.xlsx").touch()
        (output / "analysis_run_summary.json").write_text(json.dumps({
            "workbook_written": True, "analyzed": 1, "status": "complete",
        }), encoding="utf-8")

    def test_failures_continue_and_each_folder_has_independent_settings(self):
        first, second = self._target("first"), self._target("second")
        masks = self.root / "second_masks"
        masks.mkdir()
        (masks / "image.tif").touch()
        self.config["ana_roi_mask_folder"] = "/other/template_masks"
        self.config["ana_roi_mask_choices"] = {"samples": {"image": "different.tif"}}
        snapshot = self.config_path.read_bytes()
        visits = []

        def stage(command, _environment):
            script = Path(command[1]).name
            cfg = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            visits.append((script, cfg))
            if script == "analysis.py":
                self._make_analysis_outputs(cfg)
            return int(cfg["input_root_folder"] == str(first))

        batch = create_batch_folder(self.root / "results")
        rows = run_batch(self.config, [(first, None), (second, masks)], batch,
                         run_stage=stage, stopped=lambda: False, on_update=lambda *args: None,
                         log=lambda line: None)
        self.assertEqual([row["status"] for row in rows], ["failed", "complete"])
        self.assertEqual([stage_name for stage_name, _ in visits],
                         ["segmentation.py", "segmentation.py", "analysis.py"])
        self.assertNotEqual(rows[0]["output"], rows[1]["output"])
        self.assertEqual(visits[0][1]["ana_roi_mask_folder"], "")
        job = visits[-1][1]
        self.assertEqual(job["ana_roi_mask_folder"], str(masks))
        self.assertEqual(job["ana_roi_mask_choices"], {})
        self.assertEqual(job["seg_min_object_size"], 41)
        self.assertEqual(job["ana_iou_threshold"], "0.42")
        self.assertEqual(self.config_path.read_bytes(), snapshot)
        self.assertEqual(json.loads((batch / "batch_report.json").read_text(encoding="utf-8")), rows)

    def test_cancelling_does_not_start_another_stage_or_folder(self):
        folders = [self._target("first"), self._target("second")]
        cancelled = False

        def stage(_command, _environment):
            nonlocal cancelled
            self.assertFalse(cancelled)
            cancelled = True
            return 1

        batch = create_batch_folder(self.root / "results")
        rows = run_batch(self.config, [(folder, None) for folder in folders], batch,
                         run_stage=stage, stopped=lambda: cancelled, on_update=lambda *args: None,
                         log=lambda line: None)
        self.assertEqual([row["status"] for row in rows], ["cancelled", "not run"])
