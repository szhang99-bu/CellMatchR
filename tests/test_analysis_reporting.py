"""Synthetic checks for analysis outcomes and batch reporting."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from cellmatchr import analysis
from cellmatchr.launcher import CellMatchrLauncher, GREEN, YELLOW
from cellmatchr.validation_state import save_edited_mask, save_review_decision


class AnalysisReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.seg_root = self.root / "Segmentation"
        self.out_root = self.root / "Analysis"
        self.seg_root.mkdir()
        self.out_root.mkdir()

        settings = {
            "MICROSAM_OUTPUT_ROOT": str(self.seg_root),
            "ANALYSIS_OUTPUT_ROOT": str(self.out_root),
            "COI_CHANNEL": 1,
            "PM_CHANNEL": 3,
            "ROI_MASK_FOLDER": "",
            "SAVE_QC_PNG": False,
            "ANALYSIS_SUBFOLDER_NAME": "analysis_results",
            "APPLY_PARTIAL_VALIDATION": False,
        }
        for key, value in settings.items():
            patcher = patch.object(analysis, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_sample(self, name, *, missing=(), shape_mismatch=False):
        sample = self.seg_root / name
        sample.mkdir()
        for ch in (1, 3):
            raw_name = f"{name}_split_ch{ch}.tif"
            mask_name = f"{name}_predict_original_ch{ch}_local_bg_subtract_tile2048.tif"
            raw = np.full((8, 8), 100 + ch, dtype=np.uint16)
            labels = np.zeros((7, 7) if (shape_mismatch and ch == 3) else (8, 8),
                              dtype=np.uint16)
            labels[2:5, 2:5] = 1
            if (ch, "raw") not in missing:
                tifffile.imwrite(sample / raw_name, raw)
            if (ch, "mask") not in missing:
                tifffile.imwrite(sample / mask_name, labels)
        return sample

    def run_analysis(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analysis.main()
        workbook = self.out_root / "combined_analysis_summary.xlsx"
        report = self.out_root / "analysis_run_summary.json"
        self.assertTrue(workbook.is_file())
        self.assertTrue(report.is_file())
        return pd.read_excel(workbook), json.loads(report.read_text()), output.getvalue()

    def test_mixed_valid_and_missing_channel_keeps_numeric_cells_blank(self):
        good = self.make_sample("good")
        missing = self.make_sample("missing_ch3", missing={(3, "mask")})

        table, report, output = self.run_analysis()
        rows = table.set_index("project_name")
        self.assertEqual(set(rows.index), {"good", "missing_ch3"})
        self.assertEqual(rows.loc["good", "analysis_status"], "analyzed")
        self.assertEqual(rows.loc["missing_ch3", "analysis_status"],
                         "skipped_missing_channel")
        self.assertIn("3", str(rows.loc["missing_ch3", "missing_channels"]))
        self.assertTrue(pd.isna(rows.loc["missing_ch3", "channel_1_total_count"]))
        self.assertTrue(pd.isna(rows.loc["missing_ch3", "channel_3_total_count"]))
        self.assertEqual(report["total_samples"], 2)
        self.assertEqual(report["analyzed"], 1)
        self.assertEqual(report["skipped_missing_channel"], 1)
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["status"], "complete_with_warnings")
        self.assertIn("1 analyzed", output)
        self.assertIn("1 skipped", output)
        self.assertIn("0 errors", output)
        self.assertTrue((good / "analysis_results" / "good_summary.csv").is_file())
        self.assertFalse((missing / "analysis_results" / "missing_ch3_summary.csv").exists())

    def test_all_missing_channels_preserves_summary_schema(self):
        self.make_sample("missing_mask", missing={(3, "mask")})
        self.make_sample("missing_raw", missing={(1, "raw")})

        table, report, output = self.run_analysis()
        self.assertEqual(set(table["analysis_status"]), {"skipped_missing_channel"})
        for column in ("channel_1_total_count", "channel_3_total_count",
                       "channel_1_percent_colocalized_whole_image",
                       "channel_3_percent_colocalized_whole_image"):
            self.assertIn(column, table.columns)
            self.assertTrue(table[column].isna().all(), column)
        self.assertEqual(report["total_samples"], 2)
        self.assertEqual(report["analyzed"], 0)
        self.assertEqual(report["skipped_missing_channel"], 2)
        self.assertEqual(report["errors"], 0)
        self.assertIn("2 skipped", output)

    def test_skipped_sample_does_not_change_legacy_validation_gate(self):
        good = self.make_sample("good")
        self.make_sample("missing_ch3", missing={(3, "mask")})
        val_dir = good / "ROI Validation Results"
        val_dir.mkdir()
        (val_dir / "roi_verification_results_ch1.tsv").write_text(
            "ROI_ID\tDecision\tNote\n1\tARTIFACT\t\n"
        )
        (val_dir / "_validation_complete_ch1").write_text("done")

        table, _, _ = self.run_analysis()
        good_row = table.set_index("project_name").loc["good"]
        self.assertEqual(good_row["channel_1_total_count"], 1)
        self.assertEqual(good_row["channel_1_validation_status"], "partial_skipped")

    def test_other_sample_error_is_logged_with_traceback_and_path(self):
        self.make_sample("good")
        broken = self.make_sample("bad_shape", shape_mismatch=True)

        table, report, output = self.run_analysis()
        rows = table.set_index("project_name")
        self.assertEqual(rows.loc["good", "analysis_status"], "analyzed")
        self.assertEqual(rows.loc["bad_shape", "analysis_status"], "error")
        self.assertEqual(report["analyzed"], 1)
        self.assertEqual(report["skipped_missing_channel"], 0)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["status"], "complete_with_warnings")
        self.assertIn("1 error", output)
        log_path = self.out_root / "analysis_errors.log"
        self.assertTrue(log_path.is_file())
        log = log_path.read_text()
        self.assertIn(str(broken), log)
        self.assertIn("ValueError", log)
        self.assertIn("Traceback", log)
        self.assertEqual(Path(report["report_path"]), log_path)

    def test_existing_config_keys_still_run_without_migration(self):
        self.make_sample("legacy_sample")
        config_path = self.root / "legacy_config.json"
        config_path.write_text(json.dumps({
            "ana_microsam_output_root": str(self.seg_root),
            "output_root_folder": str(self.out_root),
            "ana_red_channel": "1",
            "ana_pv_channel": "3",
            "ana_save_qc": False,
        }))

        analysis._apply_config(str(config_path))
        table, report, _ = self.run_analysis()
        self.assertEqual(table.loc[0, "project_name"], "legacy_sample")
        self.assertEqual(table.loc[0, "analysis_status"], "analyzed")
        self.assertEqual(report["analyzed"], 1)
        self.assertEqual(report["skipped_missing_channel"], 0)
        self.assertEqual(report["errors"], 0)

    def test_legacy_project_files_remain_unchanged_and_validation_applies(self):
        sample = self.make_sample("legacy")
        val_dir = sample / "ROI Validation Results"
        val_dir.mkdir()
        tsv = val_dir / "roi_verification_results.tsv"
        tsv.write_text("ROI_ID\tDecision\tNote\n1\tARTIFACT\told decision\n")
        marker = val_dir / "_validation_complete"
        marker.write_text("done")
        source_files = [*sample.glob("*.tif"), tsv, marker]
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        }

        table, report, _ = self.run_analysis()

        self.assertEqual(table.loc[0, "analysis_status"], "analyzed")
        self.assertEqual(table.loc[0, "channel_1_total_count"], 0)
        self.assertEqual(table.loc[0, "channel_3_total_count"], 0)
        self.assertEqual(report["segmentation_input"], str(self.seg_root))
        self.assertEqual(before, {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        })

    def test_changed_validation_is_excluded_until_focused_review_finishes(self):
        sample = self.make_sample("edited")
        mask = next(sample.glob("*_predict_original_ch1_*.tif"))
        old = tifffile.imread(mask)
        changed = old.copy()
        changed[2, 2] = 2  # new cell changes the previously validated cell
        val_dir = sample / "ROI Validation Results"
        val_dir.mkdir()
        (val_dir / "roi_verification_results_ch1.tsv").write_text(
            "ROI_ID\tDecision\tNote\n1\tARTIFACT\told\n"
        )
        (val_dir / "_validation_complete_ch1").write_text("done")
        save_edited_mask(str(mask), 1, old, changed)

        before_review, _, _ = self.run_analysis()
        self.assertEqual(before_review.loc[0, "channel_1_total_count"], 2)
        self.assertEqual(before_review.loc[0, "channel_1_validation_status"],
                         "partial_skipped")

        with patch.object(analysis, "APPLY_PARTIAL_VALIDATION", True):
            partial, _, _ = self.run_analysis()
        self.assertEqual(partial.loc[0, "channel_1_total_count"], 2)

        save_review_decision(str(sample), 1, str(mask), 1, "ARTIFACT", "rechecked")
        save_review_decision(str(sample), 1, str(mask), 2, "YES", "new cell")
        after_review, _, _ = self.run_analysis()
        self.assertEqual(after_review.loc[0, "channel_1_total_count"], 1)
        self.assertEqual(after_review.loc[0, "channel_1_validation_status"],
                         "complete")

    def test_workbook_write_failure_is_reported_separately(self):
        self.make_sample("good")
        output = io.StringIO()
        with patch.object(analysis.pd, "ExcelWriter", side_effect=OSError("disk full")):
            with contextlib.redirect_stdout(output):
                analysis.main()

        report = json.loads(
            (self.out_root / "analysis_run_summary.json").read_text()
        )
        log = (self.out_root / "analysis_errors.log").read_text()
        self.assertEqual(report["analyzed"], 1)
        self.assertEqual(report["errors"], 0)
        self.assertFalse(report["workbook_written"])
        self.assertEqual(report["status"], "complete_with_warnings")
        self.assertIn("disk full", report["output_error"])
        self.assertIn("disk full", log)
        self.assertIn("Traceback", log)
        self.assertIn("analysis.py", log)
        self.assertIn("disk full", output.getvalue())

    def test_empty_segmentation_input_gets_explicit_report(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analysis.main()

        self.assertFalse(
            (self.out_root / "combined_analysis_summary.xlsx").exists()
        )
        report = json.loads(
            (self.out_root / "analysis_run_summary.json").read_text()
        )
        log = (self.out_root / "analysis_errors.log").read_text()
        self.assertEqual(report["total_samples"], 0)
        self.assertEqual(report["analyzed"], 0)
        self.assertEqual(report["status"], "complete_with_warnings")
        self.assertIn("No project folders found", report["output_error"])
        self.assertIn("No project folders found", log)
        self.assertIn("0 analyzed", output.getvalue())
        self.assertIn("report:", output.getvalue())

    def test_launcher_status_supports_new_and_old_analysis_outputs(self):
        old_status, old_color = CellMatchrLauncher._analysis_run_status(
            str(self.out_root)
        )
        self.assertEqual(old_status, "Analysis complete.")
        self.assertEqual(old_color, GREEN)

        (self.out_root / "analysis_run_summary.json").write_text(json.dumps({
            "analyzed": 4,
            "skipped_missing_channel": 2,
            "errors": 1,
            "output_error": "",
        }))
        status, color = CellMatchrLauncher._analysis_run_status(
            str(self.out_root)
        )
        self.assertIn("4 analyzed", status)
        self.assertIn("2 skipped", status)
        self.assertIn("1 error", status)
        self.assertEqual(color, YELLOW)


if __name__ == "__main__":
    unittest.main()
