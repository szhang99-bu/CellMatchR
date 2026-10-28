"""Focused validation and old-project compatibility checks."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr import validation_state as state


class ValidationChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sample = Path(self.temp.name) / "sample"
        self.sample.mkdir()
        self.mask_path = self.sample / "sample_predict_original_ch1_method.tif"
        self.old = np.zeros((8, 8), dtype=np.uint16)
        self.old[1:4, 1:4] = 1
        self.old[5:7, 5:7] = 2
        tifffile.imwrite(self.mask_path, self.old)
        self.val_dir = self.sample / "ROI Validation Results"
        self.val_dir.mkdir()

    def write_validation(self, text=None, marker=True):
        if text is None:
            text = "ROI_ID\tDecision\tNote\n1\tARTIFACT\told\n2\tYES\tkeep\n"
        (self.val_dir / "roi_verification_results_ch1.tsv").write_text(text)
        if marker:
            (self.val_dir / "_validation_complete_ch1").write_text("done")

    def test_no_validation_keeps_legacy_behavior(self):
        changed = self.old.copy()
        changed[0, 0] = 3
        result = state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        self.assertEqual(result["status"], "legacy")
        self.assertFalse(Path(state.sidecar_path(str(self.sample), 1)).exists())
        np.testing.assert_array_equal(tifffile.imread(self.mask_path), changed)

    def test_edited_old_project_is_safe_for_old_readers_and_merges_decisions(self):
        self.write_validation()
        (self.val_dir / "user_notes.txt").write_text("keep me")
        (self.val_dir / "roi_verification_results_ch3.tsv").write_text(
            "ROI_ID\tDecision\tNote\n8\tYES\tother\n"
        )
        (self.val_dir / "_validation_complete_ch3").write_text("done")
        changed = self.old.copy()
        changed[1, 1] = 3  # new cell also changes old cell 1

        result = state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["pending_ids"], [1, 3])
        self.assertEqual(
            (self.val_dir / "roi_verification_results_ch1.tsv").read_text(),
            "ROI_ID\tDecision\tNote\n",
        )
        self.assertFalse((self.val_dir / "_validation_complete_ch1").exists())
        self.assertEqual((self.val_dir / "user_notes.txt").read_text(), "keep me")
        self.assertTrue((self.val_dir / "_validation_complete_ch3").exists())
        self.assertEqual(
            Path(state.get_validation_tsv_path(str(self.sample), 1)).read_text(),
            "ROI_ID\tDecision\tNote\n2\tYES\tkeep\n",
        )
        snapshots = list((self.sample / ".validation_history").iterdir())
        self.assertEqual(len(snapshots), 1)
        self.assertIn("ARTIFACT", (
            snapshots[0] / "roi_verification_results_ch1.tsv").read_text())

        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   1, "NO", "changed")
        self.assertEqual(
            (self.val_dir / "roi_verification_results_ch1.tsv").read_text(),
            "ROI_ID\tDecision\tNote\n",
        )
        self.assertFalse((self.val_dir / "_validation_complete_ch1").exists())
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   3, "YES", "added")
        self.assertEqual(
            state.get_change_state(str(self.sample), 1, str(self.mask_path))["status"],
            "current",
        )
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())
        final = (self.val_dir / "roi_verification_results_ch1.tsv").read_text()
        self.assertIn("1\tNO\tchanged", final)
        self.assertIn("2\tYES\tkeep", final)
        self.assertIn("3\tYES\tadded", final)
        self.assertNotIn("ARTIFACT", final)

    def test_partial_validation_queues_all_unreviewed_cells(self):
        self.write_validation(
            "ROI_ID\tDecision\tNote\n1\tYES\told\n", marker=False
        )
        changed = self.old.copy()
        changed[0, 0] = 3
        result = state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        self.assertEqual(result["pending_ids"], [2, 3])
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   2, "NO", "reviewed")
        self.assertFalse((self.val_dir / "_validation_complete_ch1").exists())
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   3, "YES", "new")
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())

    def test_unrelated_legacy_partial_tsv_remains_readable(self):
        self.write_validation()
        legacy = self.val_dir / "roi_verification_results.tsv"
        legacy.write_text("ROI_ID\tDecision\tNote\n8\tMAYBE\tother channel\n")
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        self.assertEqual(legacy.read_text(),
                         "ROI_ID\tDecision\tNote\n8\tMAYBE\tother channel\n")
        self.assertEqual(state.get_validation_tsv_path(str(self.sample), 3),
                         str(legacy))

    def test_legacy_global_validation_is_hidden_then_restored_when_safe(self):
        (self.val_dir / "roi_verification_results.tsv").write_text(
            " roi_id \t decision \t Note \n1.0\tYES\tlegacy\n2\tNO\tlegacy\n"
        )
        (self.val_dir / "_validation_complete").write_text("done")
        changed = self.old.copy()
        changed[0, 0] = 3
        result = state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        self.assertEqual(result["pending_ids"], [3])
        self.assertFalse((self.val_dir / "_validation_complete").exists())
        self.assertEqual((self.val_dir / "roi_verification_results.tsv").read_text(),
                         "ROI_ID\tDecision\tNote\n")
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   3, "YES", "new")
        self.assertTrue((self.val_dir / "_validation_complete").exists())
        final = (self.val_dir / "roi_verification_results.tsv").read_text()
        self.assertIn("1\tYES\tlegacy", final)
        self.assertIn("2\tNO\tlegacy", final)
        self.assertIn("3\tYES\tnew", final)

    def test_external_mask_change_requires_full_review(self):
        self.write_validation()
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   3, "YES", "new")
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())

        external = changed.copy()
        external[0, 1] = 4
        tifffile.imwrite(self.mask_path, external)
        unknown = state.get_change_state(str(self.sample), 1,
                                         str(self.mask_path))
        self.assertEqual(unknown["status"], "unknown")
        restarted = state.begin_review(str(self.sample), 1,
                                       str(self.mask_path))
        self.assertEqual(restarted["pending_ids"], [1, 2, 3, 4])
        self.assertFalse((self.val_dir / "_validation_complete_ch1").exists())
        self.assertEqual((self.val_dir / "roi_verification_results_ch1.tsv").read_text(),
                         "ROI_ID\tDecision\tNote\n")

    def test_partial_review_written_by_older_app_is_imported(self):
        self.write_validation()
        changed = self.old.copy()
        changed[1, 1] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        active = self.val_dir / "roi_verification_results_ch1.tsv"
        active.write_text("ROI_ID\tDecision\tNote\n1\tNO\tolder app\n")
        self.assertEqual(state.get_change_state(
            str(self.sample), 1, str(self.mask_path))["status"], "unknown")

        imported = state.begin_review(str(self.sample), 1,
                                      str(self.mask_path))
        self.assertEqual(imported["status"], "pending")
        self.assertEqual(imported["pending_ids"], [3])
        self.assertEqual(active.read_text(), "ROI_ID\tDecision\tNote\n")
        effective = Path(state.get_validation_tsv_path(str(self.sample), 1)).read_text()
        self.assertIn("1\tNO\tolder app", effective)
        self.assertIn("2\tYES\tkeep", effective)
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   3, "YES", "new app")
        self.assertIn("1\tNO\tolder app", active.read_text())

    def test_complete_review_written_by_older_app_is_adopted(self):
        self.write_validation()
        changed = self.old.copy()
        changed[1, 1] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        active = self.val_dir / "roi_verification_results_ch1.tsv"
        active.write_text(
            "ROI_ID\tDecision\tNote\n"
            "1\tNO\tolder\n2\tMAYBE\tolder\n3\tYES\tolder\n"
        )
        (self.val_dir / "_validation_complete_ch1").write_text("done")
        imported = state.begin_review(str(self.sample), 1,
                                      str(self.mask_path))
        self.assertEqual(imported["status"], "current")
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())
        self.assertIn("2\tMAYBE\tolder", active.read_text())

    def test_conflicting_older_app_decision_is_queued_for_user(self):
        self.write_validation()
        changed = self.old.copy()
        changed[1, 1] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   1, "YES", "new app")
        active = self.val_dir / "roi_verification_results_ch1.tsv"
        active.write_text("ROI_ID\tDecision\tNote\n1\tNO\tolder app\n")
        result = state.begin_review(str(self.sample), 1,
                                    str(self.mask_path))
        self.assertEqual(result["pending_ids"], [1, 3])
        self.assertFalse((self.val_dir / "_validation_complete_ch1").exists())
        self.assertNotIn("1\t", Path(
            state.get_validation_tsv_path(str(self.sample), 1)).read_text())

    def test_invalid_older_app_review_is_preserved_and_not_adopted(self):
        self.write_validation()
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        active = self.val_dir / "roi_verification_results_ch1.tsv"
        active.write_text("ROI_ID\tDecision\tNote\n99\tYES\tunknown\n")
        with self.assertRaisesRegex(RuntimeError, "absent from this mask"):
            state.begin_review(str(self.sample), 1, str(self.mask_path))
        self.assertIn("99\tYES\tunknown", active.read_text())
        self.assertEqual(state.get_change_state(
            str(self.sample), 1, str(self.mask_path))["status"], "unknown")

    def test_malformed_extra_older_app_row_is_not_silently_dropped(self):
        self.write_validation()
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        active = self.val_dir / "roi_verification_results_ch1.tsv"
        active.write_text(
            "ROI_ID\tDecision\tNote\n3\tYES\tvalid\nwrong\tNO\tbad\n"
        )
        with self.assertRaisesRegex(RuntimeError, "invalid ROI_ID"):
            state.begin_review(str(self.sample), 1, str(self.mask_path))
        self.assertIn("wrong\tNO\tbad", active.read_text())

    def test_legacy_project_imports_older_app_channel_specific_review(self):
        (self.val_dir / "roi_verification_results.tsv").write_text(
            "ROI_ID\tDecision\tNote\n1\tYES\told\n2\tNO\told\n"
        )
        (self.val_dir / "_validation_complete").write_text("done")
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        channel_file = self.val_dir / "roi_verification_results_ch1.tsv"
        channel_file.write_text("ROI_ID\tDecision\tNote\n3\tYES\tolder app\n")
        result = state.begin_review(str(self.sample), 1,
                                    str(self.mask_path))
        self.assertEqual(result["status"], "current")
        self.assertIn("3\tYES\tolder app", channel_file.read_text())
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())

    def test_ambiguous_global_older_app_review_is_not_assigned_a_channel(self):
        (self.val_dir / "roi_verification_results.tsv").write_text(
            "ROI_ID\tDecision\tNote\n1\tYES\told\n2\tNO\told\n"
        )
        (self.val_dir / "_validation_complete").write_text("done")
        other_mask = self.sample / "sample_predict_original_ch3_method.tif"
        tifffile.imwrite(other_mask, self.old)
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        global_file = self.val_dir / "roi_verification_results.tsv"
        global_file.write_text("ROI_ID\tDecision\tNote\n3\tYES\tolder app\n")
        with self.assertRaisesRegex(RuntimeError, "channel cannot be identified"):
            state.begin_review(str(self.sample), 1, str(self.mask_path))
        self.assertIn("3\tYES\tolder app", global_file.read_text())

    def test_multichannel_legacy_global_marker_stays_conservative_after_review(self):
        (self.val_dir / "roi_verification_results.tsv").write_text(
            "ROI_ID\tDecision\tNote\n1\tYES\told\n2\tNO\told\n"
        )
        (self.val_dir / "_validation_complete").write_text("done")
        tifffile.imwrite(
            self.sample / "sample_predict_original_ch3_method.tif", self.old
        )
        changed = self.old.copy()
        changed[0, 0] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        state.save_review_decision(str(self.sample), 1, str(self.mask_path),
                                   3, "YES", "new")
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())
        self.assertFalse((self.val_dir / "_validation_complete").exists())
        self.assertEqual((self.val_dir / "roi_verification_results.tsv").read_text(),
                         "ROI_ID\tDecision\tNote\n")
        self.assertIn("3\tYES\tnew", (
            self.val_dir / "roi_verification_results_ch1.tsv").read_text())

    def test_pending_project_can_be_moved(self):
        self.write_validation()
        changed = self.old.copy()
        changed[1, 1] = 3
        state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        moved = Path(self.temp.name) / "moved"
        shutil.move(str(self.sample), str(moved))
        moved_mask = moved / self.mask_path.name
        record = json.loads(Path(state.sidecar_path(str(moved), 1)).read_text())
        self.assertNotIn("base_tsv", record)
        self.assertFalse(os.path.isabs(record["base_tsv_name"]))
        self.assertIn("2\tYES\tkeep", Path(
            state.get_validation_tsv_path(str(moved), 1)).read_text())
        for roi_id in (1, 3):
            state.save_review_decision(str(moved), 1, str(moved_mask),
                                       roi_id, "YES", "moved")
        self.assertTrue((moved / "ROI Validation Results" /
                         "_validation_complete_ch1").exists())

    def test_mask_replace_failure_restores_original_validation_folder(self):
        self.write_validation()
        changed = self.old.copy()
        changed[0, 0] = 3
        real_replace = os.replace

        def fail_mask_replace(src, dst):
            if str(dst) == str(self.mask_path):
                raise OSError("cannot replace mask")
            return real_replace(src, dst)

        with patch.object(state.os, "replace", side_effect=fail_mask_replace):
            with self.assertRaisesRegex(OSError, "cannot replace mask"):
                state.save_edited_mask(str(self.mask_path), 1, self.old, changed)
        np.testing.assert_array_equal(tifffile.imread(self.mask_path), self.old)
        self.assertTrue((self.val_dir / "_validation_complete_ch1").exists())
        self.assertIn("ARTIFACT", (
            self.val_dir / "roi_verification_results_ch1.tsv").read_text())
        self.assertFalse(Path(state.sidecar_path(str(self.sample), 1)).exists())


if __name__ == "__main__":
    unittest.main()
