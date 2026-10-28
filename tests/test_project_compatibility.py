"""Compatibility checks for legacy projects and separate analysis runs."""

import ast
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from cellmatchr import analysis
from cellmatchr.launcher import CellMatchrLauncher
from cellmatchr.validation_state import (
    finalize_if_ready, get_change_state, get_validation_tsv_path, save_edited_mask,
    save_review_decision, sidecar_path,
)


class _Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ProjectCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    @staticmethod
    def _digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _rows(path):
        with open(path, newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source, delimiter="\t"))

    def _legacy_sample(self, folder, *, reviewed=True):
        sample = Path(folder)
        sample.mkdir(parents=True)
        mask = sample / "sample_predict_original_ch1_local_bg_subtract_tile2048.tif"
        old = np.zeros((8, 8), dtype=np.uint16)
        old[1:3, 1:3] = 1
        old[5:7, 5:7] = 3
        tifffile.imwrite(mask, old)
        val_dir = sample / "ROI Validation Results"
        if reviewed:
            val_dir.mkdir()
            (val_dir / "roi_verification_results_ch1.tsv").write_text(
                "ROI_ID\tDecision\tNote\n"
                "1\tARTIFACT\told decision\n"
                "3\tYES\tkeep this decision\n"
            )
            (val_dir / "_validation_complete_ch1").write_text("done")
        return sample, mask, old, val_dir

    @staticmethod
    def _edited(old):
        edited = old.copy()
        edited[2:4, 2:4] = 4  # adds a cell and changes one pixel of old ID 1
        return edited

    def test_no_sidecar_legacy_project_is_read_without_source_changes(self):
        sample, mask, _, val_dir = self._legacy_sample(self.root / "old")
        tsv = val_dir / "roi_verification_results_ch1.tsv"
        marker = val_dir / "_validation_complete_ch1"
        before = {path: self._digest(path) for path in (mask, tsv, marker)}

        state = get_change_state(str(sample), 1, str(mask))
        self.assertEqual(state["status"], "legacy")
        self.assertTrue(analysis.is_validation_complete(str(sample), 1))
        self.assertEqual(analysis.read_validation_tsv(str(sample), 1)["artifact_labels"],
                         {1})
        self.assertEqual(get_validation_tsv_path(str(sample), 1), str(tsv))
        self.assertFalse(Path(sidecar_path(str(sample), 1)).exists())
        self.assertEqual(before,
                         {path: self._digest(path) for path in (mask, tsv, marker)})

    def test_changed_old_and_new_labels_get_focused_review(self):
        sample, mask, old, val_dir = self._legacy_sample(self.root / "edited")
        original_tsv = (val_dir / "roi_verification_results_ch1.tsv").read_bytes()
        state = save_edited_mask(str(mask), 1, old, self._edited(old))

        self.assertEqual(state["status"], "pending")
        self.assertEqual(set(state["pending_ids"]), {1, 4})
        self.assertFalse((val_dir / "_validation_complete_ch1").exists())
        self.assertEqual(self._rows(val_dir / "roi_verification_results_ch1.tsv"), [])
        effective = self._rows(get_validation_tsv_path(str(sample), 1))
        self.assertEqual([(row["ROI_ID"], row["Decision"]) for row in effective],
                         [("3", "YES")])
        history = sample / ".validation_history"
        self.assertTrue(any(
            path.read_bytes() == original_tsv
            for path in history.glob("*/roi_verification_results_ch1.tsv")
        ))
        self.assertFalse(analysis.is_validation_complete(str(sample), 1))

        state = save_review_decision(str(sample), 1, str(mask), 1, "NO", "changed")
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["pending_ids"], [4])
        state = save_review_decision(str(sample), 1, str(mask), 4, "YES", "new")
        self.assertEqual(state["status"], "current")
        self.assertTrue(analysis.is_validation_complete(str(sample), 1))
        final_rows = self._rows(val_dir / "roi_verification_results_ch1.tsv")
        self.assertEqual({row["ROI_ID"]: row["Decision"] for row in final_rows},
                         {"1": "NO", "3": "YES", "4": "YES"})

    def test_unvalidated_old_sample_does_not_gain_revalidation_requirement(self):
        sample, mask, old, val_dir = self._legacy_sample(self.root / "unreviewed",
                                                       reviewed=False)
        save_edited_mask(str(mask), 1, old, self._edited(old))
        self.assertEqual(get_change_state(str(sample), 1, str(mask))["status"],
                         "legacy")
        self.assertFalse(Path(sidecar_path(str(sample), 1)).exists())
        self.assertFalse(val_dir.exists())

    def test_global_validation_gate_uses_the_mask_analysis_will_select(self):
        segmentation = self.root / "mixed_masks" / "Segmentation"
        output = self.root / "mixed_masks" / "Analysis"
        segmentation.mkdir(parents=True)
        output.mkdir()
        for name in ("ambiguous", "good"):
            sample, mask, old, val_dir = self._legacy_sample(
                segmentation / name
            )
            mask.rename(sample / f"{name}_predict_original_ch1_a.tif")
            mask_a = sample / f"{name}_predict_original_ch1_a.tif"
            tifffile.imwrite(sample / f"{name}_split_ch1.tif",
                             np.full((8, 8), 100, dtype=np.uint16))
            tifffile.imwrite(sample / f"{name}_split_ch3.tif",
                             np.full((8, 8), 90, dtype=np.uint16))
            tifffile.imwrite(sample / f"{name}_predict_original_ch3_a.tif", old)
            if name == "ambiguous":
                save_edited_mask(str(mask_a), 1, old, old)
                self.assertEqual(finalize_if_ready(
                    str(sample), 1, str(mask_a))["status"], "current")
                tifffile.imwrite(
                    sample / f"{name}_predict_original_ch1_z.tif", old
                )
                self.assertFalse(analysis.is_validation_complete(str(sample), 1))
            else:
                self.assertTrue((val_dir / "_validation_complete_ch1").exists())

        settings = {
            "MICROSAM_OUTPUT_ROOT": str(segmentation),
            "ANALYSIS_OUTPUT_ROOT": str(output),
            "COI_CHANNEL": 1,
            "PM_CHANNEL": 3,
            "ROI_MASK_FOLDER": "",
            "SAVE_QC_PNG": False,
            "ANALYSIS_SUBFOLDER_NAME": "analysis_results",
            "APPLY_PARTIAL_VALIDATION": False,
        }
        with contextlib.ExitStack() as stack:
            for key, value in settings.items():
                stack.enter_context(patch.object(analysis, key, value))
            with contextlib.redirect_stdout(io.StringIO()):
                analysis.main()
            self.assertFalse(analysis.CHANNEL_FULLY_VALIDATED[1])
        table = pd.read_excel(output / "combined_analysis_summary.xlsx")
        good = table.set_index("project_name").loc["good"]
        ambiguous = table.set_index("project_name").loc["ambiguous"]
        self.assertEqual(good["channel_1_total_count"], 2)
        self.assertEqual(good["channel_1_validation_status"], "partial_skipped")
        self.assertEqual(ambiguous["channel_1_validation_status"],
                         "validation_review_required")

    def test_pending_review_survives_project_move(self):
        sample, mask, old, _ = self._legacy_sample(self.root / "before_move")
        save_edited_mask(str(mask), 1, old, self._edited(old))
        moved = self.root / "after_move"
        shutil.move(str(sample), moved)
        moved_mask = moved / mask.name

        self.assertEqual(get_change_state(str(moved), 1, str(moved_mask))["status"],
                         "pending")
        save_review_decision(str(moved), 1, str(moved_mask), 1, "NO", "rechecked")
        state = save_review_decision(str(moved), 1, str(moved_mask), 4, "YES", "new")
        self.assertEqual(state["status"], "current")
        final_rows = self._rows(
            moved / "ROI Validation Results" / "roi_verification_results_ch1.tsv"
        )
        self.assertEqual({row["ROI_ID"]: row["Decision"] for row in final_rows},
                         {"1": "NO", "3": "YES", "4": "YES"})

    def test_earlier_analysis_reader_does_not_accept_pending_as_complete(self):
        if not shutil.which("git"):
            self.skipTest("Git history unavailable")
        repo = Path(__file__).resolve().parents[1]
        source = subprocess.run(
            ["git", "show", "bbeb583:cellmatchr/analysis.py"],
            cwd=repo, text=True, capture_output=True,
        )
        if source.returncode != 0:
            self.skipTest("Earlier analysis revision unavailable")
        module = ast.parse(source.stdout)
        old_function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "is_validation_complete"
        )
        namespace = {"os": os}
        exec(compile(ast.Module(body=[old_function], type_ignores=[]),
                     "prior_analysis.py", "exec"), namespace)
        old_reader = namespace["is_validation_complete"]

        sample, mask, old, _ = self._legacy_sample(self.root / "prior_reader")
        self.assertTrue(old_reader(str(sample), 1))
        save_edited_mask(str(mask), 1, old, self._edited(old))
        self.assertFalse(old_reader(str(sample), 1))
        save_review_decision(str(sample), 1, str(mask), 1, "NO", "rechecked")
        save_review_decision(str(sample), 1, str(mask), 4, "YES", "new")
        self.assertTrue(old_reader(str(sample), 1))

    def test_run_choice_and_replacement_are_scoped_and_recoverable(self):
        project = self.root / "project"
        sample = project / "Segmentation" / "sample" / "analysis_results"
        sample.mkdir(parents=True)
        (project / "Segmentation_other").mkdir()
        old_analysis = project / "Analysis"
        old_analysis.mkdir()
        other_analysis = project / "Analysis_other"
        other_analysis.mkdir()
        workbook = old_analysis / "combined_analysis_summary.xlsx"
        summary = sample / "sample_summary.csv"
        other_workbook = other_analysis / "combined_analysis_summary.xlsx"
        workbook.write_bytes(b"old workbook")
        summary.write_bytes(b"old sample summary")
        other_workbook.write_bytes(b"other run")
        before = {path: self._digest(path)
                  for path in (workbook, summary, other_workbook)}

        launcher = object.__new__(CellMatchrLauncher)
        launcher.config = {"project_output_folder": str(project),
                           "seg_input_subfolder": ""}
        launcher._ana_input_var = _Variable()
        launcher.root = None
        with patch("cellmatchr.launcher._SegmentationRunDialog",
                   return_value=SimpleNamespace(result="Segmentation_other")):
            chosen = launcher._choose_analysis_run(ask_if_multiple=True)
        self.assertEqual(chosen, "Segmentation_other")
        self.assertEqual(launcher._analysis_output_dir(chosen), str(other_analysis))

        with patch("cellmatchr.launcher.messagebox.askyesno", return_value=False):
            self.assertIsNone(launcher._prepare_analysis_output("Segmentation"))
        self.assertEqual(before,
                         {path: self._digest(path)
                          for path in (workbook, summary, other_workbook)})

        with patch("cellmatchr.launcher.messagebox.askyesno", return_value=True):
            prepared = launcher._prepare_analysis_output("Segmentation")
        self.assertIsNotNone(prepared)
        output_dir, backup_dir = prepared
        self.assertEqual(output_dir, str(old_analysis))
        self.assertFalse(workbook.exists())
        self.assertFalse(summary.exists())
        self.assertEqual(self._digest(other_workbook), before[other_workbook])
        self.assertTrue(Path(backup_dir, "backup_manifest.json").is_file())
        workbook.write_bytes(b"failed new workbook")
        summary.write_bytes(b"failed new summary")
        launcher._restore_analysis_backup(backup_dir)
        self.assertEqual(before,
                         {path: self._digest(path)
                          for path in (workbook, summary, other_workbook)})

    def test_replacement_backs_up_every_actual_analysis_output(self):
        project = self.root / "actual_outputs"
        segmentation = project / "Segmentation"
        output = project / "Analysis"
        sample, _, old, _ = self._legacy_sample(segmentation / "sample")
        tifffile.imwrite(sample / "sample_split_ch1.tif",
                         np.full((8, 8), 100, dtype=np.uint16))
        tifffile.imwrite(sample / "sample_split_ch3.tif",
                         np.full((8, 8), 90, dtype=np.uint16))
        tifffile.imwrite(sample / "sample_predict_original_ch3_a.tif", old)
        roi_folder = project / "roi"
        roi_folder.mkdir()
        tifffile.imwrite(roi_folder / "sample_mask_overview.tif",
                         np.ones((8, 8), dtype=np.uint8))
        settings = {
            "MICROSAM_OUTPUT_ROOT": str(segmentation),
            "ANALYSIS_OUTPUT_ROOT": str(output),
            "COI_CHANNEL": 1,
            "PM_CHANNEL": 3,
            "ROI_MASK_FOLDER": str(roi_folder),
            "SAVE_QC_PNG": True,
            "ANALYSIS_SUBFOLDER_NAME": "analysis_results",
            "APPLY_PARTIAL_VALIDATION": False,
        }
        with contextlib.ExitStack() as stack:
            for key, value in settings.items():
                stack.enter_context(patch.object(analysis, key, value))
            stack.enter_context(patch.object(
                analysis, "make_qc_overlay",
                side_effect=lambda **kw: Path(kw["output_path"]).write_bytes(b"qc")))
            stack.enter_context(patch.object(
                analysis, "make_roi_mask_preview",
                side_effect=lambda **kw: Path(kw["output_path"]).write_bytes(b"qc")))
            with contextlib.redirect_stdout(io.StringIO()):
                analysis.main()
        sample_results = sample / "analysis_results"
        actual = sorted(output.iterdir()) + sorted(sample_results.iterdir())
        expected_names = {
            "sample_summary.csv",
            "sample_channel_1_object_table.csv",
            "sample_channel_3_object_table.csv",
            "sample_channel_1_to_channel_3_matches.csv",
            "sample_channel_3_to_channel_1_matches.csv",
            "sample_channel_1_to_channel_3_iou_candidates.csv",
            "sample_channel_3_to_channel_1_iou_candidates.csv",
            "sample_QC_overlay.tif",
            "sample_QC_roi_preview.tif",
            "sample_roi_mask_resized.tif",
        }
        self.assertTrue(expected_names.issubset({path.name for path in actual}))

        launcher = object.__new__(CellMatchrLauncher)
        launcher.config = {"project_output_folder": str(project),
                           "ana_subfolder_name": "analysis_results"}
        launcher.root = None
        selected = launcher._analysis_generated_files("Segmentation", str(output))
        self.assertEqual(set(selected), {str(path) for path in actual})
        before = {path: self._digest(path) for path in actual}
        with patch("cellmatchr.launcher.messagebox.askyesno", return_value=True):
            _, backup_dir = launcher._prepare_analysis_output("Segmentation")
        self.assertTrue(all(not path.exists() for path in actual))
        for path in actual:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"replacement")
        launcher._restore_analysis_backup(backup_dir)
        self.assertEqual(before, {path: self._digest(path) for path in actual})

    def test_existing_segmentation_masks_require_a_distinct_new_run(self):
        project = self.root / "segmentation_project"
        old_sample = project / "Segmentation" / "sample"
        old_sample.mkdir(parents=True)
        old_mask = old_sample / "sample_predict_original_ch1_old_tile2048.tif"
        old_mask.write_bytes(b"previous segmentation")
        old_hash = self._digest(old_mask)

        launcher = object.__new__(CellMatchrLauncher)
        launcher.config = {"project_output_folder": str(project),
                           "seg_backend": "cellpose"}
        launcher.root = None
        with patch("cellmatchr.launcher._ConflictDialog",
                   return_value=SimpleNamespace(result="cellpose_new")) as dialog:
            new_output = launcher._resolve_segmentation_output_dir()
        self.assertEqual(new_output,
                         str(project / "Segmentation_cellpose_new"))
        self.assertFalse(dialog.call_args.kwargs["allow_replace"])
        self.assertEqual(self._digest(old_mask), old_hash)

        # Check the direct script's guard without importing model libraries.
        source_path = Path(__file__).resolve().parents[1] / "cellmatchr" / "segmentation.py"
        syntax = ast.parse(source_path.read_text())
        guard = next(node for node in syntax.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "_check_new_run_output")
        namespace = {"os": os, "RESUME": False}
        exec(compile(ast.Module(body=[guard], type_ignores=[]),
                     str(source_path), "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "different folder"):
            namespace["_check_new_run_output"](str(project / "Segmentation"))
        self.assertEqual(self._digest(old_mask), old_hash)

    def test_interrupted_analysis_replacement_recovers_only_selected_run(self):
        for stage in ("before_run", "partial_new", "new_complete"):
            with self.subTest(stage=stage):
                project = self.root / stage
                sample = project / "Segmentation" / "sample" / "analysis_results"
                sample.mkdir(parents=True)
                (project / "Segmentation_other").mkdir()
                old_analysis = project / "Analysis"
                old_analysis.mkdir()
                other_analysis = project / "Analysis_other"
                other_analysis.mkdir()
                workbook = old_analysis / "combined_analysis_summary.xlsx"
                run_summary = old_analysis / "analysis_run_summary.json"
                sample_csv = sample / "sample_summary.csv"
                other_workbook = other_analysis / "combined_analysis_summary.xlsx"
                workbook.write_bytes(b"old workbook")
                run_summary.write_text(json.dumps({"workbook_written": True}))
                sample_csv.write_bytes(b"old sample")
                other_workbook.write_bytes(b"untouched other run")
                old = {path: self._digest(path)
                       for path in (workbook, run_summary, sample_csv,
                                    other_workbook)}

                launcher = object.__new__(CellMatchrLauncher)
                launcher.config = {"project_output_folder": str(project)}
                launcher.root = None
                with patch("cellmatchr.launcher.messagebox.askyesno",
                           return_value=True):
                    output_dir, backup_dir = launcher._prepare_analysis_output(
                        "Segmentation"
                    )
                self.assertEqual(output_dir, str(old_analysis))
                self.assertTrue(Path(backup_dir, "backup_manifest.json").is_file())
                if stage in ("partial_new", "new_complete"):
                    workbook.write_bytes(b"new workbook")
                    sample_csv.write_bytes(b"new sample")
                if stage == "new_complete":
                    run_summary.write_text(json.dumps({"workbook_written": True}))

                # The next launcher process recovers without the original
                # instance or any in-memory process state.
                reopened = object.__new__(CellMatchrLauncher)
                reopened.config = {"project_output_folder": str(project)}
                reopened.root = None
                reopened._recover_analysis_backups("Segmentation")
                manifest = json.loads(
                    Path(backup_dir, "backup_manifest.json").read_text()
                )
                if stage == "new_complete":
                    self.assertEqual(manifest["state"], "committed")
                    self.assertEqual(workbook.read_bytes(), b"new workbook")
                    self.assertEqual(sample_csv.read_bytes(), b"new sample")
                    self.assertEqual(
                        Path(backup_dir, "Analysis",
                             "combined_analysis_summary.xlsx").read_bytes(),
                        b"old workbook",
                    )
                else:
                    self.assertEqual(manifest["state"], "restored")
                    self.assertEqual(old,
                                     {path: self._digest(path) for path in old})
                self.assertEqual(self._digest(other_workbook),
                                 old[other_workbook])

if __name__ == "__main__":
    unittest.main()
