"""Checks for multi-project training plans, previews, and saved reports."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr.retraining import (collect_source_examples, plan_examples,
                                  project_runs, _prepare_pair,
                                  microsam_architecture, select_training_labels,
                                  train_model)
from cellmatchr.training_gallery import composite_preview, project_preview_settings
from cellmatchr.training_report import (build_report, object_scores, save_history)


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def project(self, name):
        project = self.root / name
        run = project / "Segmentation_MicroSAM"
        run.mkdir(parents=True)
        (project / "cellmatchr_project.json").write_text(json.dumps({
            "project_name": name, "project_output_folder": "/old/path"}))
        return project, run

    def sample(self, run, name, channel=1, complete=True):
        folder = run / name
        folder.mkdir()
        raw = np.full((768, 768), 100, np.uint16)
        raw[80:180, 80:180] = 500
        raw[370:420, 370:420] = 300
        mask = np.zeros((768, 768), np.uint32)
        mask[90:145, 90:145] = 1
        mask[385:410, 385:410] = 2
        mask[650:710, 650:710] = 3
        tifffile.imwrite(folder / f"{name}_split_ch{channel}.tif", raw)
        tifffile.imwrite(folder / f"{name}_split_ch2.tif", raw // 2)
        tifffile.imwrite(folder / f"{name}_predict_original_ch{channel}_sam.tif", mask)
        review = folder / "ROI Validation Results"
        review.mkdir()
        (review / f"roi_verification_results_ch{channel}.tsv").write_text(
            "ROI_ID\tDecision\n1\tYES\n2\tARTIFACT\n3\tNO\n")
        if complete:
            (review / f"_validation_complete_ch{channel}").touch()
        (folder / f"_manual_deletions_ch{channel}.tsv").write_text(
            "label_id\tchannel\n9\t1\n")
        return folder

    def test_cell_label_selection_preserves_artifact_meaning(self):
        project, run = self.project("current")
        self.sample(run, "a")
        examples, _ = collect_source_examples([
            {"project": str(project), "seg_run": run.name, "channel": 1}])
        self.assertEqual(examples[0]["label_ids"], {"NO": [3], "YES": [1]})
        examples[0]["manual_added_ids"] = [3]
        all_cells, omitted = select_training_labels(examples, [])
        self.assertEqual((all_cells[0]["accepted_ids"], omitted), ([1, 3], []))
        no_cells, omitted = select_training_labels(examples, ["no"])
        self.assertEqual((no_cells[0]["accepted_ids"], omitted), ([3], []))
        self.assertEqual(no_cells[0]["manual_added_ids"], [3])
        self.assertEqual(no_cells[0]["rejected_ids"], [2])
        yes_cells, _ = select_training_labels(examples, ["YES"])
        self.assertEqual(yes_cells[0]["manual_added_ids"], [])
        with tempfile.TemporaryDirectory() as stage:
            _image, labels_path = _prepare_pair(no_cells[0], Path(stage), 0,
                                                "raw_percentile")
            staged = tifffile.imread(labels_path)
            self.assertEqual(set(np.unique(staged)), {0, 3})
        both, omitted = select_training_labels(examples, ["YES", "NO"])
        self.assertEqual((both[0]["accepted_ids"], omitted), ([1, 3], []))

    def test_unfinished_review_requires_explicit_completed_only_choice(self):
        project, run = self.project("current")
        self.sample(run, "done")
        self.sample(run, "unfinished", complete=False)
        sources = [{"project": str(project), "seg_run": run.name, "channel": 1}]
        examples, skipped, unfinished = collect_source_examples(
            sources, with_review_status=True)
        self.assertEqual([item["sample"] for item in examples], ["done"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(unfinished), 1)
        with self.assertRaisesRegex(ValueError, "Confirm training on completed images"):
            train_model(str(project), run.name, 1, "cellpose", 1,
                        validation_mode="training_image", sources=sources)

        def fake_trainer(train_pairs, _validation_pairs, output, *_args):
            staged = tifffile.imread(train_pairs[0][1])
            self.assertEqual(set(np.unique(staged)), {0, 3})
            model = output / "weights.pth"
            model.touch()
            return str(model)

        with patch.dict(os.environ, {"CELLMATCHR_MODEL_LIBRARY":
                                  str(self.root / "models")}), \
             patch.dict("cellmatchr.retraining.TRAINERS", {"cellpose": fake_trainer}):
            result = train_model(str(project), run.name, 1, "cellpose", 1,
                                 validation_mode="training_image", sources=sources,
                                 allow_completed_only=True, selected_labels=["NO"])
        self.assertEqual(result["selected_cell_labels"], ["NO"])
        self.assertEqual(result["training_samples"], ["done"])
        self.assertEqual(len(result["unfinished_review"]), 1)

    def test_imported_project_uses_project_file_folder_and_keeps_sources_unchanged(self):
        first, run_a = self.project("current")
        second, run_b = self.project("imported")
        self.sample(run_a, "a")
        imported = self.sample(run_b, "b")
        mask = next(imported.glob("*_predict_original_ch1*.tif"))
        before = mask.read_bytes()
        self.assertEqual(project_runs(str(second / "cellmatchr_project.json")),
                         [{"name": "Segmentation_MicroSAM", "channels": [1]}])
        examples, skipped = collect_source_examples([
            {"project": str(first), "seg_run": run_a.name, "channel": 1},
            {"project": str(second), "seg_run": run_b.name, "channel": 1},
        ])
        self.assertEqual(skipped, [])
        self.assertEqual(len(examples), 2)
        self.assertNotEqual(examples[0]["key"], examples[1]["key"])
        self.assertEqual(examples[1]["project"], str(second.resolve()))
        self.assertEqual((examples[1]["original_count"],
                          examples[1]["incorrect_count"],
                          examples[1]["final_count"]), (4, 2, 2))
        self.assertEqual(mask.read_bytes(), before)

    def test_three_validation_choices_are_explicit(self):
        project, run = self.project("current")
        self.sample(run, "a")
        self.sample(run, "b")
        examples, _ = collect_source_examples([
            {"project": str(project), "seg_run": run.name, "channel": 1}])
        with self.assertRaisesRegex(ValueError, "Choose a validation approach"):
            plan_examples(examples, "")
        train, val, plan = plan_examples(examples, "whole_image", examples[1]["key"])
        self.assertEqual([entry["key"] for entry in train], [examples[0]["key"]])
        self.assertEqual(val[0]["key"], examples[1]["key"])
        self.assertEqual(plan["mode"], "whole_image")
        train, val, plan = plan_examples(examples[:1], "training_image")
        self.assertEqual(train[0]["key"], val[0]["key"])
        self.assertFalse(plan["independent"])

    def test_patch_holdout_never_reuses_its_pixels_for_training(self):
        project, run = self.project("current")
        self.sample(run, "a")
        examples, _ = collect_source_examples([
            {"project": str(project), "seg_run": run.name, "channel": 1}])
        with patch("cellmatchr.retraining.secrets.randbits", return_value=123456):
            train, val, plan = plan_examples(
                examples, "patch", patch_target_incorrect=1)
        self.assertEqual(plan["random_seed"], 123456)
        self.assertTrue(train)
        y0, y1, x0, x1 = plan["crop"]
        for entry in train:
            a, b, c, d = entry["crop"]
            self.assertFalse(a < y1 and b > y0 and c < x1 and d > x0)
        self.assertEqual(val[0]["crop"], tuple(plan["crop"]))
        with tempfile.TemporaryDirectory() as stage:
            image, labels = _prepare_pair(val[0], Path(stage), 0,
                                          "raw_percentile")
            self.assertEqual(tifffile.imread(image).shape, (y1-y0, x1-x0))
            self.assertEqual(tifffile.imread(labels).shape, (y1-y0, x1-x0))

    def test_gallery_and_report_are_saved_without_changing_reviewed_data(self):
        project, run = self.project("current")
        folder = self.sample(run, "a")
        examples, _ = collect_source_examples([
            {"project": str(project), "seg_run": run.name, "channel": 1}])
        preview = composite_preview(str(folder), (360, 240),
                                    {1: "red", 2: "purple"})
        self.assertLessEqual(preview.width, 360)
        self.assertLessEqual(preview.height, 240)
        self.assertEqual(preview.mode, "RGB")
        output = self.root / "library_model"
        output.mkdir()
        model = output / "weights.pth"
        model.touch()
        save_history(output, {"train_loss": [2.0, 1.0],
                              "val_loss": [2.2, 1.4]})
        result = {"backend": "microsam", "model_path": str(model),
                  "name": "Trial", "validation_plan": {"mode": "whole_image"},
                  "training_sources": [], "training_samples": [],
                  "excluded_keys": [], "epochs": 2, "advanced": {}}
        raw_mask = next(folder.glob("*_predict_original_ch1*.tif"))
        before = raw_mask.read_bytes()
        labels = tifffile.imread(raw_mask)
        with patch("cellmatchr.training_report._predict_mask",
                   side_effect=[np.zeros_like(labels), labels]):
            report_path = build_report(project, output, result, examples,
                                       "vit_b_lm", "raw_percentile")
        report = json.loads(report_path.read_text())
        self.assertEqual(report["validation"][0]["before"]["missed_cell"], 2)
        self.assertEqual(report["validation"][0]["after"]["missed_cell"], 0)
        self.assertTrue((report_path.parent / "learning_curve.png").is_file())
        self.assertTrue((report_path.parent / "image_1_reviewed.png").is_file())
        self.assertTrue((report_path.parent / "image_1_after.png").is_file())
        self.assertEqual(raw_mask.read_bytes(), before)

    def test_imported_preview_uses_its_own_colors_and_brightfield_mode(self):
        project, run = self.project("imported")
        folder = self.sample(run, "a", channel=0)
        next(folder.glob("*_split_ch2.tif")).unlink()
        project_file = project / "cellmatchr_project.json"
        project_file.write_text(json.dumps({
            "input_type": "vsi", "slide_signal_type": "brightfield",
            "val_channel_colors": "red,green"}))
        colors, brightfield = project_preview_settings(str(project))
        self.assertEqual(colors[0], "red")
        self.assertTrue(brightfield)
        active_colors, active_brightfield = project_preview_settings(
            str(project), str(project), {"val_channel_colors": "blue",
                                         "slide_signal_type": "fluorescence"})
        self.assertEqual(active_colors[0], "blue")
        self.assertFalse(active_brightfield)
        grayscale = np.asarray(composite_preview(str(folder), brightfield=True))
        np.testing.assert_array_equal(grayscale[..., 0], grayscale[..., 1])

    def test_object_counts_use_one_to_one_matching(self):
        true = np.array([[1, 1, 0, 2, 2], [1, 1, 0, 2, 2]])
        predicted = np.array([[8, 8, 0, 9, 9], [8, 8, 0, 9, 9]])
        self.assertEqual(object_scores(predicted, true)["true_positive"], 2)
        predicted[:, 3:] = 0
        self.assertEqual(object_scores(predicted, true)["missed_cell"], 1)

    def test_microsam_saved_model_keeps_its_backbone(self):
        folder = self.root / "trained_model"
        folder.mkdir()
        checkpoint = folder / "model.pth"
        checkpoint.touch()
        (folder / "model.json").write_text(json.dumps({
            "backend": "microsam", "microsam_architecture": "vit_l"}))
        self.assertEqual(microsam_architecture("vit_h_lm"), "vit_h")
        self.assertEqual(microsam_architecture(str(checkpoint)), "vit_l")
        (folder / "model.json").write_text(json.dumps({
            "backend": "microsam", "starting_model_name": "vit_h_lm"}))
        self.assertEqual(microsam_architecture(str(checkpoint)), "vit_h")

    def test_training_aggregates_projects_excludes_image_and_reports_to_current_project(self):
        current, current_run = self.project("current")
        imported, imported_run = self.project("imported")
        self.sample(current_run, "a")
        self.sample(current_run, "b")
        self.sample(imported_run, "exclude_me")
        self.sample(imported_run, "holdout")
        sources = [
            {"project": str(current), "seg_run": current_run.name, "channel": 1},
            {"project": str(imported), "seg_run": imported_run.name, "channel": 1},
        ]
        examples, skipped = collect_source_examples(sources)
        self.assertEqual(skipped, [])
        keys = {item["sample"]: item["key"] for item in examples}
        holdout_mask = tifffile.imread(next(
            (imported_run / "holdout").glob("*_predict_original_ch1*.tif")))

        def fake_trainer(_train, _val, output, _epochs, _base, _advanced):
            path = output / "model.pth"
            path.touch()
            save_history(output, {"train_loss": [1.0], "val_loss": [1.1]})
            return str(path)

        with patch.dict(os.environ, {"CELLMATCHR_MODEL_LIBRARY":
                                  str(self.root / "library")}), \
             patch.dict("cellmatchr.retraining.TRAINERS", {"cellpose": fake_trainer}), \
             patch("cellmatchr.training_report._predict_mask",
                   side_effect=[np.zeros_like(holdout_mask), holdout_mask]):
            result = train_model(
                str(current), current_run.name, 1, "cellpose", 1, "cpsam",
                "raw_percentile", "Multiple projects", sources=sources,
                excluded_keys={keys["exclude_me"]},
                validation_mode="whole_image", validation_key=keys["holdout"],
                generate_report=True)

        self.assertEqual(result["training_samples"], ["a", "b"])
        self.assertEqual({item["project"] for item in result["training_images"]},
                         {str(current.resolve())})
        self.assertEqual(result["validation_samples"], ["holdout"])
        self.assertEqual(result["excluded_keys"], [keys["exclude_me"]])
        self.assertTrue(Path(result["model_path"]).is_file())
        self.assertTrue(Path(result["report_path"]).resolve().is_relative_to(
            current.resolve()))
        self.assertFalse((imported / "Training_Reports").exists())
        report = json.loads(Path(result["report_path"]).read_text())
        self.assertEqual(report["validation"][0]["sample"], "holdout")


if __name__ == "__main__":
    unittest.main()
