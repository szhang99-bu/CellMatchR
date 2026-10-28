"""Input listing and preview routing for Leica files in nested folders."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from cellmatchr.launcher import (
    CellMatchrLauncher,
    _first_confocal_preview_input,
    _has_analysis_roi_masks,
    _scanned_input_files,
)


class LifLauncherRoutingTests(TestCase):
    def test_listing_matches_root_and_one_subfolder_scan(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            direct = root / "direct.tif"
            child = root / "Leica confocal" / "sample.lif"
            deep = root / "Leica confocal" / "deeper" / "hidden.lif"
            child.parent.mkdir()
            deep.parent.mkdir()
            for path in (direct, child, deep):
                path.touch()

            exts = CellMatchrLauncher._INPUT_EXTS
            self.assertEqual(
                _scanned_input_files(str(root), exts),
                [str(child), str(direct)],
            )
            self.assertEqual(
                _scanned_input_files(str(root), exts, skip_subdirs=True),
                [str(direct)],
            )

    def test_preview_skips_slide_scanner_and_selects_confocal_scene(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            scanner = root / "A slide scanner" / "same.lif"
            confocal = root / "B confocal" / "same.lif"
            scanner.parent.mkdir()
            confocal.parent.mkdir()
            scanner.touch()
            confocal.touch()

            def classify(path):
                if path == str(scanner):
                    return {"file_type": "slide_scanner",
                            "scenes": [(0, "mosaic", "slide_scanner"),
                                       (1, "merged", "slide_scanner")]}
                return {"file_type": "mixed",
                        "scenes": [(0, "mosaic", "slide_scanner"),
                                   (1, "scan", "confocal")]}

            with patch("cellmatchr.lif_classify.classify_lif_file",
                       side_effect=classify):
                selected = _first_confocal_preview_input(
                    str(root), CellMatchrLauncher._INPUT_EXTS)
            self.assertEqual(selected, (str(confocal), 1))

    def test_preview_rejects_slide_only_and_unreadable_lif(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "slide.lif").touch()
            with patch("cellmatchr.lif_classify.classify_lif_file",
                       return_value={"file_type": "slide_scanner",
                                     "scenes": [(0, "merged", "slide_scanner")]}):
                self.assertIsNone(_first_confocal_preview_input(
                    str(root), CellMatchrLauncher._INPUT_EXTS))
            with patch("cellmatchr.lif_classify.classify_lif_file",
                       return_value={"file_type": "unknown", "scenes": []}):
                self.assertIsNone(_first_confocal_preview_input(
                    str(root), CellMatchrLauncher._INPUT_EXTS))
            with patch("cellmatchr.lif_classify.classify_lif_file",
                       side_effect=OSError("bad header")):
                self.assertIsNone(_first_confocal_preview_input(
                    str(root), CellMatchrLauncher._INPUT_EXTS))

    def test_custom_roi_folder_check_accepts_analysis_formats(self):
        with TemporaryDirectory(prefix="cellmatchr_[roi]_") as temp:
            folder = Path(temp)
            self.assertFalse(_has_analysis_roi_masks(str(folder)))
            (folder / "scan.roi").touch()
            self.assertTrue(_has_analysis_roi_masks(str(folder)))

    def _run_all_stub(self, project, analysis_enabled=True):
        app = object.__new__(CellMatchrLauncher)
        app.root = object()
        app.config = {
            "project_output_folder": str(project),
            "input_root_folder": "/input/current",
            "ana_enabled": analysis_enabled,
            "val_enabled": False,
        }
        app._ana_input_var = None
        app._is_running = Mock(return_value=False)
        app._sync_vars_to_config = Mock()
        app._step_is_done = Mock(return_value=True)
        app._resolve_segmentation_output_dir = Mock(
            return_value=str(project / "Segmentation_microsam_new"))
        app._choose_analysis_run = Mock(return_value="Segmentation")
        app._prepare_analysis_output = Mock(
            return_value=(str(project / "Analysis"), None))
        app._script = Mock(side_effect=lambda filename: filename)
        app._log = Mock()
        app._set_run_status = Mock()
        return app

    def test_run_all_new_choice_runs_current_input_as_new_run(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            app = self._run_all_stub(project)
            with patch("cellmatchr.launcher._ExistingSegmentationChoiceDialog") as dialog, \
                 patch("cellmatchr.launcher.save_config"), \
                 patch("cellmatchr.launcher.save_project_config"), \
                 patch("cellmatchr.launcher.threading.Thread") as thread:
                dialog.return_value.result = "new"
                app._run_all()
            app._resolve_segmentation_output_dir.assert_called_once_with(
                auto_new=True)
            app._choose_analysis_run.assert_not_called()
            self.assertEqual(app.config["seg_input_subfolder"],
                             "Segmentation_microsam_new")
            thread.return_value.start.assert_called_once()

    def test_run_all_existing_choice_asks_which_run_to_analyze(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            app = self._run_all_stub(project)
            with patch("cellmatchr.launcher._ExistingSegmentationChoiceDialog") as dialog, \
                 patch("cellmatchr.launcher.save_config"), \
                 patch("cellmatchr.launcher.save_project_config"), \
                 patch("cellmatchr.launcher.threading.Thread") as thread:
                dialog.return_value.result = "existing"
                app._run_all()
            app._resolve_segmentation_output_dir.assert_not_called()
            app._choose_analysis_run.assert_called_once_with(ask_if_multiple=True)
            app._prepare_analysis_output.assert_called_once_with("Segmentation")
            thread.return_value.start.assert_called_once()

    def test_run_all_cancel_keeps_existing_results(self):
        with TemporaryDirectory() as temp:
            app = self._run_all_stub(Path(temp))
            with patch("cellmatchr.launcher._ExistingSegmentationChoiceDialog") as dialog, \
                 patch("cellmatchr.launcher.save_config") as save, \
                 patch("cellmatchr.launcher.threading.Thread") as thread:
                dialog.return_value.result = None
                app._run_all()
            app._resolve_segmentation_output_dir.assert_not_called()
            app._choose_analysis_run.assert_not_called()
            save.assert_not_called()
            thread.assert_not_called()

    def test_run_all_external_scanner_input_requires_roi_confirmation(self):
        with TemporaryDirectory() as temp:
            project = Path(temp) / "project"
            project.mkdir()
            masks = project / "VSI_Converted_TIFF_A" / "DONOTDELETE-MASK_FOR_VSI"
            masks.mkdir(parents=True)
            (masks / "other_scan_mask_overview.tif").touch()
            external_input = Path(temp) / "external_exports"
            external_input.mkdir()
            app = self._run_all_stub(project)
            app.config.update({
                "input_type": "vsi",
                "input_root_folder": str(external_input),
                "ana_roi_mask_folder": str(masks),
            })
            with patch("cellmatchr.launcher._ExistingSegmentationChoiceDialog") as dialog, \
                 patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=False) as confirm, \
                 patch("cellmatchr.launcher.save_config") as save, \
                 patch("cellmatchr.launcher.threading.Thread") as thread:
                dialog.return_value.result = "new"
                app._run_all()
            self.assertIn(str(external_input), confirm.call_args.args[1])
            self.assertIn(str(masks), confirm.call_args.args[1])
            app._resolve_segmentation_output_dir.assert_not_called()
            app._prepare_analysis_output.assert_not_called()
            save.assert_not_called()
            thread.assert_not_called()

    def test_run_all_external_scanner_accepts_confirmed_custom_roi(self):
        with TemporaryDirectory() as temp:
            project = Path(temp) / "project"
            project.mkdir()
            external_input = Path(temp) / "external_exports"
            external_input.mkdir()
            custom = Path(temp) / "custom_masks"
            custom.mkdir()
            (custom / "crop.roi").touch()
            app = self._run_all_stub(project)
            app.config.update({
                "input_type": "vsi",
                "input_root_folder": str(external_input),
                "ana_roi_mask_folder": str(custom),
            })
            with patch("cellmatchr.launcher._ExistingSegmentationChoiceDialog") as dialog, \
                 patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=True) as confirm, \
                 patch("cellmatchr.launcher.save_config"), \
                 patch("cellmatchr.launcher.save_project_config"), \
                 patch("cellmatchr.launcher.threading.Thread") as thread:
                dialog.return_value.result = "new"
                app._run_all()
            confirm.assert_called_once()
            self.assertEqual(app.config["ana_roi_mask_folder"], str(custom))
            thread.return_value.start.assert_called_once()

    def test_converter_roi_mask_folder_follows_selected_output(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            original = project / "VSI_Converted_TIFF"
            newer = project / "VSI_Converted_TIFF_scanner"
            for folder in (original, newer):
                (folder / "exports").mkdir(parents=True)
            app = object.__new__(CellMatchrLauncher)
            app.config = {
                "project_output_folder": str(project),
                "ana_roi_mask_folder": str(original / "DONOTDELETE-MASK_FOR_VSI"),
            }
            roi_var = Mock()
            app._vars = {"ana_roi_mask_folder": roi_var}
            app._align_converter_roi_folder(str(newer / "exports"))
            expected = str(newer / "DONOTDELETE-MASK_FOR_VSI")
            self.assertEqual(app.config["ana_roi_mask_folder"], expected)
            self.assertEqual(app.config["vsi_output_dir"], str(newer))
            roi_var.set.assert_called_once_with(expected)

            app.config["ana_roi_mask_folder"] = str(project / "my_manual_masks")
            roi_var.reset_mock()
            app._align_converter_roi_folder(str(original / "exports"))
            self.assertEqual(app.config["ana_roi_mask_folder"],
                             str(project / "my_manual_masks"))
            roi_var.set.assert_not_called()

            # A standard mask folder from another converter run is automatic:
            # new segmentation on the selected input must use its own masks.
            app.config["vsi_output_dir"] = str(original)
            app.config["ana_roi_mask_folder"] = expected
            app._align_converter_roi_folder(str(original / "exports"))
            self.assertEqual(app.config["ana_roi_mask_folder"],
                             str(original / "DONOTDELETE-MASK_FOR_VSI"))

            app.config["ana_roi_mask_folder"] = (
                "/old/location/VSI_Converted_TIFF/DONOTDELETE-MASK_FOR_VSI")
            app.config["vsi_output_dir"] = "/old/location/VSI_Converted_TIFF"
            app._align_converter_roi_folder(str(newer / "exports"))
            self.assertEqual(app.config["ana_roi_mask_folder"], expected)

            app.config["input_type"] = "czi"
            app._clear_converter_roi_folder_for_direct_input()
            self.assertEqual(app.config["ana_roi_mask_folder"], "")
            roi_var.set.assert_called_with("")

            app.config["ana_roi_mask_folder"] = str(project / "my_manual_masks")
            app._clear_converter_roi_folder_for_direct_input()
            self.assertEqual(app.config["ana_roi_mask_folder"],
                             str(project / "my_manual_masks"))

    def test_analysis_uses_roi_masks_from_selected_segmentation_run(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            for variant in ("VSI_Converted_TIFF_A", "VSI_Converted_TIFF_B"):
                (project / variant / "exports").mkdir(parents=True)
            for run, source in (("Segmentation_A", "VSI_Converted_TIFF_A"),
                                ("Segmentation_direct", "")):
                folder = project / run
                folder.mkdir()
                (folder / "_run_progress.json").write_text(json.dumps({
                    "input_sources": {"converter_run": source},
                }))
            app = object.__new__(CellMatchrLauncher)
            app.root = object()
            app.config = {
                "project_output_folder": str(project),
                "vsi_output_dir": str(project / "VSI_Converted_TIFF_B"),
                "ana_roi_mask_folder": str(project / "VSI_Converted_TIFF_B"
                                           / "DONOTDELETE-MASK_FOR_VSI"),
            }
            roi_var = Mock()
            app._vars = {"ana_roi_mask_folder": roi_var}

            self.assertTrue(app._pair_roi_masks_with_run("Segmentation_A"))
            expected_a = str(project / "VSI_Converted_TIFF_A"
                             / "DONOTDELETE-MASK_FOR_VSI")
            self.assertEqual(app.config["ana_roi_mask_folder"], expected_a)
            with patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=False) as missing:
                self.assertFalse(app._pair_roi_masks_with_run(
                    "Segmentation_A", confirm_unknown=True))
            self.assertIn("no ROI masks", missing.call_args.args[1])
            self.assertTrue(app._pair_roi_masks_with_run("Segmentation_direct"))
            self.assertEqual(app.config["ana_roi_mask_folder"], "")

            app.config["ana_roi_mask_folder"] = expected_a
            with patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=False) as confirm:
                self.assertFalse(app._pair_roi_masks_with_run(
                    "Segmentation_legacy", confirm_unknown=True))
            self.assertIn("Segmentation_legacy", confirm.call_args.args[1])

    def test_known_converter_run_requires_custom_roi_confirmation(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            masks = project / "VSI_Converted_TIFF_A" / "DONOTDELETE-MASK_FOR_VSI"
            masks.mkdir(parents=True)
            (masks / "scan_mask_overview.tif").touch()
            run = project / "Segmentation_A"
            run.mkdir()
            (run / "_run_progress.json").write_text(json.dumps({
                "input_sources": {"converter_run": "VSI_Converted_TIFF_A"},
            }))
            custom = project / "custom_masks"
            custom.mkdir()
            app = object.__new__(CellMatchrLauncher)
            app.root = object()
            app.config = {
                "project_output_folder": str(project),
                "ana_roi_mask_folder": str(custom),
            }
            app._vars = {"ana_roi_mask_folder": Mock()}
            with patch("cellmatchr.launcher.messagebox.askyesnocancel",
                       return_value=None) as confirm:
                self.assertFalse(app._pair_roi_masks_with_run(
                    "Segmentation_A", confirm_unknown=True))
            self.assertIn(str(custom), confirm.call_args.args[1])
            self.assertEqual(app.config["ana_roi_mask_folder"], str(custom))
            with patch("cellmatchr.launcher.messagebox.askyesnocancel",
                       return_value=False):
                self.assertTrue(app._pair_roi_masks_with_run(
                    "Segmentation_A", confirm_unknown=True))
            self.assertEqual(app.config["ana_roi_mask_folder"], str(masks))

            app.config["ana_roi_mask_folder"] = str(custom)
            with patch("cellmatchr.launcher.messagebox.askyesnocancel",
                       return_value=True), \
                 patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=False) as missing:
                self.assertFalse(app._pair_roi_masks_with_run(
                    "Segmentation_A", confirm_unknown=True))
            self.assertIn("No .tif or .roi masks", missing.call_args.args[1])
            (custom / "scan.roi").touch()
            with patch("cellmatchr.launcher.messagebox.askyesnocancel",
                       return_value=True), \
                 patch("cellmatchr.launcher.messagebox.askyesno") as missing:
                self.assertTrue(app._pair_roi_masks_with_run(
                    "Segmentation_A", confirm_unknown=True))
            missing.assert_not_called()
            self.assertEqual(app.config["ana_roi_mask_folder"], str(custom))

    def test_external_symlinked_exports_are_not_auto_paired(self):
        with TemporaryDirectory() as temp:
            project = Path(temp) / "project"
            variant = project / "VSI_Converted_TIFF_A"
            variant.mkdir(parents=True)
            external = Path(temp) / "external_exports"
            external.mkdir()
            (variant / "exports").symlink_to(external, target_is_directory=True)
            app = object.__new__(CellMatchrLauncher)
            app.config = {"project_output_folder": str(project)}
            self.assertIsNone(app._input_converter_variant(str(variant / "exports")))

    def test_legacy_scanner_run_retains_single_converter_roi_default(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            masks = project / "VSI_Converted_TIFF" / "DONOTDELETE-MASK_FOR_VSI"
            masks.mkdir(parents=True)
            (masks / "sample_mask_overview.tif").touch()
            app = object.__new__(CellMatchrLauncher)
            app.root = object()
            app.config = {
                "project_output_folder": str(project),
                "input_type": "vsi",
                "ana_roi_mask_folder": "",
            }
            app._vars = {"ana_roi_mask_folder": Mock()}
            self.assertTrue(app._pair_roi_masks_with_run("Segmentation"))
            self.assertEqual(app.config["ana_roi_mask_folder"], str(masks))
            with patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=True) as confirm:
                self.assertTrue(app._pair_roi_masks_with_run(
                    "Segmentation", confirm_unknown=True))
            self.assertIn(str(masks), confirm.call_args.args[1])

    def test_multiple_converter_outputs_need_explicit_legacy_roi_choice(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            for name in ("VSI_Converted_TIFF_A", "VSI_Converted_TIFF_B"):
                (project / name).mkdir()
            app = object.__new__(CellMatchrLauncher)
            app.root = object()
            app.config = {
                "project_output_folder": str(project),
                "input_type": "vsi",
                "ana_roi_mask_folder": "",
            }
            app._vars = {"ana_roi_mask_folder": Mock()}
            with patch("cellmatchr.launcher.messagebox.askyesno",
                       return_value=False) as confirm:
                self.assertFalse(app._pair_roi_masks_with_run(
                    "Segmentation", confirm_unknown=True))
            self.assertIn("whole-image analysis", confirm.call_args.args[1])
