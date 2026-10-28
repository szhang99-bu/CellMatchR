"""LIF input routing and sample names; no image or model output is created."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cellmatchr.lif_classify import classify_lif_image


def _load_planner():
    """Load only the lightweight planner, without importing model libraries."""
    path = Path(__file__).resolve().parents[1] / "cellmatchr" / "segmentation.py"
    tree = ast.parse(path.read_text())
    names = {"_sanitize_scene_name", "_plan_work_items"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


def _load_source_guard():
    path = Path(__file__).resolve().parents[1] / "cellmatchr" / "segmentation.py"
    tree = ast.parse(path.read_text())
    names = {"_run_progress_path", "_input_source_record",
             "_check_resume_sources"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"os": os, "RESUME": True,
                 "RUN_PROGRESS_FILENAME": "_run_progress.json"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


class LifRoutingTests(unittest.TestCase):
    def setUp(self):
        self.planner = _load_planner()
        self.root = "/example/input"

    def _plan(self, files, scenes):
        self.planner["lif_scenes_to_process"] = lambda path: scenes[path]
        return self.planner["_plan_work_items"](files, self.root)

    def test_scene_metadata_distinguishes_scanner_from_confocal(self):
        confocal = SimpleNamespace(dims=SimpleNamespace(m=1), settings={"Zoom": 1})
        raw_scan = SimpleNamespace(dims=SimpleNamespace(m=40), settings={"Zoom": 1})
        merged_scan = SimpleNamespace(dims=SimpleNamespace(m=1), settings={})
        self.assertEqual(classify_lif_image(confocal), "confocal")
        self.assertEqual(classify_lif_image(raw_scan), "slide_scanner")
        self.assertEqual(classify_lif_image(merged_scan), "slide_scanner")

    def test_unique_names_keep_legacy_layout_and_slide_scan_is_skipped(self):
        confocal = "/example/input/confocal/Confocal_63X.lif"
        scanner = "/example/input/scanner/Slidescanner_10x rep.lif"
        plain = "/example/input/other.czi"
        result = self._plan(
            [confocal, scanner, plain],
            {confocal: [(0, "first scene")], scanner: []},
        )
        self.assertEqual(result, [
            (confocal, 0, "Confocal_63X"),
            (plain, None, "other"),
        ])

    def test_mixed_lif_uses_only_confocal_scenes(self):
        source = "/example/input/mixed.lif"
        result = self._plan([source], {source: [(1, "confocal")]})
        self.assertEqual(result, [(source, 1, "mixed")])

    def test_same_stem_in_two_folders_is_stable_and_distinct(self):
        first = "/example/input/a/sample.lif"
        second = "/example/input/b/sample.lif"
        scenes = {first: [(0, "a")], second: [(0, "b")]}
        one = self._plan([first, second], scenes)
        two = self._plan([second, first], scenes)
        by_source = {path: name for path, _, name in one}
        self.assertEqual(by_source, {path: name for path, _, name in two})
        self.assertEqual(len({name.casefold() for name in by_source.values()}), 2)
        self.assertTrue(all(name.startswith("sample__source_")
                            for name in by_source.values()))
        planned, collided = self.planner["_plan_work_items"](
            [first, second], self.root, return_collisions=True)
        self.assertTrue(collided)
        self.assertEqual(planned, one)

    def test_duplicate_scene_names_get_scene_specific_names(self):
        source = "/example/input/multi.lif"
        result = self._plan([source], {source: [(0, "same"), (1, "same")]})
        names = [name for _, _, name in result]
        self.assertEqual(len(set(names)), 2)
        self.assertIn("_scene1", names[0])
        self.assertIn("_scene2", names[1])

    def test_case_insensitive_collisions_are_disambiguated(self):
        lif = "/example/input/Sample.lif"
        czi = "/example/input/sample.czi"
        result = self._plan([lif, czi], {lif: [(0, "only")]})
        self.assertEqual(len({name.casefold() for _, _, name in result}), 2)

    def test_unreadable_lif_fails_before_any_work_item_is_returned(self):
        valid = "/example/input/valid.czi"
        broken = "/example/input/broken.lif"
        def fail(_):
            raise ValueError("unreadable metadata")
        self.planner["lif_scenes_to_process"] = fail
        with self.assertRaisesRegex(RuntimeError, "Segmentation has not started"):
            self.planner["_plan_work_items"]([valid, broken], self.root)

    def test_resume_uses_same_relative_inputs_and_sample_plan(self):
        source_guard = _load_source_guard()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_one = root / "first" / "inputs"
            input_two = root / "moved" / "inputs"
            for input_root in (input_one, input_two):
                (input_root / "a").mkdir(parents=True)
                (input_root / "a" / "sample.lif").write_bytes(b"fixture")
            first = input_one / "a" / "sample.lif"
            moved = input_two / "a" / "sample.lif"
            record = source_guard["_input_source_record"](
                str(input_one), [str(first)], [(str(first), 0, "sample")])
            moved_record = source_guard["_input_source_record"](
                str(input_two), [str(moved)], [(str(moved), 0, "sample")])
            self.assertEqual(record, moved_record)

            run = root / "run"
            run.mkdir()
            progress = run / "_run_progress.json"
            progress.write_text(json.dumps({"input_sources": record}))
            source_guard["_check_resume_sources"](str(run), moved_record)

            changed = source_guard["_input_source_record"](
                str(input_two), [str(moved)], [(str(moved), 1, "sample")])
            with self.assertRaisesRegex(ValueError, "input files or Leica scenes"):
                source_guard["_check_resume_sources"](str(run), changed)

            progress.write_text(json.dumps({"run_identity": {}}))
            # Older progress records have no source list and must still open.
            source_guard["_check_resume_sources"](str(run), changed)
            # Except when current inputs would collide with old sample names.
            with self.assertRaisesRegex(ValueError, "Start a new run"):
                source_guard["_check_resume_sources"](
                    str(run), changed, has_name_collisions=True)

    def test_source_record_ignores_slide_scanner_file_skipped_by_direct_read(self):
        source_guard = _load_source_guard()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            confocal = root / "confocal.lif"
            scanner = root / "slide.lif"
            confocal.write_bytes(b"confocal")
            scanner.write_bytes(b"slide")
            sample = [(str(confocal), 0, "confocal")]
            first = source_guard["_input_source_record"](
                str(root), [str(confocal), str(scanner)], sample)
            scanner.write_bytes(b"changed slide")
            second = source_guard["_input_source_record"](
                str(root), [str(confocal), str(scanner)], sample)
            self.assertEqual(first, second)
            self.assertEqual([entry["path"] for entry in first["files"]],
                             ["confocal.lif"])

    def test_source_record_links_converter_exports_to_its_project_run(self):
        source_guard = _load_source_guard()
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            exports = project / "VSI_Converted_TIFF_scan2" / "exports"
            exports.mkdir(parents=True)
            source = exports / "crop.tif"
            source.write_bytes(b"crop")
            record = source_guard["_input_source_record"](
                str(exports), [str(source)], [(str(source), None, "crop")],
                project_root=str(project))
            self.assertEqual(record["converter_run"],
                             "VSI_Converted_TIFF_scan2")
            direct = source_guard["_input_source_record"](
                str(exports), [str(source)], [(str(source), None, "crop")],
                project_root=str(project / "other"))
            self.assertEqual(direct["converter_run"], "")
            external = source_guard["_input_source_record"](
                str(exports), [str(source)], [(str(source), None, "crop")],
                project_root=str(project / "other"), input_type="vsi")
            self.assertIsNone(external["converter_run"])

    def test_source_record_does_not_pair_external_symlinked_exports(self):
        source_guard = _load_source_guard()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            variant = project / "VSI_Converted_TIFF_scan"
            variant.mkdir(parents=True)
            external = root / "external_exports"
            external.mkdir()
            linked = variant / "exports"
            linked.symlink_to(external, target_is_directory=True)
            source = linked / "crop.tif"
            source.write_bytes(b"crop")
            record = source_guard["_input_source_record"](
                str(linked), [str(source)], [(str(source), None, "crop")],
                project_root=str(project), input_type="vsi")
            self.assertIsNone(record["converter_run"])


if __name__ == "__main__":
    unittest.main()
