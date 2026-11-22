"""Metadata-only checks for Leica LIF routing in the slide converter."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr.roi_mask_matching import roi_mask_files, roi_match_candidates


def _scene(name, *, mosaic=1, scanner=False):
    return SimpleNamespace(
        name=name,
        dims=SimpleNamespace(x=100, y=80, m=mosaic),
        channels=2,
        nz=1,
        settings={} if scanner else {"ScanMode": "confocal"},
    )


class LifConverterRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The converter's napari/slideio imports are for the interactive UI.
        # These tests exercise only its metadata routing and never start Qt.
        with patch.dict(sys.modules, {
            "napari": SimpleNamespace(Viewer=object),
            "slideio": SimpleNamespace(),
        }):
            cls.converter = importlib.import_module("cellmatchr.vsi_converter")

    def _open(self, images):
        fake_lif = SimpleNamespace(get_iter_image=lambda: iter(images))
        with patch("readlif.reader.LifFile", return_value=fake_lif):
            return self.converter._LifSlide("example.lif")

    def test_confocal_lif_is_rejected_with_direct_read_guidance(self):
        images = [_scene("acquisition 1"), _scene("acquisition 2")]
        with self.assertRaisesRegex(ValueError, "Confocal direct-read in Step 2"):
            self._open(images)

    def test_mixed_lif_exposes_only_stitched_scanner_scene(self):
        images = [
            _scene("confocal acquisition"),
            _scene("raw tile", mosaic=40, scanner=True),
            _scene("merged slide", scanner=True),
        ]
        slide = self._open(images)
        self.assertEqual(slide.num_scenes, 1)
        self.assertEqual(slide.get_scene(0).name, "merged slide")

    def test_raw_mosaic_without_merged_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no stitched image"):
            self._open([_scene("raw tile", mosaic=40, scanner=True)])

    def test_brightfield_rgb_is_one_natural_color_layer(self):
        app = self.converter.VSINapariROIApp.__new__(self.converter.VSINapariROIApp)
        app.brightfield = True
        app.overview = np.full((12, 16, 3), (12, 57, 93), dtype=np.uint8)
        images = []
        app.viewer = SimpleNamespace(layers=[], add_image=lambda pixels, **kwargs: images.append((pixels, kwargs)))
        app._add_overview_layers()
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0][1]["rgb"])
        np.testing.assert_array_equal(images[0][0], app.overview)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stain.tif"
            self.converter.write_multichannel_tiff(path, app.overview, brightfield=True)
            with tifffile.TiffFile(path) as file:
                self.assertEqual(file.series[0].axes, "YXS")
                np.testing.assert_array_equal(file.asarray(), app.overview)

    def test_same_stem_scanner_sources_keep_earlier_exports_and_pair_new_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = ["abc.vsi", "ABC.lif"]
            saved = []
            for index, filename in enumerate(variants):
                source = root / str(index) / filename
                app = self.converter.VSINapariROIApp(str(source), directory)
                polygon = np.array([[1, 1], [6, 1], [6, 5], [1, 5]])
                app.shapes = SimpleNamespace(data=[polygon])
                app.overview = np.zeros((8, 8), dtype=np.uint8)
                app.roi_names = ["region"]
                app._overview_metadata = {"vsi_path": str(source)}
                app._roi_bbox_full_from_polygon = lambda _polygon: (0, 0, 8, 8)
                app.scene = SimpleNamespace(read_block=lambda _rectangle:
                                             np.full((8, 8), index + 1, np.uint16))
                masks = app.save_all_roi_masks_to_disk()
                exports = app.export_all_rois_fullres()
                export = Path(exports[0]["tiff_path"])
                saved.append((export, Path(masks[0]["mask_path"])))

            self.assertEqual(saved[0][0].name, "abc_slice01_roi01_FULLRES_region.tif")
            self.assertNotEqual(saved[0][0].name.casefold(), saved[1][0].name.casefold())
            candidates = roi_mask_files(str(saved[0][1].parent))
            for index, (export, mask) in enumerate(saved):
                selected, choices = roi_match_candidates(export.stem, candidates)
                self.assertEqual(Path(selected), mask)
                self.assertEqual(choices, [])
                self.assertEqual(int(tifffile.imread(export)[0, 0]), index + 1)
            self.assertEqual(len(list((root / "intermediate_files").glob("*_overview_meta.json"))), 2)

    def test_return_picker_lists_saved_crops_and_ignores_debug_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.converter.saved_roi_exports(root), [])
            exports = root / "exports"
            exports.mkdir()
            for name in ("slide_slice02_roi01_FULLRES_right V1.tif",
                         "slide_slice01_roi02_FULLRES_left V1.TIFF",
                         "DEBUG_slide_slice01_roi01_FULLRES_trial.tif",
                         "slide_slice01_roi01_mask_overview.tif"):
                tifffile.imwrite(exports / name, np.ones((4, 4), np.uint8))
            (exports / "slide_slice03_roi01_FULLRES_interrupted.tif").touch()
            saved = self.converter.saved_roi_exports(root)
            self.assertEqual([(scene, name) for _, scene, _, name, _ in saved],
                             [(1, "left V1"), (2, "right V1")])


if __name__ == "__main__":
    unittest.main()
