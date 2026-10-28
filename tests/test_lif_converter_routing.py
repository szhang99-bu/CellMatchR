"""Metadata-only checks for Leica LIF routing in the slide converter."""

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
