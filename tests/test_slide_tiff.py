import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from cellmatchr.slide_tiff import TiffSlide


class SlideTiffTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)

    def test_tiled_bigtiff_keeps_channels_and_projects_only_first_timepoint(self):
        path = self.folder / "scan [1].ome.tif"
        rng = np.random.default_rng(7)
        image = rng.integers(0, 5000, (2, 3, 2, 74, 91), dtype=np.uint16)
        image[1] = 6000
        tifffile.imwrite(path, image, ome=True, bigtiff=True, tile=(16, 16),
                         compression="deflate", metadata={"axes": "TCZYX"})

        slide = TiffSlide(path)
        self.addCleanup(slide.close)
        scene = slide.get_scene(0)
        self.assertEqual(scene.size, (91, 74))
        self.assertEqual(scene.num_channels, 3)
        with patch.object(tifffile.TiffPageSeries, "asarray", side_effect=AssertionError("full TIFF read")):
            crop = scene.read_block((13, 18, 40, 39))
        expected = image[0, :, :, 18:57, 13:53].max(axis=1).transpose(1, 2, 0)
        np.testing.assert_array_equal(crop, expected)

    def test_pyramid_preview_and_full_resolution_export_use_different_levels(self):
        path = self.folder / "pyramid.tiff"
        base = np.arange(128 * 96, dtype=np.uint16).reshape(96, 128)
        preview = np.full((48, 64), 20000, dtype=np.uint16)
        with tifffile.TiffWriter(path, bigtiff=True) as writer:
            writer.write(base, subifds=1, tile=(16, 16), compression="deflate")
            writer.write(preview, subfiletype=1, tile=(16, 16), compression="deflate")

        slide = TiffSlide(path)
        self.addCleanup(slide.close)
        scene = slide.get_scene(0)
        np.testing.assert_array_equal(scene.read_block(size=(64, 48)), preview)
        np.testing.assert_array_equal(scene.read_block((20, 17, 25, 29)), base[17:46, 20:45])

    def test_rgb_preserves_contiguous_and_planar_samples(self):
        image = np.arange(33 * 32 * 3, dtype=np.uint16).reshape(33, 32, 3)
        for layout, source in (("contig", image), ("separate", np.moveaxis(image, -1, 0))):
            with self.subTest(layout=layout):
                path = self.folder / f"{layout}.tif"
                tifffile.imwrite(path, source, photometric="rgb", planarconfig=layout)
                slide = TiffSlide(path)
                self.addCleanup(slide.close)
                scene = slide.get_scene(0)
                self.assertEqual(scene.num_channels, 3)
                np.testing.assert_array_equal(scene.read_block((2, 3, 17, 18)), image[3:21, 2:19])

    def test_unlabelled_pages_remain_separate_grayscale_scenes(self):
        path = self.folder / "pages.tif"
        pages = np.arange(3 * 19 * 17, dtype=np.uint16).reshape(3, 19, 17)
        tifffile.imwrite(path, pages, photometric="minisblack", metadata=None)

        slide = TiffSlide(path)
        self.addCleanup(slide.close)
        self.assertEqual(slide.num_scenes, 3)
        np.testing.assert_array_equal(slide.get_scene(1).read_block(), pages[1])
