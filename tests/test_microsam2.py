"""Regression checks for native color input and separate segmentation identities."""

import tempfile
import unittest
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tifffile

from cellmatchr import segmentation as seg
from cellmatchr.launcher import CellMatchrLauncher
from cellmatchr.microsam2 import _download_weight_ranges, _prepare_model_weights


class MicroSam2Tests(unittest.TestCase):
    def test_cached_encoder_is_reused_and_only_missing_decoder_is_fetched(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            names = ("hvit_b_cells", "hvit_b_cells_decoder")
            (folder / names[0]).write_bytes(b"known encoder")
            registry = SimpleNamespace(abspath=folder, fetch=Mock(), registry={
                names[0]: hashlib.sha256(b"known encoder").hexdigest(),
                names[1]: hashlib.sha256(b"known decoder").hexdigest(),
            })
            util = SimpleNamespace(models=lambda: registry)
            with patch.dict(sys.modules, {"micro_sam.v2": SimpleNamespace(util=util)}):
                _prepare_model_weights(names[0])
                self.assertEqual(registry.fetch.call_args.args, (names[1],))
                registry.fetch.reset_mock()
                (folder / names[1]).write_bytes(b"known decoder")
                _prepare_model_weights(names[0])
            registry.fetch.assert_not_called()

    def test_slow_download_switches_connections_without_redownloading_bytes(self):
        from io import BytesIO

        size = 32_000_000
        payload = b"ABCD1234" * (size // 8)

        class Response:
            status_code = 206

            def __init__(self, range_header):
                first, last = map(int, range_header.removeprefix("bytes=").split("-"))
                self.headers = {"Content-Range": f"bytes {first}-{last}/{size}"}
                self.body = BytesIO(payload[first:last + 1])

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.body.close()

            def iter_content(self, chunk_size):
                return iter(lambda: self.body.read(chunk_size), b"")

        for slow in (False, True):
            with self.subTest(slow=slow), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "download"
                elapsed = 7 if slow else 0
                clock = iter([0, elapsed])
                with patch("time.monotonic", side_effect=lambda: next(clock, elapsed)), patch(
                    "requests.get", side_effect=lambda _url, *, headers, **_kw: Response(headers["Range"]),
                ) as get:
                    self.assertTrue(_download_weight_ranges(
                        "https://owncloud.gwdg.de/checkpoint", destination, "decoder (file 2 of 2)"))
                requests = [call.kwargs["headers"]["Range"] for call in get.call_args_list]
                self.assertEqual(requests[:2], ["bytes=0-0", f"bytes=0-{size - 1}"])
                self.assertEqual(len(requests), 6 if slow else 2)
                if slow:
                    remaining_ranges = sorted(
                        tuple(map(int, bounds.removeprefix("bytes=").split("-"))) for bounds in requests[2:]
                    )
                    start = 256 * 1024
                    for first, last in remaining_ranges:
                        self.assertEqual(first, start)
                        start = last + 1
                    self.assertEqual(start, size)
                self.assertEqual(destination.read_bytes(), payload)

    def test_converter_tiff_feeds_native_rgb_to_model_and_gray_to_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "nissl.tif"
            rgb = np.full((32, 48, 3), (220, 230, 240), dtype=np.uint8)
            rgb[5:12, 5:12] = (12, 20, 65)
            tifffile.imwrite(source, np.moveaxis(rgb, -1, 0), imagej=True,
                             photometric="minisblack", metadata={"axes": "CYX"})
            paths, native = seg.prepare_prediction_channels(
                str(source), directory, "nissl", lif_scene=None, brightfield=True, use_microsam2=True)
            gray = tifffile.imread(paths[0])
            self.assertEqual(list(paths), [0])
            self.assertEqual(gray.shape, rgb.shape[:2])
            self.assertGreater(gray[19, 19], gray[7, 7])

            proposals = np.zeros(gray.shape, dtype=np.uint32)
            proposals[5:12, 5:12] = 1
            proposals[17:24, 17:24] = 2
            model = Mock()
            model.predict.return_value = proposals
            with patch.multiple(seg, SLIDE_SIGNAL_TYPE="brightfield", MIN_OBJECT_SIZE=0,
                                MAX_OBJECT_SIZE=0, MAX_MAJOR_MINOR_RATIO=0,
                                RAW_MEAN_MIN=0, RAW_MEAN_MAX=150):
                labels = seg.segment_by_tiling(
                    None, None, gray, (64, 64), (0, 0), 2,
                    microsam2_model=model, native_brightfield=native)
            np.testing.assert_array_equal(model.predict.call_args.args[0], rgb)
            self.assertGreater(labels[7, 7], 0)
            self.assertEqual(labels[19, 19], 0)

    def test_resume_distinguishes_microsam2_from_microsam(self):
        launcher = object.__new__(CellMatchrLauncher)
        launcher.config = {
            "input_type": "vsi", "slide_signal_type": "brightfield",
            "seg_backend": "microsam2", "seg_channels": "0",
            "seg_microsam2_model": "hvit_t_cells", "seg_model_type": "vit_b_lm",
            "seg_preprocess_mode": "brightfield_invert",
        }
        with patch.multiple(seg, SEG_BACKEND="microsam2", MICROSAM2_MODEL="hvit_t_cells",
                            CHANNELS_TO_PREDICT=[0], SLIDE_SIGNAL_TYPE="brightfield"):
            identity = seg._run_identity()
        self.assertEqual(identity, launcher._segmentation_identity_from_config())
        self.assertEqual(identity["preprocess"], "native_rgb")
        launcher.config["seg_backend"] = "microsam"
        original = launcher._segmentation_identity_from_config()
        self.assertEqual(original["model"], "vit_b_lm")
        self.assertEqual(original["preprocess"], "brightfield_invert")
        self.assertNotEqual(identity, original)
