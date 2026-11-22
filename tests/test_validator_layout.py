"""Regression checks for the one-channel object-review layout."""

import unittest

from matplotlib.figure import Figure

from cellmatchr.validator import _build_review_axes


class ValidatorLayoutTests(unittest.TestCase):
    def test_image_and_context_axes_exist_with_one_or_many_channels(self):
        for count in (1, 2, 4):
            with self.subTest(channels=count):
                figure = Figure()
                color, gray, nearby, merge, context = _build_review_axes(figure, count)
                self.assertEqual([len(color), len(gray), len(nearby)], [count] * 3)
                self.assertIsNot(merge, context)
                self.assertGreater(context.get_position().height, 0)
                self.assertEqual(len(figure.axes), count * 3 + 2)
