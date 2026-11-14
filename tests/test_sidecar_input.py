"""Check Sidecar navigation while preserving ordinary Napari tools."""

import sys
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cellmatchr.roi_editor import RoiEditorApp
from cellmatchr.sidecar_input import (
    _native_classes, install_sidecar_controls, pan_viewer_camera,
)
from cellmatchr.vsi_converter import VSINapariROIApp


class SidecarInputTests(unittest.TestCase):
    def test_non_mac_install_does_not_access_or_change_viewer(self):
        with patch.object(sys, "platform", "linux"):
            self.assertIsNone(install_sidecar_controls(None, lambda: None))

    @unittest.skipUnless(
        sys.platform == "darwin" and importlib.util.find_spec("objc") is not None,
        "requires the macOS native bridge",
    )
    def test_native_touch_delegate_leaves_mouse_and_pencil_to_napari(self):
        delegate_type = _native_classes()[0]
        delegate = delegate_type.alloc().init()
        self.assertFalse(delegate.gestureRecognizer_shouldAttemptToRecognizeWithEvent_(None, None))
        self.assertTrue(delegate.gestureRecognizer_shouldReceiveTouch_(None, None))

    def test_finger_pan_uses_camera_zoom_and_view_orientation(self):
        viewer = SimpleNamespace(camera=SimpleNamespace(zoom=2.0, center=(0, 20, 30)))
        pan_viewer_camera(viewer, 8, 4, True)
        self.assertEqual(viewer.camera.center, (0, 18, 26))
        pan_viewer_camera(viewer, 8, 4, False)
        self.assertEqual(viewer.camera.center, (0, 20, 22))

    def test_pencil_toggle_chooses_existing_add_and_erase_tools(self):
        app = object.__new__(RoiEditorApp)
        app._add_lyr = SimpleNamespace(selected_label=4)
        app._erase_lyr = object()
        selection = set()
        app.viewer = SimpleNamespace(layers=SimpleNamespace(selection=selection))
        app._select_eraser = Mock()
        app._set_brush_colour = Mock()
        app._set_status = Mock()

        app._toggle_sidecar_tool()
        app._select_eraser.assert_called_once_with()
        self.assertFalse(app._set_brush_colour.called)

        selection.add(app._erase_lyr)
        app._toggle_sidecar_tool()
        app._set_brush_colour.assert_called_once_with(4)

    def test_slide_scanner_keeps_ordinary_polygon_behavior(self):
        app = object.__new__(VSINapariROIApp)
        app.shapes = SimpleNamespace(mode="add_polygon", data=[])
        app.viewer = SimpleNamespace(
            layers=SimpleNamespace(selection=SimpleNamespace(active=None)), status=""
        )
        app._sidecar_select_mode = False
        app._sync_roi_ui = Mock()
        app._ensure_roi_on_top = Mock()
        app._force_polygon_mode = Mock()

        app._on_shapes_data_changed()
        app._force_polygon_mode.assert_called_once_with()

        app._force_polygon_mode.reset_mock()
        app._toggle_sidecar_shape_tool()
        self.assertEqual(app.shapes.mode, "select")
        self.assertIs(app.viewer.layers.selection.active, app.shapes)
        app._on_shapes_data_changed()
        app._force_polygon_mode.assert_not_called()

        app._toggle_sidecar_shape_tool()
        self.assertEqual(app.shapes.mode, "add_polygon")
        app._on_shapes_data_changed()
        app._force_polygon_mode.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
