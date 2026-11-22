"""Check Sidecar navigation while preserving ordinary Napari tools."""

import sys
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from cellmatchr.roi_editor import RoiEditorApp
from cellmatchr.sidecar_input import (
    SidecarControls, _native_classes, install_sidecar_controls,
    pan_viewer_camera, zoom_viewer_camera,
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

    def test_two_finger_pinch_zooms_around_touch_point(self):
        viewer = SimpleNamespace(camera=SimpleNamespace(zoom=2.0, center=(0, 20, 30)))
        zoom_viewer_camera(viewer, 1.0, 70, 60, 100, 100, True)
        self.assertEqual(viewer.camera.zoom, 4.0)
        self.assertEqual(viewer.camera.center, (0, 22.5, 35.0))
        zoom_viewer_camera(viewer, -0.5, 50, 50, 100, 100, False)
        self.assertEqual(viewer.camera.zoom, 2.0)
        self.assertEqual(viewer.camera.center, (0, 22.5, 35.0))

    def test_closing_viewer_removes_both_touch_recognizers(self):
        native = Mock()
        pan, pinch = object(), object()
        controls = SidecarControls(native, pan, object(), object())
        controls.pinch_recognizer = pinch
        controls.pinch_target = object()
        controls.close()
        self.assertEqual(native.removeGestureRecognizer_.call_args_list,
                         [call(pan), call(pinch)])
        self.assertIsNone(controls.pinch_target)

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
        app._sync_roi_ui = Mock()

        app._on_shapes_data_changed(SimpleNamespace(action="adding"))
        app._sync_roi_ui.assert_not_called()
        app._on_shapes_data_changed(SimpleNamespace(action="added"))
        app._sync_roi_ui.assert_called_once_with()
        self.assertEqual(app.shapes.mode, "add_polygon")

        app._toggle_sidecar_shape_tool()
        self.assertEqual(app.shapes.mode, "select")
        self.assertIs(app.viewer.layers.selection.active, app.shapes)
        app._on_shapes_data_changed()
        self.assertEqual(app.shapes.mode, "select")

        app._toggle_sidecar_shape_tool()
        self.assertEqual(app.shapes.mode, "add_polygon")
        app.shapes.mode = "add_ellipse"
        app._on_shapes_data_changed()
        self.assertEqual(app.shapes.mode, "add_ellipse")


if __name__ == "__main__":
    unittest.main()
