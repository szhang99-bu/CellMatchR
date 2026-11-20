"""Optional Sidecar touch controls for CellMatchR's Napari viewers.

Native recognizers accept direct one-finger pan and two-finger pinch touches.
Mouse and Pencil strokes remain with Napari, so ordinary drawing and shortcuts
still work.
"""

from __future__ import annotations

from ctypes import c_void_p
from functools import lru_cache
import sys
from typing import Any, Callable


def pan_viewer_camera(viewer: Any, dx: float, dy: float, flipped: bool) -> None:
    """Move the image with a finger, accounting for the native view's Y axis."""
    zoom = max(float(viewer.camera.zoom), 1e-6)
    center = list(viewer.camera.center)
    center[-1] -= dx / zoom
    center[-2] -= (dy if flipped else -dy) / zoom
    viewer.camera.center = tuple(center)


def zoom_viewer_camera(viewer: Any, magnification: float, x: float, y: float,
                       width: float, height: float, flipped: bool) -> None:
    """Zoom around the direct-touch pinch point in the Napari canvas."""
    old_zoom = max(float(viewer.camera.zoom), 1e-6)
    factor = max(0.1, min(10.0, 1.0 + magnification))
    new_zoom = max(1e-4, min(1e5, old_zoom * factor))
    center = list(viewer.camera.center)
    offset_x = x - width / 2
    offset_y = y - height / 2 if flipped else height / 2 - y
    center[-1] += offset_x * (1 / old_zoom - 1 / new_zoom)
    center[-2] += offset_y * (1 / old_zoom - 1 / new_zoom)
    viewer.camera.center = tuple(center)
    viewer.camera.zoom = new_zoom


@lru_cache(maxsize=1)
def _native_classes() -> tuple[Any, ...]:
    # Keep Objective-C class definitions unique across consecutive Napari windows.
    import objc
    from AppKit import (
        NSGestureRecognizerStateBegan, NSGestureRecognizerStateChanged,
        NSMagnificationGestureRecognizer, NSPanGestureRecognizer, NSView,
        NSTouchTypeMaskDirect,
    )
    from Foundation import NSObject

    class FingerOnlyDelegate(
        NSObject, protocols=[objc.protocolNamed("NSGestureRecognizerDelegate")]
    ):
        def gestureRecognizer_shouldAttemptToRecognizeWithEvent_(
            self, recognizer: Any, event: Any
        ) -> bool:
            return False  # Mouse and Pencil are handled by Napari.

        def gestureRecognizer_shouldReceiveTouch_(
            self, recognizer: Any, touch: Any
        ) -> bool:
            return True

    class FingerPanTarget(NSObject):
        @objc.IBAction
        def handlePan_(self, recognizer: Any) -> None:
            if recognizer.state() == NSGestureRecognizerStateBegan:
                recognizer.setTranslation_inView_((0, 0), self.native_view)
            elif recognizer.state() == NSGestureRecognizerStateChanged:
                delta = recognizer.translationInView_(self.native_view)
                recognizer.setTranslation_inView_((0, 0), self.native_view)
                self.on_pan(float(delta.x), float(delta.y), bool(self.native_view.isFlipped()))

    class FingerPinchTarget(NSObject):
        @objc.IBAction
        def handlePinch_(self, recognizer: Any) -> None:
            if recognizer.state() == NSGestureRecognizerStateBegan:
                recognizer.setMagnification_(0.0)
            elif recognizer.state() == NSGestureRecognizerStateChanged:
                amount = float(recognizer.magnification())
                recognizer.setMagnification_(0.0)
                point = recognizer.locationInView_(self.native_view)
                bounds = self.native_view.bounds()
                self.on_pinch(amount, float(point.x), float(point.y),
                              float(bounds.size.width), float(bounds.size.height),
                              bool(self.native_view.isFlipped()))

    return (FingerOnlyDelegate, FingerPanTarget, FingerPinchTarget,
            NSPanGestureRecognizer, NSMagnificationGestureRecognizer, NSView,
            NSTouchTypeMaskDirect, objc)


class SidecarControls:
    """Hold native objects alive and remove them when their viewer closes."""

    def __init__(self, native_view: Any, recognizer: Any, delegate: Any,
                 target: Any, monitor: Any = None, handler: Any = None):
        self.native_view = native_view
        self.recognizer = recognizer
        self.delegate = delegate
        self.target = target
        self.pinch_recognizer = None
        self.pinch_target = None
        self.monitor = monitor
        self.handler = handler
        self.close_filter = None

    def close(self) -> None:
        if self.monitor is not None:
            from AppKit import NSEvent
            NSEvent.removeMonitor_(self.monitor)
            self.monitor = None
        if self.recognizer is not None:
            self.native_view.removeGestureRecognizer_(self.recognizer)
            self.recognizer = None
        if self.pinch_recognizer is not None:
            self.native_view.removeGestureRecognizer_(self.pinch_recognizer)
            self.pinch_recognizer = None
        self.pinch_target = None
        if self.close_filter is not None:
            self.close_filter.on_close = None
            self.close_filter = None
        self.handler = None


def install_sidecar_controls(
    viewer: Any, on_double_tap: Callable[[], None] | None = None
) -> SidecarControls | None:
    """Add Sidecar finger navigation and optional Pencil tool switching.

    Returns None on non-Mac platforms. On macOS, a missing PyObjC dependency
    raises ImportError so callers can leave the ordinary viewer running.
    """
    if sys.platform != "darwin":
        return None

    (delegate_type, pan_target_type, pinch_target_type, pan_type, pinch_type,
     view_type, direct_touches, objc) = _native_classes()
    qt_window = viewer.window._qt_window
    canvas = qt_window._qt_viewer.canvas.native
    native_view = objc.objc_object(c_void_p=c_void_p(int(canvas.winId())))
    if not native_view.isKindOfClass_(view_type):
        raise TypeError("Napari canvas does not expose an AppKit NSView")

    target = pan_target_type.alloc().init()
    target.native_view = native_view
    target.on_pan = lambda dx, dy, flipped: pan_viewer_camera(viewer, dx, dy, flipped)
    delegate = delegate_type.alloc().init()
    recognizer = pan_type.alloc().initWithTarget_action_(target, b"handlePan:")
    recognizer.setMinimumNumberOfTouches_(1)
    recognizer.setMaximumNumberOfTouches_(1)
    recognizer.setAllowedTouchTypes_(direct_touches)
    recognizer.setDelegate_(delegate)
    native_view.addGestureRecognizer_(recognizer)

    controls = SidecarControls(native_view, recognizer, delegate, target)
    try:
        pinch_target = pinch_target_type.alloc().init()
        pinch_target.native_view = native_view
        pinch_target.on_pinch = lambda amount, x, y, width, height, flipped: (
            zoom_viewer_camera(viewer, amount, x, y, width, height, flipped))
        pinch = pinch_type.alloc().initWithTarget_action_(pinch_target, b"handlePinch:")
        pinch.setAllowedTouchTypes_(direct_touches)
        pinch.setDelegate_(delegate)
        native_view.addGestureRecognizer_(pinch)
        controls.pinch_recognizer = pinch
        controls.pinch_target = pinch_target

        from qtpy.QtCore import QEvent, QObject

        class WindowCloseFilter(QObject):
            def eventFilter(self, watched: Any, event: Any) -> bool:
                if event.type() == QEvent.Close and self.on_close is not None:
                    self.on_close()
                return False

        close_filter = WindowCloseFilter(qt_window)
        close_filter.on_close = controls.close
        controls.close_filter = close_filter
        qt_window.installEventFilter(close_filter)
        if on_double_tap is not None:
            from AppKit import NSEvent, NSEventMaskChangeMode

            def handle_double_tap(event: Any) -> Any:
                # Local monitors are application-wide; only the foreground
                # viewer may switch tools when several windows are open.
                if not qt_window.isActiveWindow():
                    return event
                on_double_tap()
                return None

            controls.handler = handle_double_tap
            controls.monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskChangeMode, handle_double_tap
            )
        qt_window.destroyed.connect(lambda *_: controls.close())
    except Exception:
        controls.close()
        raise
    return controls
