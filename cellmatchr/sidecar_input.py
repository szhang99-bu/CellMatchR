"""Optional Sidecar touch controls for CellMatchR's Napari viewers.

The native recognizer accepts one-finger touch only. Mouse and Pencil strokes
remain with Napari, so its normal drawing, navigation, and shortcuts still work.
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


@lru_cache(maxsize=1)
def _native_classes() -> tuple[Any, ...]:
    # Keep Objective-C class definitions unique across consecutive Napari windows.
    import objc
    from AppKit import (
        NSGestureRecognizerStateBegan, NSGestureRecognizerStateChanged,
        NSPanGestureRecognizer, NSView,
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

    return FingerOnlyDelegate, FingerPanTarget, NSPanGestureRecognizer, NSView, objc


class SidecarControls:
    """Hold native objects alive and remove them when their viewer closes."""

    def __init__(self, native_view: Any, recognizer: Any, delegate: Any,
                 target: Any, monitor: Any = None, handler: Any = None):
        self.native_view = native_view
        self.recognizer = recognizer
        self.delegate = delegate
        self.target = target
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
        if self.close_filter is not None:
            self.close_filter.on_close = None
            self.close_filter = None
        self.handler = None


def install_sidecar_controls(
    viewer: Any, on_double_tap: Callable[[], None] | None = None
) -> SidecarControls | None:
    """Add finger pan and optional Pencil double-tap without changing Napari tools.

    Returns None on non-Mac platforms. On macOS, a missing PyObjC dependency
    raises ImportError so callers can leave the ordinary viewer running.
    """
    if sys.platform != "darwin":
        return None

    delegate_type, target_type, recognizer_type, view_type, objc = _native_classes()
    qt_window = viewer.window._qt_window
    canvas = qt_window._qt_viewer.canvas.native
    native_view = objc.objc_object(c_void_p=c_void_p(int(canvas.winId())))
    if not native_view.isKindOfClass_(view_type):
        raise TypeError("Napari canvas does not expose an AppKit NSView")

    target = target_type.alloc().init()
    target.native_view = native_view
    target.on_pan = lambda dx, dy, flipped: pan_viewer_camera(viewer, dx, dy, flipped)
    delegate = delegate_type.alloc().init()
    recognizer = recognizer_type.alloc().initWithTarget_action_(target, b"handlePan:")
    recognizer.setMinimumNumberOfTouches_(1)
    recognizer.setMaximumNumberOfTouches_(1)
    recognizer.setDelegate_(delegate)
    native_view.addGestureRecognizer_(recognizer)

    controls = SidecarControls(native_view, recognizer, delegate, target)
    try:
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
