"""Active window detection — cross-platform helpers."""

import logging
import platform

logger = logging.getLogger(__name__)


def get_active_window_rect():
    os_name = platform.system()
    if os_name == "Windows":
        return _get_windows_rect()
    elif os_name == "Darwin":
        return _get_macos_rect_native()
    else:
        logger.warning("get_active_window_rect: unsupported OS %s", os_name)
        return None


def _get_windows_rect():
    import ctypes
    from ctypes import wintypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return None

    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    x, y = rect.left, rect.top
    w, h = rect.right - rect.left, rect.bottom - rect.top
    return (x, y, w, h)


def _get_macos_rect_native():
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )
        from Cocoa import NSWorkspace
    except ImportError:
        logger.error("pyobjc-framework-Quartz / Cocoa not installed")
        return None

    workspace = NSWorkspace.sharedWorkspace()
    active_app = workspace.frontmostApplication()
    if not active_app:
        return None

    active_pid = active_app.processIdentifier()
    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    )

    for window in window_list:
        if window.get("kCGWindowOwnerPID") != active_pid:
            continue
        if window.get("kCGWindowLayer", 0) != 0:
            continue
        bounds = window.get("kCGWindowBounds")
        if bounds:
            x, y = int(bounds["X"]), int(bounds["Y"])
            w, h = int(bounds["Width"]), int(bounds["Height"])
            if w > 10 and h > 10:
                return (x, y, w, h)

    logger.debug("Active app found but no valid main window detected")
    return None
