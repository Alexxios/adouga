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


def get_active_window_info() -> tuple:
    """Return ``(rect, app_name, window_title)`` for the focused window.

    ``rect`` is ``(x, y, w, h)`` or ``None`` when unavailable. ``app_name``
    is the executable basename (Windows) or localized application name
    (macOS); ``window_title`` is the window's caption. Both strings default
    to ``""`` when they cannot be determined.
    """
    os_name = platform.system()
    if os_name == "Windows":
        return _get_windows_info()
    if os_name == "Darwin":
        return _get_macos_info_native()
    logger.warning("get_active_window_info: unsupported OS %s", os_name)
    return (None, "", "")


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


def _get_windows_info() -> tuple:
    import ctypes
    import os
    from ctypes import wintypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return (None, "", "")

    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    x, y = rect.left, rect.top
    w, h = rect.right - rect.left, rect.bottom - rect.top
    rect_tuple = (x, y, w, h)

    # Window title
    title = ""
    try:
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value or ""
    except Exception:
        logger.debug("GetWindowTextW failed", exc_info=True)

    # App name — basename of the owning process executable
    app_name = ""
    try:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h_proc = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value,
        )
        if h_proc:
            try:
                buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(len(buf))
                if kernel32.QueryFullProcessImageNameW(
                    h_proc, 0, buf, ctypes.byref(size),
                ):
                    app_name = os.path.basename(buf.value)
            finally:
                kernel32.CloseHandle(h_proc)
    except Exception:
        logger.debug("QueryFullProcessImageNameW failed", exc_info=True)

    return (rect_tuple, app_name, title)


def _get_macos_info_native() -> tuple:
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )
        from Cocoa import NSWorkspace
    except ImportError:
        logger.error("pyobjc-framework-Quartz / Cocoa not installed")
        return (None, "", "")

    workspace = NSWorkspace.sharedWorkspace()
    active_app = workspace.frontmostApplication()
    if not active_app:
        return (None, "", "")

    app_name = ""
    try:
        app_name = str(active_app.localizedName() or "")
    except Exception:
        logger.debug("localizedName() failed", exc_info=True)

    active_pid = active_app.processIdentifier()
    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    )

    rect_tuple = None
    title = ""
    for window in window_list:
        if window.get("kCGWindowOwnerPID") != active_pid:
            continue
        if window.get("kCGWindowLayer", 0) != 0:
            continue
        bounds = window.get("kCGWindowBounds")
        if not bounds:
            continue
        x, y = int(bounds["X"]), int(bounds["Y"])
        w, h = int(bounds["Width"]), int(bounds["Height"])
        if w > 10 and h > 10:
            rect_tuple = (x, y, w, h)
            title = str(window.get("kCGWindowName") or "")
            break

    return (rect_tuple, app_name, title)
