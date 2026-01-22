import platform
import psutil
import threading
from pynput import keyboard, mouse

def get_active_processes(pids=[]):
    processes = []
    for proc in psutil.process_iter():
        try:
            # Get process name
            if proc.pid not in pids:
                continue
            process_name = str(proc.pid) + ": " + proc.name()
            processes.append(process_name)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def get_active_window_rect():
    os_name = platform.system()

    if os_name == "Windows":
        return _get_windows_rect()
    elif os_name == "Darwin":
        return _get_macos_rect_native()
    else:
        print("Unsupported OS")
        return None

def _get_windows_rect():
    import ctypes
    from ctypes import wintypes

    # 1. Handle High DPI (scaling) so coordinates are accurate
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    # 2. Get the handle of the active window
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return None

    # 3. Get the coordinates
    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))

    x = rect.left
    y = rect.top
    w = rect.right - rect.left
    h = rect.bottom - rect.top

    return (x, y, w, h)

def _get_macos_rect_native():
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID
        )
        from Cocoa import NSWorkspace
    except ImportError:
        print("Error: Library missing.")
        print("Please run: pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa")
        return None

    # 1. Get the PID of the active application
    workspace = NSWorkspace.sharedWorkspace()
    active_app = workspace.frontmostApplication()

    if not active_app:
        return None

    active_pid = active_app.processIdentifier()

    # 2. Ask Quartz for a list of all on-screen windows
    # This usually bypasses Accessibility permissions because we ask for "All"
    # and filter manually, rather than asking for "Targeted" info.
    options = kCGWindowListOptionOnScreenOnly
    window_list = CGWindowListCopyWindowInfo(options, kCGNullWindowID)

    # 3. Filter the list to find the main window of the active PID
    for window in window_list:
        pid = window.get('kCGWindowOwnerPID')

        # Check if this window belongs to the active app
        if pid == active_pid:
            # We filter out overlays (layer > 0) to find the main window
            # Standard windows happen at Layer 0
            if window.get('kCGWindowLayer', 0) == 0:
                bounds = window.get('kCGWindowBounds')
                if bounds:
                    x = int(bounds['X'])
                    y = int(bounds['Y'])
                    w = int(bounds['Width'])
                    h = int(bounds['Height'])

                    # Filtering out weird tiny windows (tooltips, invisible helper windows)
                    if w > 10 and h > 10:
                        return (x, y, w, h)

    print("Active app found, but no valid window detected.")
    return None


# ==========================================
# 2. INPUT MONITOR (Threaded)
# ==========================================
class InputMonitor:
    def __init__(self):
        self.counter = 0
        self._lock = threading.Lock()

        # Start Listeners (Non-blocking)
        self.kb_listener = keyboard.Listener(on_press=self._on_event)
        self.mouse_listener = mouse.Listener(on_click=self._on_event, on_scroll=self._on_event)

        self.kb_listener.start()
        self.mouse_listener.start()

    def _on_event(self, *args):
        # We don't care WHAT key was pressed, just THAT it was pressed
        with self._lock:
            self.counter += 1

    def get_and_reset_count(self):
        """Returns the number of events since last check and resets to 0"""
        with self._lock:
            count = self.counter
            self.counter = 0
        return count
