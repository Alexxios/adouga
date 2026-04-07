import logging
import platform
import time
import threading
from collections import deque
from typing import Optional

import psutil
from pynput import keyboard, mouse

logger = logging.getLogger(__name__)

# ==========================================
# 1. PROCESS / WINDOW HELPERS
# ==========================================

def get_active_processes(pids=[]):
    processes = []
    for proc in psutil.process_iter():
        try:
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


# ==========================================
# 2. INPUT MONITOR (Threaded)
# ==========================================

_FLICK_MIN_MAG = 10        # pixels — minimum movement magnitude to count as a flick
_FLICK_BUFFER = 300        # how many flick vectors to retain
_EVENT_BUFFER_SECONDS = 180  # how far back to keep raw input events (3 minutes)

# (window_seconds, label) — used for heatmap aggregation
_HEATMAP_INTERVALS: list[tuple[int, str]] = [
    (1,   "1s"),
    (5,   "5s"),
    (15,  "15s"),
    (30,  "30s"),
    (60,  "1m"),
    (180, "3m"),
]


def _key_name(key) -> str:
    """Normalise a pynput key to a short, readable string."""
    try:
        char = key.char
        if char is not None:
            return char.lower()
    except AttributeError:
        pass
    # Special key: pynput gives e.g. "Key.space", "Key.ctrl_l"
    return str(key).replace("Key.", "").lower()


class InputMonitor:
    """Tracks keyboard and mouse input in background threads.

    Exposes:
    - aggregate event count (``get_and_reset_count``)
    - raw flick vectors (``get_flicks``)
    - timestamped event sequence for the last 3 min (``get_input_sequence``)
    - per-key frequency dicts over multiple intervals (``get_key_heatmaps``)
    """

    def __init__(self) -> None:
        self.counter: int = 0
        self._lock = threading.Lock()

        # Flick vectors
        self._flicks: deque = deque(maxlen=_FLICK_BUFFER)
        self._last_pos: tuple | None = None

        # Timestamped raw event log — each entry is a plain dict:
        # {"timestamp": float, "type": str, "value": str}
        self._events: deque = deque()

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
            on_move=self._on_move,
        )
        self.kb_listener.start()
        self.mouse_listener.start()
        logger.debug("InputMonitor started")

    # ------------------------------------------------------------------
    # Listener callbacks
    # ------------------------------------------------------------------

    def _on_key_press(self, key) -> None:
        name = _key_name(key)
        with self._lock:
            self.counter += 1
            self._record_event("key_press", name)

    def _on_mouse_click(self, x, y, button, pressed) -> None:
        if not pressed:
            return
        # "Button.left" → "left"
        btn_name = str(button).split(".")[-1]
        with self._lock:
            self.counter += 1
            self._record_event("mouse_click", btn_name)

    def _on_mouse_scroll(self, x, y, dx, dy) -> None:
        direction = "scroll_up" if dy > 0 else "scroll_down"
        with self._lock:
            self.counter += 1
            self._record_event("mouse_scroll", direction)

    def _on_move(self, x, y) -> None:
        if self._last_pos is not None:
            ddx = x - self._last_pos[0]
            ddy = y - self._last_pos[1]
            if ddx * ddx + ddy * ddy >= _FLICK_MIN_MAG ** 2:
                with self._lock:
                    self._flicks.append((ddx, ddy))
        self._last_pos = (x, y)

    def _record_event(self, event_type: str, value: str) -> None:
        """Append a timestamped event and prune entries older than the buffer window.

        Must be called under ``self._lock``.
        """
        now = time.time()
        self._events.append({"timestamp": now, "type": event_type, "value": value})
        cutoff = now - _EVENT_BUFFER_SECONDS
        while self._events and self._events[0]["timestamp"] < cutoff:
            self._events.popleft()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_and_reset_count(self) -> int:
        """Return the number of input events since the last call and reset to 0."""
        with self._lock:
            count = self.counter
            self.counter = 0
        return count

    def get_flicks(self) -> list:
        """Return a snapshot of recent mouse-flick vectors as [(dx, dy), ...]."""
        with self._lock:
            return list(self._flicks)

    def get_input_sequence(self, window_seconds: float = _EVENT_BUFFER_SECONDS) -> list:
        """Return all raw input events from the last *window_seconds*.

        Each event is a dict: ``{"timestamp": float, "type": str, "value": str}``.
        Types: ``"key_press"``, ``"mouse_click"``, ``"mouse_scroll"``.
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            return [e for e in self._events if e["timestamp"] >= cutoff]

    def get_key_heatmaps(self) -> dict[str, dict[str, int]]:
        """Return per-key press frequency dicts for each standard time interval.

        Returns a dict keyed by interval label (``"1s"``, ``"5s"``, ``"15s"``,
        ``"30s"``, ``"1m"``, ``"3m"``), each mapping key/button names to counts.

        Example::

            {
              "1s":  {"a": 2, "space": 1},
              "5s":  {"a": 8, "space": 3, "left": 1},
              ...
            }
        """
        now = time.time()
        with self._lock:
            events = list(self._events)

        result: dict[str, dict[str, int]] = {}
        for secs, label in _HEATMAP_INTERVALS:
            cutoff = now - secs
            counts: dict[str, int] = {}
            for ev in events:
                if ev["timestamp"] >= cutoff:
                    counts[ev["value"]] = counts.get(ev["value"], 0) + 1
            result[label] = counts

        logger.debug(
            "get_key_heatmaps: total_events=%d intervals=%s",
            len(events),
            list(result.keys()),
        )
        return result

    def stop(self) -> None:
        """Stop both pynput listener threads."""
        self.kb_listener.stop()
        self.mouse_listener.stop()
        logger.debug("InputMonitor stopped")


# ==========================================
# 3. HARDWARE MONITOR (Background sampler)
# ==========================================

_HW_SAMPLE_INTERVAL = 1    # seconds between hardware readings
_HW_BUFFER_SECONDS = 180   # rolling window (3 minutes)


def _sample_gpu_once() -> Optional[dict]:
    """Return a single GPU reading dict, or ``None`` if unavailable."""
    try:
        import GPUtil  # type: ignore[import]
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            return {
                "load_percent": round(g.load * 100, 1),
                "memory_used_gb": round(g.memoryUsed / 1024, 3),
                "memory_total_gb": round(g.memoryTotal / 1024, 3),
                "temperature_c": g.temperature,
            }
    except ImportError:
        pass
    except Exception:
        logger.debug("GPUtil query failed", exc_info=True)
    return None


class HardwareMonitor:
    """Samples CPU, RAM, GPU and disk I/O once per second in a daemon thread,
    keeping a rolling 3-minute history for each metric.

    Each history entry is a plain dict with a ``"timestamp"`` key plus the
    metric-specific fields listed below.

    CPU entry keys:  ``percent``, ``freq_ghz``
    RAM entry keys:  ``percent``, ``used_gb``, ``total_gb``
    GPU entry keys:  ``load_percent``, ``memory_used_gb``, ``memory_total_gb``,
                     ``temperature_c``  (omitted entirely when no GPU found)
    Disk entry keys: ``read_bps``, ``write_bps``

    Usage::

        hw = HardwareMonitor()
        hw.start()
        ...
        cpu = hw.get_cpu_history()   # list of dicts
        hw.stop()
    """

    def __init__(
        self,
        sample_interval: int = _HW_SAMPLE_INTERVAL,
        buffer_seconds: int = _HW_BUFFER_SECONDS,
    ) -> None:
        self._interval = sample_interval
        max_entries = max(1, buffer_seconds // sample_interval)

        self._cpu_hist: deque = deque(maxlen=max_entries)
        self._ram_hist: deque = deque(maxlen=max_entries)
        self._gpu_hist: deque = deque(maxlen=max_entries)
        self._disk_hist: deque = deque(maxlen=max_entries)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_disk_io = None

        logger.info(
            "HardwareMonitor initialised — interval=%ds buffer=%ds max_entries=%d",
            sample_interval, buffer_seconds, max_entries,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background sampling thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("HardwareMonitor.start() called while already running")
            return
        # Warm up cpu_percent (first call always returns 0.0)
        psutil.cpu_percent(interval=None)
        self._last_disk_io = psutil.disk_io_counters()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="hw-monitor", daemon=True
        )
        self._thread.start()
        logger.info("HardwareMonitor started")

    def stop(self) -> None:
        """Signal the sampling thread to stop."""
        self._stop_event.set()
        logger.info("HardwareMonitor stop requested")

    # ------------------------------------------------------------------
    # History accessors
    # ------------------------------------------------------------------

    def get_cpu_history(self) -> list:
        """Return a snapshot of the CPU reading history."""
        with self._lock:
            return list(self._cpu_hist)

    def get_ram_history(self) -> list:
        """Return a snapshot of the RAM reading history."""
        with self._lock:
            return list(self._ram_hist)

    def get_gpu_history(self) -> list:
        """Return a snapshot of the GPU reading history (empty if no GPU)."""
        with self._lock:
            return list(self._gpu_hist)

    def get_disk_history(self) -> list:
        """Return a snapshot of the disk I/O reading history."""
        with self._lock:
            return list(self._disk_hist)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample()
            except Exception:
                logger.exception("HardwareMonitor._sample() raised unexpectedly")
            self._stop_event.wait(self._interval)
        logger.debug("HardwareMonitor thread exited")

    def _sample(self) -> None:
        now = time.time()

        # CPU
        cpu_pct = psutil.cpu_percent(interval=None)
        freq = psutil.cpu_freq()
        cpu_entry = {
            "timestamp": now,
            "percent": cpu_pct,
            "freq_ghz": round(freq.current / 1000, 3) if freq else None,
        }

        # RAM
        mem = psutil.virtual_memory()
        ram_entry = {
            "timestamp": now,
            "percent": mem.percent,
            "used_gb": round(mem.used / 1e9, 3),
            "total_gb": round(mem.total / 1e9, 3),
        }

        # GPU (best-effort)
        gpu_reading = _sample_gpu_once()
        if gpu_reading:
            gpu_reading["timestamp"] = now

        # Disk I/O delta
        disk_entry = self._disk_delta(now)

        with self._lock:
            self._cpu_hist.append(cpu_entry)
            self._ram_hist.append(ram_entry)
            if gpu_reading:
                self._gpu_hist.append(gpu_reading)
            if disk_entry:
                self._disk_hist.append(disk_entry)

        logger.debug(
            "HW sample — cpu=%.1f%% ram=%.1f%% gpu=%s disk=%s",
            cpu_pct, mem.percent,
            f"{gpu_reading['load_percent']}%" if gpu_reading else "n/a",
            f"r={disk_entry['read_bps']:.0f} w={disk_entry['write_bps']:.0f}"
            if disk_entry else "n/a",
        )

    def _disk_delta(self, now: float) -> Optional[dict]:
        """Compute bytes/sec since the previous sample."""
        try:
            current = psutil.disk_io_counters()
            if current is None or self._last_disk_io is None:
                self._last_disk_io = current
                return None
            dt = self._interval or 1
            entry = {
                "timestamp": now,
                "read_bps": (current.read_bytes - self._last_disk_io.read_bytes) / dt,
                "write_bps": (current.write_bytes - self._last_disk_io.write_bytes) / dt,
            }
            self._last_disk_io = current
            return entry
        except Exception:
            logger.debug("disk_io_counters() failed", exc_info=True)
            return None
