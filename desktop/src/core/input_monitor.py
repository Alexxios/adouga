"""Threaded input monitor — keyboard and mouse event tracking."""

import logging
import time
import threading
from collections import deque

from pynput import keyboard, mouse

logger = logging.getLogger(__name__)

_FLICK_MIN_MAG = 10        # pixels — minimum movement magnitude to count as a flick
_FLICK_BUFFER = 300        # how many flick vectors to retain
_EVENT_BUFFER_SECONDS = 180  # how far back to keep raw input events (3 minutes)

# Keys whose per-sample counts feed the ML tabular branch (gaming-relevant).
_GAMING_KEYS: tuple[str, ...] = (
    "w", "a", "s", "d", "space", "shift", "left", "right",
)

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

        # Per-sample drainable counts (separate from `counter` which UI consumes).
        self._key_press_count: int = 0
        self._mouse_click_count: int = 0
        self._mouse_scroll_count: int = 0
        self._mouse_move_count: int = 0
        self._gaming_key_counts: dict[str, int] = {k: 0 for k in _GAMING_KEYS}

        # Flick vectors
        self._flicks: deque = deque(maxlen=_FLICK_BUFFER)
        self._last_pos: tuple | None = None

        # Timestamped raw event log — each entry is a plain dict:
        # {"timestamp": float, "type": str, "value": str}
        self._events: deque = deque()

        # Hotkeys — registered via add_hotkey(); checked inside the single
        # keyboard listener so we never need a second pynput hook.
        self._hotkeys: list[keyboard.HotKey] = []

        self.kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
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
            self._key_press_count += 1
            if name in self._gaming_key_counts:
                self._gaming_key_counts[name] += 1
            self._record_event("key_press", name)
        # Forward to hotkeys (uses canonical form from the listener)
        for hk in self._hotkeys:
            hk.press(self.kb_listener.canonical(key))

    def _on_key_release(self, key) -> None:
        for hk in self._hotkeys:
            hk.release(self.kb_listener.canonical(key))

    def _on_mouse_click(self, x, y, button, pressed) -> None:
        if not pressed:
            return
        # "Button.left" -> "left"
        btn_name = str(button).split(".")[-1]
        with self._lock:
            self.counter += 1
            self._mouse_click_count += 1
            self._record_event("mouse_click", btn_name)

    def _on_mouse_scroll(self, x, y, dx, dy) -> None:
        direction = "scroll_up" if dy > 0 else "scroll_down"
        with self._lock:
            self.counter += 1
            self._mouse_scroll_count += 1
            self._record_event("mouse_scroll", direction)

    def _on_move(self, x, y) -> None:
        if self._last_pos is not None:
            ddx = x - self._last_pos[0]
            ddy = y - self._last_pos[1]
            if ddx * ddx + ddy * ddy >= _FLICK_MIN_MAG ** 2:
                with self._lock:
                    self._flicks.append((ddx, ddy))
                    self._mouse_move_count += 1
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

    def get_and_reset_flicks(self) -> list:
        """Return all buffered flick vectors and clear the buffer.

        Used by the recorder so each captured sample carries only the flicks
        observed since the previous sample (avoiding the saturation that the
        bounded ``_FLICK_BUFFER`` deque otherwise causes).
        """
        with self._lock:
            flicks = list(self._flicks)
            self._flicks.clear()
        return flicks

    def get_and_reset_input_aggregates(self) -> dict:
        """Return per-sample input aggregates and reset all per-type counters.

        Returned shape::

            {
              "key_press_count":   int,
              "mouse_click_count": int,
              "mouse_scroll_count":int,
              "mouse_move_count":  int,
              "total_count":       int,   # sum of the four above
              "gaming_keys": {"w": int, "a": int, ..., "right": int},
            }

        Independent of ``get_and_reset_count()`` — both can be drained per
        sample without interfering.
        """
        with self._lock:
            kp = self._key_press_count
            mc = self._mouse_click_count
            ms = self._mouse_scroll_count
            mm = self._mouse_move_count
            gk = dict(self._gaming_key_counts)
            self._key_press_count = 0
            self._mouse_click_count = 0
            self._mouse_scroll_count = 0
            self._mouse_move_count = 0
            for k in self._gaming_key_counts:
                self._gaming_key_counts[k] = 0
        return {
            "key_press_count": kp,
            "mouse_click_count": mc,
            "mouse_scroll_count": ms,
            "mouse_move_count": mm,
            "total_count": kp + mc + ms + mm,
            "gaming_keys": gk,
        }

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

    def add_hotkey(self, hotkey_str: str, callback) -> None:
        """Register a global hotkey handled by the existing keyboard listener.

        Uses :class:`pynput.keyboard.HotKey` so no second listener is needed.

        Parameters
        ----------
        hotkey_str:
            pynput hotkey string, e.g. ``"<ctrl>+<shift>+r"``.
        callback:
            Zero-argument callable invoked when the hotkey fires.
        """
        hk = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), callback)
        self._hotkeys.append(hk)
        logger.info("Hotkey registered: %s", hotkey_str)

    def stop(self) -> None:
        """Stop both pynput listener threads."""
        self.kb_listener.stop()
        self.mouse_listener.stop()
        logger.debug("InputMonitor stopped")
