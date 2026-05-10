"""Global hotkey manager for dev mode.

Registers hotkeys through an existing :class:`InputMonitor` so they share
a single pynput keyboard listener — avoids the double-hook conflict that
breaks ``GlobalHotKeys`` in PyInstaller windowed builds on Windows.

Default bindings:
    Ctrl+Shift+R — toggle recording on/off
    Ctrl+Shift+N — advance to the next classification state
"""

import logging
from typing import Callable, Optional

from src.core.input_monitor import InputMonitor

logger = logging.getLogger(__name__)

DEFAULT_TOGGLE_HOTKEY = "<ctrl>+<shift>+r"
DEFAULT_NEXT_STATE_HOTKEY = "<ctrl>+<shift>+n"


class HotkeyManager:
    """Registers global hotkeys via an :class:`InputMonitor`.

    Callbacks are invoked from the pynput listener thread; use
    ``root.after(0, callback)`` inside the callbacks if you need to touch
    the Tkinter main thread.

    Parameters
    ----------
    input_monitor:
        The running InputMonitor whose keyboard listener will handle hotkeys.
    on_toggle_recording:
        Called when the toggle-recording hotkey fires.
    on_next_state:
        Called when the next-state hotkey fires.
    toggle_hotkey:
        pynput hotkey string for toggling recording.
    next_state_hotkey:
        pynput hotkey string for advancing the classification state.
    """

    def __init__(
        self,
        input_monitor: Optional[InputMonitor],
        on_toggle_recording: Callable[[], None],
        on_next_state: Callable[[], None],
        toggle_hotkey: str = DEFAULT_TOGGLE_HOTKEY,
        next_state_hotkey: str = DEFAULT_NEXT_STATE_HOTKEY,
    ) -> None:
        self._input_monitor = input_monitor
        self._on_toggle = on_toggle_recording
        self._on_next_state = on_next_state
        self._toggle_hotkey = toggle_hotkey
        self._next_state_hotkey = next_state_hotkey
        self._started = False

        logger.info(
            "HotkeyManager configured — toggle=%s  next_state=%s",
            toggle_hotkey,
            next_state_hotkey,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register hotkeys on the InputMonitor's listener."""
        if self._started:
            logger.warning("start() called while already running — ignored")
            return

        if self._input_monitor is None:
            logger.warning("No InputMonitor available — hotkeys disabled")
            return

        self._input_monitor.add_hotkey(self._toggle_hotkey, self._handle_toggle)
        self._input_monitor.add_hotkey(self._next_state_hotkey, self._handle_next_state)
        self._started = True
        logger.info("Global hotkeys registered")

    def stop(self) -> None:
        """No-op — hotkeys are cleaned up when InputMonitor stops."""
        self._started = False
        logger.info("HotkeyManager stopped")

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_toggle(self) -> None:
        logger.debug("Hotkey fired: toggle recording")
        try:
            self._on_toggle()
        except Exception:
            logger.exception("Exception in on_toggle_recording callback")

    def _handle_next_state(self) -> None:
        logger.debug("Hotkey fired: next state")
        try:
            self._on_next_state()
        except Exception:
            logger.exception("Exception in on_next_state callback")
