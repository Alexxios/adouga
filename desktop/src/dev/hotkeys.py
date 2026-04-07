"""Global hotkey manager for dev mode.

Registers system-wide hotkeys via pynput so they work even when the
application window is not focused.

Default bindings:
    Ctrl+Shift+R — toggle recording on/off
    Ctrl+Shift+N — advance to the next classification state
"""

import logging
from typing import Callable, Optional

from pynput import keyboard

logger = logging.getLogger(__name__)

DEFAULT_TOGGLE_HOTKEY = "<ctrl>+<shift>+r"
DEFAULT_NEXT_STATE_HOTKEY = "<ctrl>+<shift>+n"


class HotkeyManager:
    """Registers and manages global hotkeys for the dev application.

    Callbacks are invoked from the pynput listener thread; use
    ``root.after(0, callback)`` inside the callbacks if you need to touch
    the Tkinter main thread.

    Parameters
    ----------
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
        on_toggle_recording: Callable[[], None],
        on_next_state: Callable[[], None],
        toggle_hotkey: str = DEFAULT_TOGGLE_HOTKEY,
        next_state_hotkey: str = DEFAULT_NEXT_STATE_HOTKEY,
    ) -> None:
        self._on_toggle = on_toggle_recording
        self._on_next_state = on_next_state
        self._toggle_hotkey = toggle_hotkey
        self._next_state_hotkey = next_state_hotkey
        self._listener: Optional[keyboard.GlobalHotKeys] = None

        logger.info(
            "HotkeyManager configured — toggle=%s  next_state=%s",
            toggle_hotkey,
            next_state_hotkey,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register hotkeys and start listening in a daemon thread."""
        if self._listener is not None:
            logger.warning("start() called while already running — ignored")
            return

        self._listener = keyboard.GlobalHotKeys(
            {
                self._toggle_hotkey: self._handle_toggle,
                self._next_state_hotkey: self._handle_next_state,
            }
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("Global hotkeys registered and listening")

    def stop(self) -> None:
        """Unregister hotkeys and stop the listener thread."""
        if self._listener is None:
            logger.warning("stop() called while not running — ignored")
            return
        self._listener.stop()
        self._listener = None
        logger.info("Global hotkeys stopped")

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
