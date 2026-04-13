"""Dev mode UI page — recording controls and classification state selector."""

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

from src.core.theme import ModernTheme as T

logger = logging.getLogger(__name__)


class DevPage(tk.Frame):
    """Dev mode page with recording controls, state selector, and status panel.

    Parameters
    ----------
    parent:
        Parent Tkinter widget.
    states:
        Ordered list of classification state labels shown in the combobox.
    on_start:
        Called when the user clicks *Start Recording*.
    on_stop:
        Called when the user clicks *Stop*.
    on_save:
        Called when the user clicks *Save & Upload*.
    on_state_change:
        Called with the new label string whenever the state changes
        (via combobox selection *or* :meth:`set_state` / :meth:`next_state`).
    """

    def __init__(
        self,
        parent: tk.Widget,
        states: Sequence[str],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_save: Callable[[], None],
        on_state_change: Callable[[str], None],
        on_tester_change: Callable[[str], None] = lambda s: None,
        on_delay_change: Callable[[float], None] = lambda f: None,
        initial_delay: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=T.BACKGROUND_DARK, **kwargs)
        self._states = list(states)
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_save = on_save
        self._on_state_change = on_state_change
        self._on_tester_change = on_tester_change
        self._on_delay_change = on_delay_change
        self._initial_delay = initial_delay

        self._build_ui()
        logger.debug("DevPage initialised with states: %s", self._states)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Title ----
        tk.Label(
            self,
            text="Dev Mode \u2014 ML Data Collection",
            bg=T.BACKGROUND_DARK,
            fg=T.TEXT_PRIMARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_XLARGE, "bold"),
        ).pack(pady=(28, 2))

        tk.Label(
            self,
            text=(
                "Hotkeys: \u25b6 Ctrl+Shift+R \u2014 toggle recording "
                "\u25b6 Ctrl+Shift+N \u2014 next state"
            ),
            bg=T.BACKGROUND_DARK,
            fg=T.TEXT_SECONDARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_SMALL),
        ).pack(pady=(0, 20))

        # ---- Tester name ----
        tester_row = tk.Frame(self, bg=T.BACKGROUND_DARK)
        tester_row.pack(pady=6)

        tk.Label(
            tester_row,
            text="Tester Name:",
            bg=T.BACKGROUND_DARK,
            fg=T.TEXT_PRIMARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_MEDIUM),
        ).pack(side="left", padx=(0, 10))

        self._tester_var = tk.StringVar(value="")
        self._tester_entry = tk.Entry(
            tester_row,
            textvariable=self._tester_var,
            width=20,
            bg=T.SURFACE,
            fg=T.TEXT_PRIMARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_MEDIUM),
            insertbackground=T.TEXT_PRIMARY,
        )
        self._tester_entry.pack(side="left")

        # ---- First-capture delay ----
        delay_row = tk.Frame(self, bg=T.BACKGROUND_DARK)
        delay_row.pack(pady=6)

        tk.Label(
            delay_row,
            text="Delay before first capture (s):",
            bg=T.BACKGROUND_DARK,
            fg=T.TEXT_PRIMARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_MEDIUM),
        ).pack(side="left", padx=(0, 10))

        self._delay_var = tk.DoubleVar(value=self._initial_delay)
        self._delay_spinner = tk.Spinbox(
            delay_row,
            from_=0.0,
            to=30.0,
            increment=0.5,
            textvariable=self._delay_var,
            width=6,
            command=self._on_delay_spin,
            bg=T.SURFACE,
            fg=T.TEXT_PRIMARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_MEDIUM),
            buttonbackground=T.SURFACE_ELEVATED,
        )
        self._delay_spinner.pack(side="left")

        # ---- State selector ----
        state_row = tk.Frame(self, bg=T.BACKGROUND_DARK)
        state_row.pack(pady=6)

        tk.Label(
            state_row,
            text="Classification state:",
            bg=T.BACKGROUND_DARK,
            fg=T.TEXT_PRIMARY,
            font=(T.FONT_FAMILY, T.FONT_SIZE_MEDIUM),
        ).pack(side="left", padx=(0, 10))

        self._state_var = tk.StringVar(value=self._states[0] if self._states else "")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Dev.TCombobox",
            fieldbackground=T.SURFACE,
            background=T.SURFACE,
            foreground=T.TEXT_PRIMARY,
            selectbackground=T.PRIMARY,
            selectforeground=T.TEXT_PRIMARY,
            arrowcolor=T.TEXT_SECONDARY,
        )
        self._combo = ttk.Combobox(
            state_row,
            textvariable=self._state_var,
            values=self._states,
            state="readonly",
            width=22,
            style="Dev.TCombobox",
        )
        self._combo.pack(side="left")
        self._combo.bind("<<ComboboxSelected>>", self._on_combo_change)

        # ---- Control buttons ----
        btn_row = tk.Frame(self, bg=T.BACKGROUND_DARK)
        btn_row.pack(pady=18)

        self._start_btn = tk.Button(
            btn_row,
            text="\u25b6  Start Recording",
            command=self._handle_start,
            **T.button_style("success"),
            width=18,
        )
        self._start_btn.pack(side="left", padx=8)

        self._stop_btn = tk.Button(
            btn_row,
            text="\u23f9  Stop",
            command=self._on_stop,
            bg=T.ERROR,
            fg=T.TEXT_PRIMARY,
            activebackground="#dc2626",
            activeforeground=T.TEXT_PRIMARY,
            bd=0,
            relief="flat",
            font=(T.FONT_FAMILY, T.FONT_SIZE_NORMAL, "bold"),
            padx=T.PADDING_LARGE,
            pady=T.PADDING_MEDIUM,
            cursor="hand2",
            width=12,
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=8)

        self._save_btn = tk.Button(
            btn_row,
            text="\u2601  Flush Now",
            command=self._on_save,
            **T.button_style("primary"),
            width=16,
            state="disabled",
        )
        self._save_btn.pack(side="left", padx=8)

        # ---- Status panel ----
        panel = tk.Frame(
            self,
            bg=T.SURFACE,
            bd=0,
            highlightthickness=1,
            highlightbackground=T.BORDER,
        )
        panel.pack(fill="x", padx=40, pady=12)

        def _row(label_text: str) -> tk.Label:
            row = tk.Frame(panel, bg=T.SURFACE)
            row.pack(fill="x", padx=14, pady=4)
            tk.Label(
                row,
                text=label_text,
                bg=T.SURFACE,
                fg=T.TEXT_SECONDARY,
                font=(T.FONT_FAMILY, T.FONT_SIZE_NORMAL),
                anchor="w",
                width=22,
            ).pack(side="left")
            val = tk.Label(
                row,
                text="\u2014",
                bg=T.SURFACE,
                fg=T.TEXT_PRIMARY,
                font=(T.FONT_FAMILY, T.FONT_SIZE_NORMAL),
                anchor="w",
            )
            val.pack(side="left", fill="x", expand=True)
            return val

        self._status_val = _row("Status:")
        self._samples_val = _row("Buffer:")
        self._upload_val = _row("Last upload:")
        self._batches_val = _row("Batches uploaded:")
        self._pending_val = _row("Pending in queue:")

        # Initial state
        self._status_val.config(text="Idle", fg=T.TEXT_SECONDARY)

    # ------------------------------------------------------------------
    # Public API (called by controller)
    # ------------------------------------------------------------------

    def set_recording(self, recording: bool) -> None:
        """Update button states and status label to match recording state."""
        if recording:
            self._start_btn.config(state="disabled")
            self._stop_btn.config(state="normal")
            self._save_btn.config(state="normal")
            self._tester_entry.config(state="disabled")
            self._delay_spinner.config(state="disabled")
            self._status_val.config(text="Recording\u2026", fg=T.STATUS_ONLINE)
            logger.debug("DevPage: recording=True")
        else:
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._save_btn.config(state="normal")
            self._tester_entry.config(state="normal")
            self._delay_spinner.config(state="normal")
            self._status_val.config(text="Stopped", fg=T.TEXT_SECONDARY)
            logger.debug("DevPage: recording=False")

    def update_sample_count(self, count: int, max_count: int) -> None:
        """Refresh the buffer progress display."""
        secs = count * 36 // max(1, max_count) * 5  # rough seconds
        self._samples_val.config(text=f"{count} / {max_count} samples")

    def set_upload_status(self, message: str, success: bool = True) -> None:
        """Update the last-upload status row."""
        color = T.STATUS_ONLINE if success else T.STATUS_OFFLINE
        self._upload_val.config(text=message, fg=color)
        logger.debug("DevPage upload status: %s (success=%s)", message, success)

    def update_upload_stats(
        self,
        uploaded: int,
        failed: int,
        pending: int,
        samples_uploaded: int,
        last_error: str,
    ) -> None:
        """Refresh the batch upload statistics display."""
        self._batches_val.config(
            text=f"{uploaded} ok, {failed} failed ({samples_uploaded} samples)",
        )
        self._pending_val.config(text=str(pending))
        if last_error:
            self._upload_val.config(text=f"Error: {last_error}", fg=T.STATUS_OFFLINE)

    def set_state(self, label: str) -> None:
        """Programmatically select a state (e.g. from a hotkey)."""
        if label not in self._states:
            logger.warning("set_state called with unknown label %r \u2014 ignored", label)
            return
        self._state_var.set(label)
        self._on_state_change(label)
        logger.info("State set programmatically: %r", label)

    def next_state(self) -> str:
        """Advance to the next state in the list and return its label."""
        current = self._state_var.get()
        try:
            idx = self._states.index(current)
        except ValueError:
            idx = -1
        next_label = self._states[(idx + 1) % len(self._states)]
        self.set_state(next_label)
        return next_label

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_start(self) -> None:
        """Validate tester name, then delegate to the start callback."""
        tester = self._tester_var.get().strip()
        if not tester:
            self.set_upload_status("Tester name is required!", success=False)
            return
        self._on_tester_change(tester)
        self._on_start()

    def _on_delay_spin(self) -> None:
        try:
            val = self._delay_var.get()
        except tk.TclError:
            return
        self._on_delay_change(val)

    def _on_combo_change(self, _event=None) -> None:
        label = self._state_var.get()
        logger.info("State selected via combobox: %r", label)
        self._on_state_change(label)
