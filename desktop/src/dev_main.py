"""Dev mode entry point — ML training data collection tool.

Run from the ``desktop/`` directory:

    python src/dev_main.py [options]

Options
-------
--states STR [STR ...]   Classification labels (default: "Not Gaming" "Gaming")
--interval INT           Seconds between samples (default: 5)
--window INT             Rolling window in seconds (default: 180 = 3 min)
--output-dir PATH        Local directory for exported ZIPs
--no-upload              Skip Yandex Disk upload; save locally only
--log-level LEVEL        DEBUG / INFO / WARNING / ERROR (default: INFO)
"""

import argparse
import logging
import sys
import tkinter as tk
from pathlib import Path

from src.dev.batch_uploader import BatchUploader
from src.dev.hotkeys import HotkeyManager
from src.dev.recorder import DataRecorder
from src.dev.uploader import YaDiskUploader
from src.system_monitor import HardwareMonitor, InputMonitor
from src.ui.dev_page import DevPage
from src.ui.theme import ModernTheme as T

_DEFAULT_STATES = ["Idle", "Not Gaming", "Gaming"]
_DEFAULT_OUTPUT_DIR = Path.home() / "adouga_ml_data"


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dev_main",
        description="Adouga Dev Mode — ML data collection tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--states",
        nargs="+",
        default=_DEFAULT_STATES,
        metavar="STATE",
        help="Classification state labels",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Sample capture interval in seconds",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Delay in seconds before the first capture after Start",
    )
    p.add_argument(
        "--window",
        type=int,
        default=180,
        metavar="SECONDS",
        help="Rolling buffer window in seconds (3 min = 180)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help="Local directory where ZIP archives are saved",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip Yandex Disk upload; save locally only",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p.parse_args()


logger = logging.getLogger(__name__)


class DevApp(tk.Tk):
    """Dev mode application — wires together recorder, uploader, hotkeys, and UI.

    Parameters
    ----------
    args:
        Parsed CLI arguments (from :func:`_parse_args`).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self.title("Adouga Dev — ML Data Collection")
        self.geometry("720x500")
        self.configure(bg=T.BACKGROUND_DARK)
        self.resizable(False, False)

        logger.info("Initialising DevApp (states=%s)", args.states)

        # ---- Input monitor ----
        try:
            self._input_monitor = InputMonitor()
            logger.info("InputMonitor started")
        except Exception:
            logger.exception("Failed to start InputMonitor")
            self._input_monitor = None

        # ---- Hardware monitor ----
        try:
            self._hw_monitor = HardwareMonitor()
            self._hw_monitor.start()
            logger.info("HardwareMonitor started")
        except Exception:
            logger.exception("Failed to start HardwareMonitor")
            self._hw_monitor = None

        # ---- Recorder ----
        self._recorder = DataRecorder(
            input_monitor=self._input_monitor,
            hardware_monitor=self._hw_monitor,
            window_seconds=args.window,
            sample_interval=args.interval,
            first_capture_delay=args.delay,
            batch_size=10,
            on_batch_ready=self._on_batch_ready,
        )

        # Set initial label from first state
        if args.states:
            self._recorder.current_label = args.states[0]

        # ---- Uploader ----
        if args.no_upload:
            self._uploader: YaDiskUploader | None = None
            logger.info("Upload disabled via --no-upload")
        else:
            try:
                self._uploader = YaDiskUploader()
                logger.info("YaDiskUploader ready")
            except EnvironmentError as exc:
                logger.warning("Upload unavailable: %s", exc)
                self._uploader = None

        # ---- Batch uploader ----
        self._batch_uploader = BatchUploader(
            uploader=self._uploader,
            output_dir=self._args.output_dir,
        )
        self._batch_uploader.start()

        # ---- Hotkey manager ----
        self._hotkeys = HotkeyManager(
            on_toggle_recording=self._toggle_recording,
            on_next_state=self._hotkey_next_state,
        )

        # ---- UI ----
        self._page = DevPage(
            self,
            states=args.states,
            on_start=self._start_recording,
            on_stop=self._stop_recording,
            on_save=self._save_and_upload,
            on_state_change=self._on_state_change,
            on_tester_change=self._on_tester_change,
            on_delay_change=self._on_delay_change,
            initial_delay=args.delay,
        )
        self._page.pack(fill="both", expand=True)

        # ---- Start subsystems ----
        self._hotkeys.start()
        self._tick_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        logger.info("DevApp ready — window open")

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        logger.info("User started recording")
        self._recorder.start()
        self._page.set_recording(True)

    def _stop_recording(self) -> None:
        logger.info("User stopped recording")
        self._recorder.stop()
        self._page.set_recording(False)

    def _toggle_recording(self) -> None:
        """Called from hotkey listener thread — delegate to main thread."""
        if self._recorder.is_recording:
            self.after(0, self._stop_recording)
        else:
            self.after(0, self._start_recording)

    def _hotkey_next_state(self) -> None:
        """Called from hotkey listener thread — delegate to main thread."""
        self.after(0, self._page.next_state)

    def _on_state_change(self, label: str) -> None:
        logger.info("Classification state changed: %r", label)
        self._recorder.current_label = label

    def _on_tester_change(self, tester_id: str) -> None:
        logger.info("Tester ID set: %r", tester_id)
        self._batch_uploader.tester_id = tester_id
        if self._uploader is not None:
            self._uploader.set_tester_id(tester_id)

    def _on_delay_change(self, delay: float) -> None:
        logger.info("First-capture delay changed: %.1f", delay)
        self._recorder.first_capture_delay = delay

    def _on_batch_ready(self, samples: list) -> None:
        """Called from recorder thread when a batch of samples is ready."""
        self._batch_uploader.enqueue(samples)

    # ------------------------------------------------------------------
    # Save & upload
    # ------------------------------------------------------------------

    def _save_and_upload(self) -> None:
        """Flush any remaining unflushed samples to the upload queue."""
        remaining = self._recorder.flush_unflushed()
        if remaining:
            self._batch_uploader.enqueue(remaining)
            self._page.set_upload_status(
                f"Flushed {len(remaining)} samples to upload queue", success=True,
            )
        else:
            self._page.set_upload_status("No unflushed samples", success=True)

    # ------------------------------------------------------------------
    # UI refresh
    # ------------------------------------------------------------------

    def _tick_ui(self) -> None:
        """Refresh sample counter and upload stats every second."""
        self._page.update_sample_count(
            self._recorder.sample_count,
            self._recorder.max_samples,
        )
        self._page.update_upload_stats(
            uploaded=self._batch_uploader.batches_uploaded,
            failed=self._batch_uploader.batches_failed,
            pending=self._batch_uploader.pending_count,
            samples_uploaded=self._batch_uploader.samples_uploaded,
            last_error=self._batch_uploader.last_error,
        )
        self.after(1000, self._tick_ui)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        logger.info("Closing DevApp")
        self._recorder.stop()
        # Flush remaining samples before shutting down
        remaining = self._recorder.flush_unflushed()
        if remaining:
            self._batch_uploader.enqueue(remaining)
        self._batch_uploader.stop()
        self._hotkeys.stop()
        if self._hw_monitor is not None:
            try:
                self._hw_monitor.stop()
            except Exception:
                logger.warning("HardwareMonitor.stop() raised", exc_info=True)
        if self._input_monitor is not None:
            try:
                self._input_monitor.stop()
            except Exception:
                logger.warning("InputMonitor.stop() raised", exc_info=True)
        self.destroy()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger.info("Starting Adouga Dev Mode")
    app = DevApp(args)
    app.mainloop()
    logger.info("DevApp exited cleanly")


if __name__ == "__main__":
    main()
