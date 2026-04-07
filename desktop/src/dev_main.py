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
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

from src.dev.hotkeys import HotkeyManager
from src.dev.recorder import DataRecorder
from src.dev.uploader import YaDiskUploader
from src.system_monitor import InputMonitor
from src.ui.dev_page import DevPage
from src.ui.theme import ModernTheme as T

_DEFAULT_STATES = ["Not Gaming", "Gaming"]
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
        type=int,
        default=5,
        metavar="SECONDS",
        help="Sample capture interval in seconds",
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

        # ---- Recorder ----
        self._recorder = DataRecorder(
            input_monitor=self._input_monitor,
            window_seconds=args.window,
            sample_interval=args.interval,
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

    # ------------------------------------------------------------------
    # Save & upload
    # ------------------------------------------------------------------

    def _save_and_upload(self) -> None:
        samples = self._recorder.get_samples()
        if not samples:
            logger.warning("Save requested but buffer is empty — nothing to do")
            self._page.set_upload_status("Buffer is empty — nothing to save", success=False)
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label_slug = self._recorder.current_label.replace(" ", "_").lower()
        filename = f"session_{ts}_{label_slug}.zip"
        out_path = self._args.output_dir / filename

        try:
            self._recorder.export_zip(out_path)
        except Exception as exc:
            logger.exception("Export failed")
            self._page.set_upload_status(f"Export failed: {exc}", success=False)
            return

        if self._uploader is not None:
            self._page.set_upload_status("Uploading…", success=True)
            Thread(
                target=self._do_upload,
                args=(out_path,),
                name="yadisk-upload",
                daemon=True,
            ).start()
        else:
            self._page.set_upload_status(f"Saved locally: {out_path.name}", success=True)

    def _do_upload(self, path: Path) -> None:
        """Run in a daemon thread — post result back to main thread."""
        try:
            remote = self._uploader.upload(path)
            self.after(0, self._page.set_upload_status, f"Uploaded: {remote}", True)
        except Exception as exc:
            logger.exception("Upload failed")
            self.after(0, self._page.set_upload_status, f"Upload failed: {exc}", False)

    # ------------------------------------------------------------------
    # UI refresh
    # ------------------------------------------------------------------

    def _tick_ui(self) -> None:
        """Refresh sample counter every second."""
        self._page.update_sample_count(
            self._recorder.sample_count,
            self._recorder.max_samples,
        )
        self.after(1000, self._tick_ui)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        logger.info("Closing DevApp")
        self._recorder.stop()
        self._hotkeys.stop()
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
