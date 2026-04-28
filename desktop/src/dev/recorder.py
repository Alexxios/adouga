"""Rolling-buffer data recorder for ML training data collection.

Captures screenshots and input metrics at a fixed interval, retaining the
last ``window_seconds`` of samples. Each sample carries:

* ``app_name`` / ``window_title`` — foreground app metadata
* ``hw_recent``       — last N=5 per-sample-aligned HW snapshots (oldest first)
* ``input_since_last``— input aggregates strictly since the previous tick
* ``screenshot``      — PIL image of the active window (excluded from JSON)

Call :meth:`export_zip` to package the current buffer as a labelled ZIP
archive ready for upload.
"""

import io
import json
import logging
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.core.models import DataSample
from src.core.sample_builders import (
    build_input_since_last,
    flatten_hw_latest,
)
from src.core.screenshot import take_screenshot
from src.core.window import get_active_window_info

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_SECONDS = 180
_DEFAULT_SAMPLE_INTERVAL = 1.0
_HW_RECENT_DEPTH = 5

class DataRecorder:
    """Continuously captures labelled samples into a fixed-size rolling buffer.

    Parameters
    ----------
    input_monitor:
        Object exposing ``get_and_reset_count``, ``get_flicks``,
        ``get_input_sequence``, ``get_key_heatmaps``.
    hardware_monitor:
        Optional :class:`src.system_monitor.HardwareMonitor`.  When provided,
        each sample includes the full CPU/RAM/GPU/disk rolling histories.
    window_seconds:
        Rolling buffer length in seconds (default 180 = 3 min).
    sample_interval:
        Seconds between captures (default 5 s).
    """

    def __init__(
        self,
        input_monitor,
        hardware_monitor=None,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        sample_interval: float = _DEFAULT_SAMPLE_INTERVAL,
        first_capture_delay: float = 0.0,
        batch_size: int = 10,
        on_batch_ready: Optional[Callable[[list], None]] = None,
    ) -> None:
        self._input_monitor = input_monitor
        self._hw = hardware_monitor
        self._window_seconds = window_seconds
        self._sample_interval = sample_interval
        self._max_samples = max(1, int(window_seconds / sample_interval))

        self._samples: deque = deque(maxlen=self._max_samples)
        self._hw_recent: deque = deque(maxlen=_HW_RECENT_DEPTH)
        self._current_label: str = ""
        self._recording: bool = False
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        # First-capture delay
        self._first_capture_delay = first_capture_delay

        # Batch auto-upload support
        self._batch_size = batch_size
        self._on_batch_ready = on_batch_ready
        self._unflushed: list = []
        self._unflushed_lock = threading.Lock()

        logger.info(
            "DataRecorder ready — window=%ds interval=%.2fs max_samples=%d hw=%s",
            window_seconds, sample_interval, self._max_samples,
            type(hardware_monitor).__name__ if hardware_monitor else "none",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_label(self) -> str:
        return self._current_label

    @current_label.setter
    def current_label(self, label: str) -> None:
        logger.info("Label changed: %r", label)
        self._current_label = label

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def max_samples(self) -> int:
        return self._max_samples

    @property
    def first_capture_delay(self) -> float:
        return self._first_capture_delay

    @first_capture_delay.setter
    def first_capture_delay(self, value: float) -> None:
        self._first_capture_delay = max(0.0, value)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin periodic capture.  No-op if already recording.

        Drains the input monitor's pre-launch accumulators so the very
        first sample only reflects events captured during the recording
        window, and clears the per-sample HW snapshot buffer.
        """
        if self._recording:
            logger.warning("start() called while already recording — ignored")
            return
        logger.info("Recording started")

        # Drain pre-launch accumulators on the input monitor — otherwise the
        # first sample is poisoned by everything that happened since the app
        # was launched.
        try:
            self._input_monitor.get_and_reset_count()
        except Exception:
            logger.debug("get_and_reset_count drain failed", exc_info=True)
        if hasattr(self._input_monitor, "get_and_reset_flicks"):
            try:
                self._input_monitor.get_and_reset_flicks()
            except Exception:
                logger.debug("get_and_reset_flicks drain failed", exc_info=True)
        if hasattr(self._input_monitor, "get_and_reset_input_aggregates"):
            try:
                self._input_monitor.get_and_reset_input_aggregates()
            except Exception:
                logger.debug(
                    "get_and_reset_input_aggregates drain failed", exc_info=True,
                )
        self._hw_recent.clear()

        self._recording = True
        if self._first_capture_delay > 0:
            logger.info("Delaying first capture by %.1fs", self._first_capture_delay)
            self._timer = threading.Timer(self._first_capture_delay, self._schedule_next)
            self._timer.daemon = True
            self._timer.start()
        else:
            self._schedule_next()

    def stop(self) -> None:
        """Pause capture.  Buffer contents are preserved."""
        if not self._recording:
            logger.warning("stop() called while not recording — ignored")
            return
        self._recording = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("Recording stopped — %d samples in buffer", len(self._samples))

    def clear(self) -> None:
        """Discard all buffered samples."""
        with self._lock:
            self._samples.clear()
        with self._unflushed_lock:
            self._unflushed.clear()
        logger.info("Buffer cleared")

    def flush_unflushed(self) -> list:
        """Return and clear any samples not yet sent to the batch callback."""
        with self._unflushed_lock:
            batch = list(self._unflushed)
            self._unflushed.clear()
        return batch

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_samples(self) -> list:
        """Return a snapshot of the buffer as a plain list."""
        with self._lock:
            return list(self._samples)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_samples_to_zip(
        samples: list,
        output_path: Path,
        window_seconds: int = 0,
        sample_interval: float = 0,
    ) -> None:
        """Write an arbitrary list of :class:`DataSample` objects to a ZIP.

        Archive layout::

            metadata.json       — session info and label summary
            samples.jsonl       — one JSON object per line (no screenshots)
            screenshots/
                <timestamp>.png — one PNG per sample that had a screenshot
        """
        if not samples:
            logger.warning("export_samples_to_zip called with empty list — nothing written")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".zip.tmp")

        logger.info("Exporting %d samples → %s", len(samples), output_path)

        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                labels_present = list({s.label for s in samples})
                meta = {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "sample_count": len(samples),
                    "window_seconds": window_seconds,
                    "sample_interval": sample_interval,
                    "labels_present": labels_present,
                    "time_range": {
                        "start": samples[0].timestamp,
                        "end": samples[-1].timestamp,
                    },
                }
                zf.writestr("metadata.json", json.dumps(meta, indent=2))

                jsonl_lines = [json.dumps(s.to_dict()) for s in samples]
                zf.writestr("samples.jsonl", "\n".join(jsonl_lines))
                logger.debug("Written samples.jsonl (%d lines)", len(jsonl_lines))

                screenshot_count = 0
                for sample in samples:
                    if sample.screenshot is not None:
                        ts_str = datetime.fromtimestamp(
                            sample.timestamp, tz=timezone.utc
                        ).strftime("%Y%m%dT%H%M%S%f")
                        buf = io.BytesIO()
                        sample.screenshot.save(buf, format="PNG")
                        zf.writestr(f"screenshots/{ts_str}.png", buf.getvalue())
                        screenshot_count += 1

                logger.debug("Written %d screenshots", screenshot_count)

            tmp_path.replace(output_path)
            size_kb = output_path.stat().st_size / 1024
            logger.info("Export complete: %s (%.1f KB)", output_path.name, size_kb)

        except Exception:
            logger.exception("export_samples_to_zip failed")
            tmp_path.unlink(missing_ok=True)
            raise

    def export_zip(self, output_path: Path) -> None:
        """Write the current buffer to a ZIP archive at *output_path*."""
        samples = self.get_samples()
        if not samples:
            logger.warning("export_zip called with empty buffer — nothing written")
            return
        self.export_samples_to_zip(
            samples, output_path,
            self._window_seconds, self._sample_interval,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        if not self._recording:
            return
        self._capture_sample()
        self._timer = threading.Timer(self._sample_interval, self._schedule_next)
        self._timer.daemon = True
        self._timer.start()

    def _capture_sample(self) -> None:
        try:
            timestamp = time.time()

            # Per-sample HW snapshot — append now, then carry the last N.
            hw_snapshot = flatten_hw_latest(
                self._hw.get_latest() if self._hw else {},
                timestamp,
            )
            self._hw_recent.append(hw_snapshot)
            hw_recent = list(self._hw_recent)

            # Input — drain so each sample reflects only this 1s window.
            input_since_last = build_input_since_last(self._input_monitor)

            # Foreground window
            rect, app_name, window_title = get_active_window_info()
            screenshot = take_screenshot(rect) if rect else None

            sample = DataSample(
                timestamp=timestamp,
                label=self._current_label,
                app_name=app_name,
                window_title=window_title,
                hw_recent=hw_recent,
                input_since_last=input_since_last,
                screenshot=screenshot,
            )

            with self._lock:
                self._samples.append(sample)

            # Batch auto-upload
            batch_to_send = None
            with self._unflushed_lock:
                self._unflushed.append(sample)
                if len(self._unflushed) >= self._batch_size and self._on_batch_ready:
                    batch_to_send = list(self._unflushed)
                    self._unflushed.clear()
            if batch_to_send is not None:
                threading.Thread(
                    target=self._on_batch_ready,
                    args=(batch_to_send,),
                    daemon=True,
                    name="batch-ready",
                ).start()

            logger.debug(
                "Sample captured — label=%r app=%r hw_pts=%d inputs=%d screenshot=%s",
                self._current_label, app_name,
                len(hw_recent),
                input_since_last["total_count"],
                "ok" if screenshot else "none",
            )

        except Exception:
            logger.exception("Error capturing sample — skipping")
