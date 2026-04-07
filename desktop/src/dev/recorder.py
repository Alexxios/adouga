"""Rolling-buffer data recorder for ML training data collection.

Captures screenshots and input metrics at a fixed interval, retaining the
last ``window_seconds`` of samples.  Hardware stats (CPU, RAM, GPU, disk) are
provided as rolling time-series by :class:`HardwareMonitor` and attached to
each sample as-is, giving every training example its full 3-minute context.

Call :meth:`export_zip` to package the current buffer as a labelled ZIP
archive ready for upload.

Per-sample contents
-------------------
* ``cpu_history``      — [{timestamp, percent, freq_ghz}, ...]
* ``ram_history``      — [{timestamp, percent, used_gb, total_gb}, ...]
* ``gpu_history``      — [{timestamp, load_percent, memory_used_gb, ...}, ...]
* ``disk_history``     — [{timestamp, read_bps, write_bps}, ...]
* ``input_count``      — keyboard + mouse events since the last sample
* ``flick_vectors``    — significant mouse-movement vectors
* ``input_sequence``   — timestamped raw event log (last 3 min)
* ``key_heatmaps``     — per-key frequency dicts for 1s/5s/15s/30s/1m/3m windows
* ``screenshot``       — PIL image of the active window (excluded from JSON)
"""

import dataclasses
import io
import json
import logging
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.system_monitor import get_active_window_rect
from src.utils import take_screenshot

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_SECONDS = 180
_DEFAULT_SAMPLE_INTERVAL = 5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DataSample:
    """Single time-stamped data point captured during recording."""

    timestamp: float
    label: str

    # Hardware histories — rolling 3-min time series from HardwareMonitor
    cpu_history: list   # [{"timestamp", "percent", "freq_ghz"}, ...]
    ram_history: list   # [{"timestamp", "percent", "used_gb", "total_gb"}, ...]
    gpu_history: list   # [{"timestamp", "load_percent", "memory_used_gb", ...}, ...]
    disk_history: list  # [{"timestamp", "read_bps", "write_bps"}, ...]

    # Input aggregate (since last sample)
    input_count: int
    flick_vectors: list  # [(dx, dy), ...]

    # Input detail — rolling 3-min window
    input_sequence: list  # [{"timestamp", "type", "value"}, ...]
    key_heatmaps: dict    # {"1s": {key: count}, "5s": ..., ...}

    # Visual (excluded from JSON export)
    screenshot: Optional[object] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """JSON-serialisable representation (screenshot excluded)."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "cpu_history": self.cpu_history,
            "ram_history": self.ram_history,
            "gpu_history": self.gpu_history,
            "disk_history": self.disk_history,
            "input_count": self.input_count,
            "flick_vectors": self.flick_vectors,
            "input_sequence": self.input_sequence,
            "key_heatmaps": self.key_heatmaps,
        }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

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
        sample_interval: int = _DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        self._input_monitor = input_monitor
        self._hw = hardware_monitor
        self._window_seconds = window_seconds
        self._sample_interval = sample_interval
        self._max_samples = max(1, window_seconds // sample_interval)

        self._samples: deque = deque(maxlen=self._max_samples)
        self._current_label: str = ""
        self._recording: bool = False
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        logger.info(
            "DataRecorder ready — window=%ds interval=%ds max_samples=%d hw=%s",
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

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin periodic capture.  No-op if already recording."""
        if self._recording:
            logger.warning("start() called while already recording — ignored")
            return
        logger.info("Recording started")
        self._recording = True
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
        logger.info("Buffer cleared")

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

    def export_zip(self, output_path: Path) -> None:
        """Write the current buffer to a ZIP archive at *output_path*.

        Archive layout::

            metadata.json       — session info and label summary
            samples.jsonl       — one JSON object per line (no screenshots)
            screenshots/
                <timestamp>.png — one PNG per sample that had a screenshot
        """
        samples = self.get_samples()
        if not samples:
            logger.warning("export_zip called with empty buffer — nothing written")
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
                    "window_seconds": self._window_seconds,
                    "sample_interval": self._sample_interval,
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
            logger.exception("export_zip failed")
            tmp_path.unlink(missing_ok=True)
            raise

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

            # Hardware histories
            cpu_history  = self._hw.get_cpu_history()  if self._hw else []
            ram_history  = self._hw.get_ram_history()  if self._hw else []
            gpu_history  = self._hw.get_gpu_history()  if self._hw else []
            disk_history = self._hw.get_disk_history() if self._hw else []

            # Input
            input_count     = self._input_monitor.get_and_reset_count()
            flicks          = self._input_monitor.get_flicks()
            input_sequence  = (
                self._input_monitor.get_input_sequence()
                if hasattr(self._input_monitor, "get_input_sequence") else []
            )
            key_heatmaps    = (
                self._input_monitor.get_key_heatmaps()
                if hasattr(self._input_monitor, "get_key_heatmaps") else {}
            )

            # Screenshot
            rect       = get_active_window_rect()
            screenshot = take_screenshot(rect) if rect else None

            sample = DataSample(
                timestamp=timestamp,
                label=self._current_label,
                cpu_history=cpu_history,
                ram_history=ram_history,
                gpu_history=gpu_history,
                disk_history=disk_history,
                input_count=input_count,
                flick_vectors=list(flicks),
                input_sequence=input_sequence,
                key_heatmaps=key_heatmaps,
                screenshot=screenshot,
            )

            with self._lock:
                self._samples.append(sample)

            logger.debug(
                "Sample captured — label=%r cpu_pts=%d ram_pts=%d "
                "gpu_pts=%d disk_pts=%d inputs=%d seq=%d screenshot=%s",
                self._current_label,
                len(cpu_history), len(ram_history),
                len(gpu_history), len(disk_history),
                input_count, len(input_sequence),
                "ok" if screenshot else "none",
            )

        except Exception:
            logger.exception("Error capturing sample — skipping")
