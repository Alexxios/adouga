"""Rolling-buffer data recorder for ML training data collection.

Captures screenshots, device stats, and input metrics at a fixed interval,
retaining the last ``window_seconds`` of samples.  Call :meth:`export_zip`
to package the current buffer as a labelled ZIP archive ready for upload.

Per-sample stats
----------------
* CPU %                 — ``psutil.cpu_percent``
* RAM %                 — ``psutil.virtual_memory``
* GPU load / memory     — via GPUtil (NVIDIA); ``None`` if unavailable
* Disk I/O (bytes/sec)  — delta of ``psutil.disk_io_counters`` between samples
* Input event count     — keyboard + mouse clicks/scrolls since last sample
* Flick vectors         — significant mouse-movement vectors (InputMonitor)
* Input sequence        — timestamped raw event log for the last 3 min
* Key heatmaps          — per-key frequency dicts for 1s/5s/15s/30s/1m/3m windows
* Screenshot            — PIL image of the active window
"""

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

import psutil

from src.system_monitor import get_active_window_rect
from src.utils import take_screenshot

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_SECONDS = 180   # 3 minutes
_DEFAULT_SAMPLE_INTERVAL = 5    # seconds between captures


# ---------------------------------------------------------------------------
# GPU helper (optional — requires GPUtil / NVIDIA drivers)
# ---------------------------------------------------------------------------

def _get_gpu_stats() -> Optional[dict]:
    """Return basic GPU stats for the first detected NVIDIA GPU, or ``None``."""
    try:
        import GPUtil  # type: ignore[import]
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            stats = {
                "name": g.name,
                "load_percent": round(g.load * 100, 1),
                "memory_used_mb": round(g.memoryUsed, 1),
                "memory_total_mb": round(g.memoryTotal, 1),
                "memory_percent": round(g.memoryUtil * 100, 1),
                "temperature_c": g.temperature,
            }
            logger.debug("GPU stats: %s", stats)
            return stats
    except ImportError:
        logger.debug("GPUtil not installed — GPU stats unavailable")
    except Exception:
        logger.debug("GPUtil query failed", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DataSample:
    """Single time-stamped data point captured during recording."""

    timestamp: float
    label: str

    # System stats
    cpu_percent: float
    ram_percent: float
    gpu_stats: Optional[dict]           # None when GPU unavailable
    disk_io: Optional[dict]             # {"read_bps": float, "write_bps": float}

    # Input aggregate (since last sample)
    input_count: int
    flick_vectors: list                 # list[tuple[int, int]]

    # Input detail (rolling 3-min window)
    input_sequence: list                # list[{"timestamp", "type", "value"}]
    key_heatmaps: dict                  # {"1s": {key: count}, "5s": ..., ...}

    # Visual
    screenshot: Optional[object] = field(default=None, repr=False)  # PIL.Image | None

    def to_dict(self) -> dict:
        """JSON-serialisable representation (screenshot excluded)."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "gpu_stats": self.gpu_stats,
            "disk_io": self.disk_io,
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
        An :class:`src.system_monitor.InputMonitor` instance (or any object
        exposing ``get_and_reset_count``, ``get_flicks``,
        ``get_input_sequence``, and ``get_key_heatmaps``).
    window_seconds:
        Length of the rolling window in seconds (default 180 = 3 min).
    sample_interval:
        Seconds between consecutive captures (default 5 s).
    """

    def __init__(
        self,
        input_monitor,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        sample_interval: int = _DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        self._input_monitor = input_monitor
        self._window_seconds = window_seconds
        self._sample_interval = sample_interval
        self._max_samples = max(1, window_seconds // sample_interval)

        self._samples: deque = deque(maxlen=self._max_samples)
        self._current_label: str = ""
        self._recording: bool = False
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        # Disk I/O baseline for delta computation
        self._last_disk_io = psutil.disk_io_counters()

        logger.info(
            "DataRecorder ready — window=%ds interval=%ds max_samples=%d",
            window_seconds,
            sample_interval,
            self._max_samples,
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
        """Return a snapshot of the current buffer as a plain list."""
        with self._lock:
            return list(self._samples)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_zip(self, output_path: Path) -> None:
        """Write the current buffer to a ZIP archive at *output_path*.

        Archive layout::

            metadata.json       — session info and label summary
            samples.jsonl       — one JSON object per line (stats, no screenshots)
            screenshots/
                <timestamp>.png — one PNG per sample that had a screenshot

        The file is created atomically via a temporary sibling file.
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
                logger.debug("Written metadata.json")

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

            # --- System stats ---
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            gpu = _get_gpu_stats()
            disk_io = self._capture_disk_io()

            # --- Input ---
            input_count = self._input_monitor.get_and_reset_count()
            flicks = self._input_monitor.get_flicks()
            input_sequence = (
                self._input_monitor.get_input_sequence()
                if hasattr(self._input_monitor, "get_input_sequence")
                else []
            )
            key_heatmaps = (
                self._input_monitor.get_key_heatmaps()
                if hasattr(self._input_monitor, "get_key_heatmaps")
                else {}
            )

            # --- Screenshot ---
            rect = get_active_window_rect()
            screenshot = take_screenshot(rect) if rect else None

            sample = DataSample(
                timestamp=timestamp,
                label=self._current_label,
                cpu_percent=cpu,
                ram_percent=ram,
                gpu_stats=gpu,
                disk_io=disk_io,
                input_count=input_count,
                flick_vectors=list(flicks),
                input_sequence=input_sequence,
                key_heatmaps=key_heatmaps,
                screenshot=screenshot,
            )

            with self._lock:
                self._samples.append(sample)

            logger.debug(
                "Sample captured — label=%r cpu=%.1f%% ram=%.1f%% "
                "gpu=%s disk_io=%s inputs=%d seq_len=%d screenshot=%s",
                self._current_label, cpu, ram,
                f"{gpu['load_percent']}%" if gpu else "n/a",
                f"r={disk_io['read_bps']:.0f} w={disk_io['write_bps']:.0f}" if disk_io else "n/a",
                input_count, len(input_sequence),
                "ok" if screenshot else "none",
            )

        except Exception:
            logger.exception("Error capturing sample — skipping")

    def _capture_disk_io(self) -> Optional[dict]:
        """Return read/write bytes-per-second since the last sample."""
        try:
            current = psutil.disk_io_counters()
            if current is None or self._last_disk_io is None:
                return None
            dt = self._sample_interval or 1
            result = {
                "read_bps": (current.read_bytes - self._last_disk_io.read_bytes) / dt,
                "write_bps": (current.write_bytes - self._last_disk_io.write_bytes) / dt,
            }
            self._last_disk_io = current
            return result
        except Exception:
            logger.debug("disk_io_counters() failed", exc_info=True)
            return None
