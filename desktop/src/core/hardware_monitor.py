"""Background hardware sampler — CPU, RAM, GPU, disk I/O."""

import logging
import time
import threading
from collections import deque
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

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

    def get_latest(self) -> dict:
        """Return the most recent reading for each metric.

        Returns a dict with keys ``cpu``, ``ram``, ``gpu``, ``disk``. Each
        value is the last entry in the corresponding rolling deque, or
        ``None`` when the deque is empty (e.g. no GPU detected, disk I/O
        unavailable, or the monitor hasn't sampled yet).
        """
        with self._lock:
            return {
                "cpu": self._cpu_hist[-1] if self._cpu_hist else None,
                "ram": self._ram_hist[-1] if self._ram_hist else None,
                "gpu": self._gpu_hist[-1] if self._gpu_hist else None,
                "disk": self._disk_hist[-1] if self._disk_hist else None,
            }

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
