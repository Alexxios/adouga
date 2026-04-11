"""Background batch uploader for auto-uploading sample batches to Yandex Disk.

Receives batches of :class:`DataSample` via a thread-safe queue, exports each
batch to a temporary ZIP archive, and uploads via :class:`YaDiskUploader`.
"""

import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.dev.recorder import DataRecorder
from src.dev.uploader import YaDiskUploader

logger = logging.getLogger(__name__)


class BatchUploader:
    """Queues sample batches and uploads them in a background thread.

    Parameters
    ----------
    uploader:
        :class:`YaDiskUploader` instance (or *None* to save locally only).
    output_dir:
        Local directory for saving batch ZIP archives.
    tester_id:
        Tester name used for namespaced remote paths.
    """

    def __init__(
        self,
        uploader: Optional[YaDiskUploader],
        output_dir: Path,
        tester_id: str = "",
    ) -> None:
        self._uploader = uploader
        self._output_dir = Path(output_dir)
        self._tester_id = tester_id
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

        self._batches_uploaded = 0
        self._batches_failed = 0
        self._samples_uploaded = 0
        self._last_error: str = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="batch-uploader", daemon=True,
        )
        self._thread.start()
        logger.info("BatchUploader started")

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)  # sentinel to unblock worker
        logger.info("BatchUploader stopping")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, samples: list) -> None:
        """Add a batch of samples to the upload queue."""
        self._queue.put(samples)

    @property
    def tester_id(self) -> str:
        return self._tester_id

    @tester_id.setter
    def tester_id(self, value: str) -> None:
        self._tester_id = value

    @property
    def batches_uploaded(self) -> int:
        with self._lock:
            return self._batches_uploaded

    @property
    def batches_failed(self) -> int:
        with self._lock:
            return self._batches_failed

    @property
    def samples_uploaded(self) -> int:
        with self._lock:
            return self._samples_uploaded

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while self._running:
            try:
                batch = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if batch is None:
                break
            self._process_batch(batch)

    def _process_batch(self, samples: list) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        tester_slug = self._tester_id.replace(" ", "_").lower() if self._tester_id else "unknown"
        filename = f"batch_{ts}_{tester_slug}.zip"
        out_path = self._output_dir / filename

        try:
            DataRecorder.export_samples_to_zip(samples, out_path)
        except Exception as exc:
            logger.exception("Batch export failed")
            with self._lock:
                self._batches_failed += 1
                self._last_error = f"Export: {exc}"
            return

        if self._uploader is not None:
            try:
                self._uploader.upload(out_path)
                with self._lock:
                    self._batches_uploaded += 1
                    self._samples_uploaded += len(samples)
                logger.info("Batch uploaded: %s (%d samples)", filename, len(samples))
            except Exception as exc:
                logger.exception("Batch upload failed")
                with self._lock:
                    self._batches_failed += 1
                    self._last_error = f"Upload: {exc}"
        else:
            with self._lock:
                self._batches_uploaded += 1
                self._samples_uploaded += len(samples)
            logger.info("Batch saved locally: %s (%d samples)", filename, len(samples))
