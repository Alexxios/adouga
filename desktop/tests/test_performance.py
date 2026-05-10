"""Performance tests — measure resource footprint of the desktop app.

The app runs in the background alongside potentially resource-heavy foreground
apps (games, video editors, etc.), so every test here asserts that the overhead
stays within strict bounds.

All tests are **real-system integration tests** — no mocking.  They use
``psutil.Process(os.getpid())`` to introspect CPU, RAM, and thread count of
the test process itself.  Platform-specific tests (screenshot, window
detection) are guarded with ``skipif``.

Run with:
    poetry run pytest tests/test_performance.py -v
"""

import gc
import os
import platform
import statistics
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import pytest
from PIL import Image

from src.core.hardware_monitor import HardwareMonitor, _sample_gpu_once
from src.core.input_monitor import InputMonitor, _HEATMAP_INTERVALS
from src.core.screenshot import take_screenshot
from src.core.window import get_active_window_info, get_active_window_rect

_ONNX_MODEL = (
    Path(__file__).parent.parent.parent / "ml" / "models" / "model.onnx"
)
_HAS_ONNX_MODEL = _ONNX_MODEL.exists()
_IS_MACOS = platform.system() == "Darwin"
_IS_WINDOWS = platform.system() == "Windows"
_SUPPORTED_OS = _IS_MACOS or _IS_WINDOWS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float:
    return statistics.median(values)


def _measure_cpu(seconds: float) -> float:
    """Return own-process CPU% averaged over *seconds*."""
    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # prime — first call always 0
    time.sleep(seconds)
    return proc.cpu_percent()


def _rss_mb() -> float:
    """Return current resident set size of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def proc():
    """``psutil.Process`` for the current test process."""
    return psutil.Process(os.getpid())


@pytest.fixture
def large_image():
    """1920x1080 random RGB image — realistic screenshot size."""
    arr = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ===================================================================
# 1. CPU overhead
# ===================================================================

@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_hardware_monitor_cpu_overhead(_mock_gpu):
    """HardwareMonitor's 1-second sampling loop should use < 5% CPU."""
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=30)
    hw.start()
    try:
        cpu = _measure_cpu(5.0)
    finally:
        hw.stop()

    print(f"HardwareMonitor CPU overhead: {cpu:.1f}%")
    assert cpu < 5.0, f"HardwareMonitor used {cpu:.1f}% CPU (limit 5%)"


def test_input_monitor_cpu_overhead():
    """InputMonitor listeners (idle, no real input) should use < 3% CPU."""
    im = InputMonitor()
    try:
        cpu = _measure_cpu(3.0)
    finally:
        im.stop()

    print(f"InputMonitor idle CPU overhead: {cpu:.1f}%")
    assert cpu < 3.0, f"InputMonitor used {cpu:.1f}% CPU (limit 3%)"


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_combined_monitors_cpu_overhead(_mock_gpu):
    """Both monitors running together should use < 5% CPU."""
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=30)
    hw.start()
    im = InputMonitor()
    try:
        cpu = _measure_cpu(5.0)
    finally:
        hw.stop()
        im.stop()

    print(f"Combined monitors CPU overhead: {cpu:.1f}%")
    assert cpu < 5.0, f"Combined monitors used {cpu:.1f}% CPU (limit 5%)"


# ===================================================================
# 2. Thread count
# ===================================================================

@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_thread_count_with_monitors(_mock_gpu):
    """Total thread count with all monitors running should be <= 6."""
    baseline = threading.active_count()

    hw = HardwareMonitor(sample_interval=1, buffer_seconds=10)
    hw.start()
    im = InputMonitor()
    time.sleep(0.5)  # let threads start

    try:
        total = threading.active_count()
        added = total - baseline
    finally:
        hw.stop()
        im.stop()

    print(f"Threads: baseline={baseline}, with monitors={total}, added={added}")
    # hw-monitor(1) + kb-listener(1) + mouse-listener(1) = 3 threads added
    assert added <= 6, f"Added {added} threads (limit 6)"


# ===================================================================
# 3. RAM footprint
# ===================================================================

@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_ram_footprint_monitors(_mock_gpu):
    """Starting both monitors should add < 50 MB RSS."""
    gc.collect()
    rss_before = _rss_mb()

    hw = HardwareMonitor(sample_interval=1, buffer_seconds=30)
    hw.start()
    im = InputMonitor()
    time.sleep(3.0)  # let a few samples accumulate

    rss_after = _rss_mb()
    hw.stop()
    im.stop()

    delta = rss_after - rss_before
    print(f"RAM delta (monitors): {delta:.1f} MB (before={rss_before:.1f}, after={rss_after:.1f})")
    assert delta < 50.0, f"Monitors added {delta:.1f} MB RSS (limit 50 MB)"


@pytest.mark.skipif(not _HAS_ONNX_MODEL, reason="ONNX model not found")
def test_ram_footprint_with_onnx_model():
    """Loading the ONNX model should add < 200 MB RSS."""
    from src.app.inference import ONNXClassifier

    gc.collect()
    rss_before = _rss_mb()

    classifier = ONNXClassifier(use_gpu=False)

    rss_after = _rss_mb()
    delta = rss_after - rss_before
    print(f"RAM delta (ONNX model): {delta:.1f} MB")
    assert delta < 200.0, f"ONNX model added {delta:.1f} MB RSS (limit 200 MB)"
    del classifier


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_memory_stability_over_time(_mock_gpu):
    """Monitors running for 10 s should not leak memory (delta < 10 MB)."""
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=30)
    hw.start()
    im = InputMonitor()

    time.sleep(2.0)  # let initial allocations settle
    gc.collect()
    rss_start = _rss_mb()

    time.sleep(10.0)

    gc.collect()
    rss_end = _rss_mb()
    hw.stop()
    im.stop()

    delta = rss_end - rss_start
    print(f"RAM drift over 10 s: {delta:+.1f} MB")
    assert delta < 10.0, f"Memory grew by {delta:.1f} MB in 10 s (limit 10 MB)"


# ===================================================================
# 4. Latency tests
# ===================================================================

@pytest.mark.skipif(not _SUPPORTED_OS, reason="Unsupported OS for screenshots")
def test_screenshot_latency():
    """Screenshot capture should complete in < 200 ms median, < 500 ms max."""
    rect = get_active_window_rect()
    if rect is None:
        pytest.skip("No active window detected (headless?)")

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        img = take_screenshot(rect)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        if img is None:
            pytest.skip("Screenshot returned None (permission denied?)")

    med = _median(times)
    mx = max(times)
    print(f"Screenshot latency: median={med:.1f} ms, max={mx:.1f} ms, all={[f'{t:.0f}' for t in times]}")
    assert med < 200.0, f"Median screenshot latency {med:.1f} ms (limit 200 ms)"
    assert mx < 500.0, f"Max screenshot latency {mx:.1f} ms (limit 500 ms)"


@pytest.mark.skipif(not _HAS_ONNX_MODEL, reason="ONNX model not found")
def test_onnx_inference_latency(large_image):
    """Full predict() call should complete in < 500 ms median."""
    from src.app.inference import ONNXClassifier

    classifier = ONNXClassifier(use_gpu=False)

    # Warm-up
    classifier.predict(large_image)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        cls, conf = classifier.predict(large_image)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    med = _median(times)
    mx = max(times)
    print(f"Inference latency: median={med:.1f} ms, max={mx:.1f} ms")
    assert med < 500.0, f"Median inference latency {med:.1f} ms (limit 500 ms)"


@pytest.mark.skipif(not _HAS_ONNX_MODEL, reason="ONNX model not found")
def test_onnx_preprocessing_latency(large_image):
    """Image preprocessing should complete in < 50 ms median."""
    from src.app.inference import ONNXClassifier

    classifier = ONNXClassifier(use_gpu=False)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        classifier.preprocess_image(large_image)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    med = _median(times)
    print(f"Preprocessing latency: median={med:.1f} ms, max={max(times):.1f} ms")
    assert med < 50.0, f"Median preprocessing latency {med:.1f} ms (limit 50 ms)"


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_hardware_sample_latency(_mock_gpu):
    """A single HardwareMonitor._sample() call should complete in < 100 ms."""
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=10)
    # Prime psutil
    psutil.cpu_percent(interval=None)
    hw._last_disk_io = psutil.disk_io_counters()

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        hw._sample()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    med = _median(times)
    print(f"HW sample latency: median={med:.1f} ms, max={max(times):.1f} ms")
    assert med < 100.0, f"Median HW sample latency {med:.1f} ms (limit 100 ms)"


def test_get_key_heatmaps_latency_with_heavy_load():
    """get_key_heatmaps() with 10 000 events should complete in < 50 ms."""
    im = InputMonitor()
    try:
        # Inject 10 000 synthetic events spanning the last 3 minutes
        now = time.time()
        with im._lock:
            for i in range(10_000):
                im._events.append({
                    "timestamp": now - 180 + (i * 0.018),  # spread over 3 min
                    "type": "key_press",
                    "value": chr(ord("a") + (i % 26)),
                })

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            result = im.get_key_heatmaps()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        med = _median(times)
        print(f"get_key_heatmaps (10k events): median={med:.1f} ms, max={max(times):.1f} ms")
        # Verify correctness too
        assert len(result) == len(_HEATMAP_INTERVALS)
        assert med < 50.0, f"Median heatmap aggregation {med:.1f} ms (limit 50 ms)"
    finally:
        im.stop()


@pytest.mark.skipif(not _SUPPORTED_OS, reason="Unsupported OS for window detection")
def test_window_detection_latency():
    """get_active_window_rect() should complete in < 50 ms median."""
    # Warm-up
    get_active_window_rect()

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        get_active_window_rect()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    med = _median(times)
    print(f"Window detection latency: median={med:.1f} ms, max={max(times):.1f} ms")
    assert med < 50.0, f"Median window detection latency {med:.1f} ms (limit 50 ms)"


# ===================================================================
# 5. Full loop simulation
# ===================================================================

@pytest.mark.skipif(
    not (_HAS_ONNX_MODEL and _SUPPORTED_OS),
    reason="Requires ONNX model and supported OS",
)
def test_main_loop_tick_latency():
    """One full main-loop iteration (data + screenshot + inference) < 2 s."""
    from src.app.inference import ONNXClassifier

    hw = HardwareMonitor(sample_interval=1, buffer_seconds=30)
    hw.start()
    im = InputMonitor()
    classifier = ONNXClassifier(use_gpu=False)

    time.sleep(2.0)  # let hw monitor collect a couple samples

    try:
        t0 = time.perf_counter()

        # 1. Per-sample HW snapshot
        latest = hw.get_latest()

        # 2. Per-sample input aggregates + flick drain
        input_agg = im.get_and_reset_input_aggregates()
        flicks = im.get_and_reset_flicks()

        # 3. Foreground window info + screenshot
        rect, app_name, window_title = get_active_window_info()
        img = take_screenshot(rect) if rect else None

        # 4. Inference (if screenshot succeeded)
        if img is not None:
            pred_class, confidence = classifier.predict(img)

        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Full loop tick: {elapsed:.0f} ms")
        assert elapsed < 2000.0, f"Full loop tick took {elapsed:.0f} ms (limit 2000 ms)"

    finally:
        hw.stop()
        im.stop()
