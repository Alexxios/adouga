"""Tests for src.dev.recorder — DataRecorder & DataSample."""

import json
import time
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.dev.recorder import (
    DataRecorder,
    DataSample,
    _DEFAULT_SAMPLE_INTERVAL,
    _DEFAULT_WINDOW_SECONDS,
    _get_gpu_stats,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_EMPTY_HEATMAPS = {"1s": {}, "5s": {}, "15s": {}, "30s": {}, "1m": {}, "3m": {}}
_SAMPLE_SEQUENCE = [
    {"timestamp": 1000.0, "type": "key_press", "value": "a"},
    {"timestamp": 1001.0, "type": "mouse_click", "value": "left"},
]
_SAMPLE_HEATMAPS = {
    "1s": {"a": 1}, "5s": {"a": 3}, "15s": {"a": 5},
    "30s": {"a": 8}, "1m": {"a": 12}, "3m": {"a": 20, "left": 5},
}
_SAMPLE_GPU = {
    "name": "Tesla T4", "load_percent": 42.0,
    "memory_used_mb": 1024.0, "memory_total_mb": 16384.0,
    "memory_percent": 6.25, "temperature_c": 55,
}
_SAMPLE_DISK_IO = {"read_bps": 1024.0, "write_bps": 512.0}


def _mock_input_monitor(
    count: int = 3,
    flicks=None,
    sequence=None,
    heatmaps=None,
):
    m = MagicMock()
    m.get_and_reset_count.return_value = count
    m.get_flicks.return_value = flicks if flicks is not None else [(1, 2), (-3, 4)]
    m.get_input_sequence.return_value = sequence if sequence is not None else _SAMPLE_SEQUENCE
    m.get_key_heatmaps.return_value = heatmaps if heatmaps is not None else _SAMPLE_HEATMAPS
    return m


def _make_sample(**overrides) -> DataSample:
    defaults = dict(
        timestamp=1000.0,
        label="Gaming",
        cpu_percent=42.5,
        ram_percent=60.0,
        gpu_stats=_SAMPLE_GPU,
        disk_io=_SAMPLE_DISK_IO,
        input_count=7,
        flick_vectors=[(1, 2)],
        input_sequence=list(_SAMPLE_SEQUENCE),
        key_heatmaps=dict(_SAMPLE_HEATMAPS),
    )
    defaults.update(overrides)
    return DataSample(**defaults)


def _rgb_image(w: int = 10, h: int = 10) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 150, 200))


@pytest.fixture
def monitor():
    return _mock_input_monitor()


@pytest.fixture
def recorder(monitor):
    return DataRecorder(input_monitor=monitor, window_seconds=30, sample_interval=5)


def _fill(recorder, n: int = 3, with_screenshot: bool = True):
    for i in range(n):
        recorder._samples.append(
            _make_sample(
                timestamp=1000.0 + i * 5,
                label="Gaming" if i % 2 == 0 else "Not Gaming",
                screenshot=_rgb_image() if with_screenshot else None,
            )
        )


# ---------------------------------------------------------------------------
# DataSample — to_dict
# ---------------------------------------------------------------------------

def test_sample_to_dict_excludes_screenshot():
    s = _make_sample(screenshot=_rgb_image())
    d = s.to_dict()
    assert "screenshot" not in d


def test_sample_to_dict_includes_all_new_fields():
    s = _make_sample()
    d = s.to_dict()
    assert d["gpu_stats"] == _SAMPLE_GPU
    assert d["disk_io"] == _SAMPLE_DISK_IO
    assert d["input_sequence"] == _SAMPLE_SEQUENCE
    assert d["key_heatmaps"] == _SAMPLE_HEATMAPS


def test_sample_to_dict_is_json_serialisable():
    s = _make_sample()
    text = json.dumps(s.to_dict())
    assert "Gaming" in text
    assert "key_heatmaps" in text
    assert "gpu_stats" in text


def test_sample_to_dict_none_fields_serialise():
    s = _make_sample(gpu_stats=None, disk_io=None)
    d = s.to_dict()
    assert d["gpu_stats"] is None
    assert d["disk_io"] is None
    assert json.dumps(d)  # must not raise


def test_sample_screenshot_defaults_to_none():
    s = _make_sample()
    assert s.screenshot is None


# ---------------------------------------------------------------------------
# _get_gpu_stats helper
# ---------------------------------------------------------------------------

def test_get_gpu_stats_returns_none_without_gputil():
    with patch.dict("sys.modules", {"GPUtil": None}):
        assert _get_gpu_stats() is None


def test_get_gpu_stats_returns_none_when_no_gpus():
    mock_gputil = MagicMock()
    mock_gputil.getGPUs.return_value = []
    with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
        assert _get_gpu_stats() is None


def test_get_gpu_stats_returns_dict_for_nvidia_gpu():
    mock_gpu = MagicMock()
    mock_gpu.name = "RTX 4090"
    mock_gpu.load = 0.42
    mock_gpu.memoryUsed = 4096
    mock_gpu.memoryTotal = 24576
    mock_gpu.memoryUtil = 0.1667
    mock_gpu.temperature = 72
    mock_gputil = MagicMock()
    mock_gputil.getGPUs.return_value = [mock_gpu]
    with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
        result = _get_gpu_stats()
    assert result["name"] == "RTX 4090"
    assert result["load_percent"] == 42.0
    assert result["memory_percent"] == pytest.approx(16.7, abs=0.1)
    assert result["temperature_c"] == 72


def test_get_gpu_stats_returns_none_on_exception():
    mock_gputil = MagicMock()
    mock_gputil.getGPUs.side_effect = RuntimeError("driver error")
    with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
        assert _get_gpu_stats() is None


# ---------------------------------------------------------------------------
# DataRecorder — initialisation
# ---------------------------------------------------------------------------

def test_recorder_default_max_samples(monitor):
    rec = DataRecorder(input_monitor=monitor)
    assert rec.max_samples == _DEFAULT_WINDOW_SECONDS // _DEFAULT_SAMPLE_INTERVAL


def test_recorder_custom_window_and_interval(monitor):
    rec = DataRecorder(input_monitor=monitor, window_seconds=60, sample_interval=10)
    assert rec.max_samples == 6


def test_recorder_initial_state(recorder):
    assert not recorder.is_recording
    assert recorder.sample_count == 0
    assert recorder.current_label == ""


def test_recorder_label_setter(recorder):
    recorder.current_label = "Gaming"
    assert recorder.current_label == "Gaming"


# ---------------------------------------------------------------------------
# DataRecorder — control (start / stop)
# ---------------------------------------------------------------------------

def test_recorder_start_sets_recording(recorder):
    with patch.object(recorder, "_schedule_next"):
        recorder.start()
    assert recorder.is_recording


def test_recorder_stop_clears_recording(recorder):
    with patch.object(recorder, "_schedule_next"):
        recorder.start()
    recorder.stop()
    assert not recorder.is_recording


def test_recorder_double_start_is_idempotent(recorder):
    with patch.object(recorder, "_schedule_next") as mock_sched:
        recorder.start()
        recorder.start()
    mock_sched.assert_called_once()


def test_recorder_stop_when_not_recording_is_safe(recorder):
    recorder.stop()  # must not raise


def test_recorder_clear_empties_buffer(recorder):
    recorder._samples.append(_make_sample())
    assert recorder.sample_count == 1
    recorder.clear()
    assert recorder.sample_count == 0


# ---------------------------------------------------------------------------
# DataRecorder — buffer
# ---------------------------------------------------------------------------

def test_recorder_get_samples_returns_list_copy(recorder):
    recorder._samples.append(_make_sample())
    samples = recorder.get_samples()
    assert isinstance(samples, list)
    samples.clear()
    assert recorder.sample_count == 1  # internal deque unaffected


def test_recorder_rolling_buffer_eviction(monitor):
    rec = DataRecorder(input_monitor=monitor, window_seconds=10, sample_interval=5)
    assert rec.max_samples == 2
    for i in range(5):
        rec._samples.append(_make_sample(timestamp=float(i)))
    assert rec.sample_count == 2
    assert rec.get_samples()[0].timestamp == 3.0
    assert rec.get_samples()[1].timestamp == 4.0


# ---------------------------------------------------------------------------
# DataRecorder — _schedule_next
# ---------------------------------------------------------------------------

@patch("src.dev.recorder.threading")
@patch("src.dev.recorder.DataRecorder._capture_sample")
def test_schedule_next_calls_capture_and_timer(mock_capture, mock_threading, recorder):
    recorder._recording = True
    recorder._schedule_next()
    mock_capture.assert_called_once()
    mock_threading.Timer.assert_called_once()


# ---------------------------------------------------------------------------
# DataRecorder — _capture_sample
# ---------------------------------------------------------------------------

@patch("src.dev.recorder._get_gpu_stats", return_value=_SAMPLE_GPU)
@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
@patch("src.dev.recorder.psutil")
def test_capture_sample_appends_to_buffer(mock_psutil, _rect, _shot, _gpu, recorder):
    mock_psutil.cpu_percent.return_value = 55.0
    mock_psutil.virtual_memory.return_value.percent = 70.0
    mock_psutil.disk_io_counters.return_value = MagicMock(
        read_bytes=2000, write_bytes=1000
    )
    recorder._last_disk_io = MagicMock(read_bytes=1000, write_bytes=500)
    recorder._current_label = "Gaming"

    recorder._capture_sample()

    assert recorder.sample_count == 1
    s = recorder.get_samples()[0]
    assert s.label == "Gaming"
    assert s.cpu_percent == 55.0
    assert s.ram_percent == 70.0
    assert s.gpu_stats == _SAMPLE_GPU
    assert s.disk_io["read_bps"] == pytest.approx(200.0)   # delta / interval (5s)
    assert s.disk_io["write_bps"] == pytest.approx(100.0)
    assert s.screenshot is None


@patch("src.dev.recorder._get_gpu_stats", return_value=None)
@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
@patch("src.dev.recorder.psutil")
def test_capture_sample_records_input_sequence(mock_psutil, _rect, _shot, _gpu, recorder):
    mock_psutil.cpu_percent.return_value = 10.0
    mock_psutil.virtual_memory.return_value.percent = 20.0
    mock_psutil.disk_io_counters.return_value = None
    recorder._last_disk_io = None

    recorder._capture_sample()

    s = recorder.get_samples()[0]
    assert s.input_sequence == _SAMPLE_SEQUENCE
    assert s.key_heatmaps == _SAMPLE_HEATMAPS


@patch("src.dev.recorder._get_gpu_stats", return_value=None)
@patch("src.dev.recorder.take_screenshot")
@patch("src.dev.recorder.get_active_window_rect", return_value=(0, 0, 100, 100))
@patch("src.dev.recorder.psutil")
def test_capture_sample_stores_screenshot(mock_psutil, _rect, mock_shot, _gpu, recorder):
    img = _rgb_image()
    mock_shot.return_value = img
    mock_psutil.cpu_percent.return_value = 10.0
    mock_psutil.virtual_memory.return_value.percent = 20.0
    mock_psutil.disk_io_counters.return_value = None
    recorder._last_disk_io = None

    recorder._capture_sample()

    assert recorder.get_samples()[0].screenshot is img


@patch("src.dev.recorder._get_gpu_stats", side_effect=RuntimeError("boom"))
@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
@patch("src.dev.recorder.psutil")
def test_capture_sample_swallows_exceptions(mock_psutil, _rect, _shot, _gpu, recorder):
    mock_psutil.cpu_percent.side_effect = RuntimeError("cpu boom")
    recorder._capture_sample()  # must not raise
    assert recorder.sample_count == 0


# ---------------------------------------------------------------------------
# DataRecorder — disk I/O delta
# ---------------------------------------------------------------------------

def test_capture_disk_io_computes_delta(recorder):
    recorder._last_disk_io = MagicMock(read_bytes=0, write_bytes=0)
    current = MagicMock(read_bytes=5000, write_bytes=2500)
    with patch("src.dev.recorder.psutil") as mp:
        mp.disk_io_counters.return_value = current
        result = recorder._capture_disk_io()
    assert result["read_bps"] == pytest.approx(1000.0)   # 5000 / 5s
    assert result["write_bps"] == pytest.approx(500.0)


def test_capture_disk_io_returns_none_when_unavailable(recorder):
    recorder._last_disk_io = None
    with patch("src.dev.recorder.psutil") as mp:
        mp.disk_io_counters.return_value = None
        result = recorder._capture_disk_io()
    assert result is None


# ---------------------------------------------------------------------------
# DataRecorder — export
# ---------------------------------------------------------------------------

def test_export_creates_file(tmp_path, recorder):
    _fill(recorder)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    assert out.exists() and out.stat().st_size > 0


def test_export_metadata(tmp_path, recorder):
    _fill(recorder, n=2)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("metadata.json"))
    assert meta["sample_count"] == 2
    assert "window_seconds" in meta
    assert "labels_present" in meta


def test_export_samples_jsonl_contains_new_fields(tmp_path, recorder):
    _fill(recorder, n=3)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        lines = zf.read("samples.jsonl").decode().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "key_heatmaps" in obj
        assert "input_sequence" in obj
        assert "gpu_stats" in obj
        assert "disk_io" in obj
        assert "screenshot" not in obj


def test_export_screenshots_included(tmp_path, recorder):
    _fill(recorder, n=2, with_screenshot=True)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        shots = [n for n in zf.namelist() if n.startswith("screenshots/")]
    assert len(shots) == 2


def test_export_screenshots_skipped_when_none(tmp_path, recorder):
    _fill(recorder, n=2, with_screenshot=False)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        shots = [n for n in zf.namelist() if n.startswith("screenshots/")]
    assert len(shots) == 0


def test_export_empty_buffer_does_not_create_file(tmp_path, recorder):
    recorder.export_zip(tmp_path / "empty.zip")
    assert not (tmp_path / "empty.zip").exists()


def test_export_creates_parent_dirs(tmp_path, recorder):
    _fill(recorder, n=1)
    out = tmp_path / "deep" / "nested" / "session.zip"
    recorder.export_zip(out)
    assert out.exists()


def test_export_produces_valid_zip(tmp_path, recorder):
    _fill(recorder, n=2)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    assert zipfile.is_zipfile(out)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_recorder_captures_real_system_stats():
    monitor = _mock_input_monitor()
    with (
        patch("src.dev.recorder.get_active_window_rect", return_value=None),
        patch("src.dev.recorder.take_screenshot", return_value=None),
        patch("src.dev.recorder._get_gpu_stats", return_value=None),
    ):
        rec = DataRecorder(input_monitor=monitor, window_seconds=10, sample_interval=1)
        rec.current_label = "Not Gaming"
        rec.start()
        time.sleep(1.5)
        rec.stop()

    samples = rec.get_samples()
    assert len(samples) >= 1
    s = samples[0]
    assert 0.0 <= s.cpu_percent <= 100.0
    assert 0.0 <= s.ram_percent <= 100.0
    assert s.label == "Not Gaming"
    assert isinstance(s.input_sequence, list)
    assert isinstance(s.key_heatmaps, dict)


def test_export_roundtrip(tmp_path):
    monitor = _mock_input_monitor()
    with (
        patch("src.dev.recorder.get_active_window_rect", return_value=None),
        patch("src.dev.recorder.take_screenshot", return_value=None),
        patch("src.dev.recorder._get_gpu_stats", return_value=None),
        patch("src.dev.recorder.psutil") as mp,
    ):
        mp.cpu_percent.return_value = 33.0
        mp.virtual_memory.return_value.percent = 55.0
        mp.disk_io_counters.return_value = None
        rec = DataRecorder(input_monitor=monitor, window_seconds=20, sample_interval=1)
        rec._last_disk_io = None
        rec.current_label = "Gaming"
        for _ in range(4):
            rec._capture_sample()

    out = tmp_path / "roundtrip.zip"
    rec.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("metadata.json"))
        lines = zf.read("samples.jsonl").decode().strip().split("\n")

    assert meta["sample_count"] == 4
    assert len(lines) == 4
    for line in lines:
        obj = json.loads(line)
        assert obj["label"] == "Gaming"
        assert "key_heatmaps" in obj
        assert "input_sequence" in obj
