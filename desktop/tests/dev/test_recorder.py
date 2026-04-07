"""Tests for src.dev.recorder — DataRecorder & DataSample."""

import dataclasses
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
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CPU_HIST  = [{"timestamp": 1000.0, "percent": 25.0, "freq_ghz": 3.5}]
_RAM_HIST  = [{"timestamp": 1000.0, "percent": 60.0, "used_gb": 8.0, "total_gb": 16.0}]
_GPU_HIST  = [{"timestamp": 1000.0, "load_percent": 42.0,
               "memory_used_gb": 4.0, "memory_total_gb": 16.0, "temperature_c": 65}]
_DISK_HIST = [{"timestamp": 1000.0, "read_bps": 1024.0, "write_bps": 512.0}]
_SEQ       = [
    {"timestamp": 999.0, "type": "key_press",   "value": "a"},
    {"timestamp": 999.5, "type": "mouse_click", "value": "left"},
]
_HEATMAPS  = {
    "1s": {"a": 1}, "5s": {"a": 3}, "15s": {"a": 5},
    "30s": {"a": 8}, "1m": {"a": 12}, "3m": {"a": 20, "left": 5},
}

# Canonical base sample — use dataclasses.replace() for overrides (type-safe)
_BASE_SAMPLE = DataSample(
    timestamp=1000.0,
    label="Gaming",
    cpu_history=_CPU_HIST,
    ram_history=_RAM_HIST,
    gpu_history=_GPU_HIST,
    disk_history=_DISK_HIST,
    input_count=7,
    flick_vectors=[(1, 2)],
    input_sequence=_SEQ,
    key_heatmaps=_HEATMAPS,
)


def _make_sample(**overrides) -> DataSample:
    return dataclasses.replace(_BASE_SAMPLE, **overrides)


def _rgb_image(w: int = 10, h: int = 10) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 150, 200))


def _mock_input_monitor(count: int = 3, flicks=None, sequence=None, heatmaps=None):
    m = MagicMock()
    m.get_and_reset_count.return_value = count
    m.get_flicks.return_value = flicks or [(1, 2), (-3, 4)]
    m.get_input_sequence.return_value = sequence if sequence is not None else list(_SEQ)
    m.get_key_heatmaps.return_value = heatmaps if heatmaps is not None else dict(_HEATMAPS)
    return m


def _mock_hw_monitor(cpu=None, ram=None, gpu=None, disk=None):
    m = MagicMock()
    m.get_cpu_history.return_value  = cpu  if cpu  is not None else list(_CPU_HIST)
    m.get_ram_history.return_value  = ram  if ram  is not None else list(_RAM_HIST)
    m.get_gpu_history.return_value  = gpu  if gpu  is not None else list(_GPU_HIST)
    m.get_disk_history.return_value = disk if disk is not None else list(_DISK_HIST)
    return m


@pytest.fixture
def monitor():
    return _mock_input_monitor()


@pytest.fixture
def hw():
    return _mock_hw_monitor()


@pytest.fixture
def recorder(monitor, hw):
    return DataRecorder(
        input_monitor=monitor,
        hardware_monitor=hw,
        window_seconds=30,
        sample_interval=5,
    )


def _fill(recorder, n: int = 3, with_screenshot: bool = True) -> None:
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
    assert "screenshot" not in s.to_dict()


def test_sample_to_dict_includes_history_fields():
    d = _BASE_SAMPLE.to_dict()
    assert d["cpu_history"]  == _CPU_HIST
    assert d["ram_history"]  == _RAM_HIST
    assert d["gpu_history"]  == _GPU_HIST
    assert d["disk_history"] == _DISK_HIST


def test_sample_to_dict_includes_input_fields():
    d = _BASE_SAMPLE.to_dict()
    assert d["input_sequence"] == _SEQ
    assert d["key_heatmaps"]   == _HEATMAPS
    assert d["input_count"]    == 7


def test_sample_to_dict_is_json_serialisable():
    text = json.dumps(_BASE_SAMPLE.to_dict())
    assert "cpu_history" in text
    assert "key_heatmaps" in text


def test_sample_empty_histories_serialise():
    s = _make_sample(cpu_history=[], ram_history=[], gpu_history=[], disk_history=[])
    d = s.to_dict()
    assert d["cpu_history"] == []
    assert json.dumps(d)  # must not raise


def test_sample_screenshot_defaults_to_none():
    assert _BASE_SAMPLE.screenshot is None


# ---------------------------------------------------------------------------
# DataRecorder — initialisation
# ---------------------------------------------------------------------------

def test_recorder_default_max_samples(monitor):
    rec = DataRecorder(input_monitor=monitor)
    assert rec.max_samples == _DEFAULT_WINDOW_SECONDS // _DEFAULT_SAMPLE_INTERVAL


def test_recorder_custom_window_and_interval(monitor):
    assert DataRecorder(input_monitor=monitor, window_seconds=60, sample_interval=10).max_samples == 6


def test_recorder_initial_state(recorder):
    assert not recorder.is_recording
    assert recorder.sample_count == 0
    assert recorder.current_label == ""


def test_recorder_label_setter(recorder):
    recorder.current_label = "Gaming"
    assert recorder.current_label == "Gaming"


def test_recorder_without_hw_monitor(monitor):
    rec = DataRecorder(input_monitor=monitor)
    assert rec._hw is None


# ---------------------------------------------------------------------------
# DataRecorder — control
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
    recorder.stop()


def test_recorder_clear_empties_buffer(recorder):
    recorder._samples.append(_make_sample())
    recorder.clear()
    assert recorder.sample_count == 0


# ---------------------------------------------------------------------------
# DataRecorder — buffer
# ---------------------------------------------------------------------------

def test_recorder_get_samples_returns_list_copy(recorder):
    recorder._samples.append(_make_sample())
    samples = recorder.get_samples()
    samples.clear()
    assert recorder.sample_count == 1


def test_recorder_rolling_buffer_eviction(monitor, hw):
    rec = DataRecorder(input_monitor=monitor, hardware_monitor=hw,
                       window_seconds=10, sample_interval=5)
    assert rec.max_samples == 2
    for i in range(5):
        rec._samples.append(_make_sample(timestamp=float(i)))
    assert rec.sample_count == 2
    assert rec.get_samples()[0].timestamp == 3.0


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

@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
def test_capture_sample_uses_hw_histories(_rect, _shot, recorder):
    recorder._current_label = "Gaming"
    recorder._capture_sample()

    assert recorder.sample_count == 1
    s = recorder.get_samples()[0]
    assert s.label          == "Gaming"
    assert s.cpu_history    == _CPU_HIST
    assert s.ram_history    == _RAM_HIST
    assert s.gpu_history    == _GPU_HIST
    assert s.disk_history   == _DISK_HIST


@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
def test_capture_sample_records_input_detail(_rect, _shot, recorder):
    recorder._capture_sample()
    s = recorder.get_samples()[0]
    assert s.input_sequence == list(_SEQ)
    assert s.key_heatmaps   == dict(_HEATMAPS)


@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
def test_capture_sample_empty_histories_without_hw_monitor(_rect, _shot, monitor):
    rec = DataRecorder(input_monitor=monitor, hardware_monitor=None,
                       window_seconds=30, sample_interval=5)
    rec._capture_sample()
    s = rec.get_samples()[0]
    assert s.cpu_history  == []
    assert s.disk_history == []


@patch("src.dev.recorder.take_screenshot")
@patch("src.dev.recorder.get_active_window_rect", return_value=(0, 0, 100, 100))
def test_capture_sample_stores_screenshot(_rect, mock_shot, recorder):
    img = _rgb_image()
    mock_shot.return_value = img
    recorder._capture_sample()
    assert recorder.get_samples()[0].screenshot is img


@patch("src.dev.recorder.take_screenshot", side_effect=RuntimeError("boom"))
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
def test_capture_sample_swallows_exceptions(_rect, _shot, recorder):
    recorder._hw.get_cpu_history.side_effect = RuntimeError("hw boom")
    recorder._capture_sample()  # must not raise
    assert recorder.sample_count == 0


# ---------------------------------------------------------------------------
# Export
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
    assert "labels_present" in meta


def test_export_samples_jsonl_has_history_fields(tmp_path, recorder):
    _fill(recorder, n=2)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        lines = zf.read("samples.jsonl").decode().strip().split("\n")
    for line in lines:
        obj = json.loads(line)
        assert "cpu_history"     in obj
        assert "ram_history"     in obj
        assert "gpu_history"     in obj
        assert "disk_history"    in obj
        assert "input_sequence"  in obj
        assert "key_heatmaps"    in obj
        assert "screenshot"  not in obj


def test_export_screenshots_included(tmp_path, recorder):
    _fill(recorder, n=2, with_screenshot=True)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        shots = [n for n in zf.namelist() if n.startswith("screenshots/")]
    assert len(shots) == 2


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

def test_recorder_captures_real_samples():
    monitor = _mock_input_monitor()
    hw = _mock_hw_monitor()
    with (
        patch("src.dev.recorder.get_active_window_rect", return_value=None),
        patch("src.dev.recorder.take_screenshot", return_value=None),
    ):
        rec = DataRecorder(input_monitor=monitor, hardware_monitor=hw,
                           window_seconds=10, sample_interval=1)
        rec.current_label = "Not Gaming"
        rec.start()
        time.sleep(1.5)
        rec.stop()

    samples = rec.get_samples()
    assert len(samples) >= 1
    s = samples[0]
    assert s.label == "Not Gaming"
    assert isinstance(s.cpu_history, list)
    assert isinstance(s.key_heatmaps, dict)


def test_export_roundtrip(tmp_path):
    monitor = _mock_input_monitor()
    hw = _mock_hw_monitor()
    with (
        patch("src.dev.recorder.get_active_window_rect", return_value=None),
        patch("src.dev.recorder.take_screenshot", return_value=None),
    ):
        rec = DataRecorder(input_monitor=monitor, hardware_monitor=hw,
                           window_seconds=20, sample_interval=1)
        rec.current_label = "Gaming"
        for _ in range(4):
            rec._capture_sample()

    out = tmp_path / "roundtrip.zip"
    rec.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        meta  = json.loads(zf.read("metadata.json"))
        lines = zf.read("samples.jsonl").decode().strip().split("\n")

    assert meta["sample_count"] == 4
    assert all(json.loads(l)["label"] == "Gaming" for l in lines)
    assert all("cpu_history" in json.loads(l) for l in lines)
