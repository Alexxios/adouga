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
    _HW_RECENT_DEPTH,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_HW_SLOT = {
    "timestamp": 1000.0,
    "cpu_percent": 25.0,
    "cpu_freq_ghz": 3.5,
    "ram_percent": 60.0,
    "ram_used_gb": 8.0,
    "gpu_present": 1,
    "gpu_load_percent": 42.0,
    "gpu_memory_used_gb": 4.0,
    "gpu_temperature_c": 65,
    "disk_read_bps": 1024.0,
    "disk_write_bps": 512.0,
}

_INPUT_AGG = {
    "key_press_count": 5,
    "mouse_click_count": 1,
    "mouse_scroll_count": 0,
    "mouse_move_count": 12,
    "total_count": 18,
    "gaming_keys": {
        "w": 2, "a": 0, "s": 1, "d": 0,
        "space": 0, "shift": 0, "left": 0, "right": 0,
    },
    "flick_count": 2,
    "flick_mag_mean": 22.36,
    "flick_mag_max": 30.0,
    "flick_dx_mean": 0.0,
    "flick_dy_mean": 0.0,
}

_BASE_SAMPLE = DataSample(
    timestamp=1000.0,
    label="Gaming",
    app_name="valorant.exe",
    window_title="VALORANT",
    hw_recent=[dict(_HW_SLOT)],
    input_since_last=dict(_INPUT_AGG),
)


def _make_sample(**overrides) -> DataSample:
    return dataclasses.replace(_BASE_SAMPLE, **overrides)


def _rgb_image(w: int = 10, h: int = 10) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 150, 200))


def _mock_input_monitor(
    count: int = 3,
    flicks=None,
    aggregates=None,
    sequence=None,
    heatmaps=None,
):
    m = MagicMock()
    m.get_and_reset_count.return_value = count
    m.get_and_reset_flicks.return_value = flicks if flicks is not None else [(3, 4), (-5, 12)]
    m.get_flicks.return_value = []
    m.get_and_reset_input_aggregates.return_value = (
        dict(aggregates) if aggregates is not None else {
            "key_press_count": 5,
            "mouse_click_count": 1,
            "mouse_scroll_count": 0,
            "mouse_move_count": 7,
            "total_count": 13,
            "gaming_keys": {
                "w": 2, "a": 0, "s": 1, "d": 0,
                "space": 0, "shift": 0, "left": 0, "right": 0,
            },
        }
    )
    m.get_input_sequence.return_value = sequence if sequence is not None else []
    m.get_key_heatmaps.return_value = heatmaps if heatmaps is not None else {}
    return m


def _mock_hw_monitor(latest=None):
    m = MagicMock()
    m.get_latest.return_value = latest if latest is not None else {
        "cpu":  {"timestamp": 1000.0, "percent": 25.0, "freq_ghz": 3.5},
        "ram":  {"timestamp": 1000.0, "percent": 60.0, "used_gb": 8.0, "total_gb": 16.0},
        "gpu":  {"timestamp": 1000.0, "load_percent": 42.0,
                 "memory_used_gb": 4.0, "memory_total_gb": 16.0, "temperature_c": 65},
        "disk": {"timestamp": 1000.0, "read_bps": 1024.0, "write_bps": 512.0},
    }
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


def test_sample_to_dict_includes_new_schema_fields():
    d = _BASE_SAMPLE.to_dict()
    assert d["app_name"] == "valorant.exe"
    assert d["window_title"] == "VALORANT"
    assert d["hw_recent"] == [_HW_SLOT]
    assert d["input_since_last"] == _INPUT_AGG


def test_sample_to_dict_is_json_serialisable():
    text = json.dumps(_BASE_SAMPLE.to_dict())
    assert "hw_recent" in text
    assert "input_since_last" in text
    assert "app_name" in text


def test_sample_empty_hw_recent_serialises():
    s = _make_sample(hw_recent=[])
    d = s.to_dict()
    assert d["hw_recent"] == []
    assert json.dumps(d)  # must not raise


def test_sample_screenshot_defaults_to_none():
    assert _BASE_SAMPLE.screenshot is None


# ---------------------------------------------------------------------------
# DataRecorder — initialisation
# ---------------------------------------------------------------------------

def test_recorder_default_max_samples(monitor):
    rec = DataRecorder(input_monitor=monitor)
    assert rec.max_samples == int(_DEFAULT_WINDOW_SECONDS / _DEFAULT_SAMPLE_INTERVAL)


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


def test_recorder_start_drains_input_monitor(recorder, monitor):
    """First sample should not be poisoned by pre-launch input accumulation."""
    with patch.object(recorder, "_schedule_next"):
        recorder.start()
    monitor.get_and_reset_count.assert_called_once()
    monitor.get_and_reset_flicks.assert_called_once()
    monitor.get_and_reset_input_aggregates.assert_called_once()


def test_recorder_start_clears_hw_recent(recorder):
    recorder._hw_recent.append(dict(_HW_SLOT))
    with patch.object(recorder, "_schedule_next"):
        recorder.start()
    assert len(recorder._hw_recent) == 0


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
@patch("src.dev.recorder.get_active_window_info", return_value=(None, "valorant.exe", "VALORANT"))
def test_capture_sample_uses_hw_latest(_info, _shot, recorder):
    recorder._current_label = "Gaming"
    recorder._capture_sample()

    assert recorder.sample_count == 1
    s = recorder.get_samples()[0]
    assert s.label == "Gaming"
    assert s.app_name == "valorant.exe"
    assert s.window_title == "VALORANT"
    assert len(s.hw_recent) == 1
    assert s.hw_recent[0]["cpu_percent"] == 25.0
    assert s.hw_recent[0]["disk_write_bps"] == 512.0


@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_info", return_value=(None, "", ""))
def test_capture_sample_records_input_aggregates(_info, _shot, recorder):
    recorder._capture_sample()
    s = recorder.get_samples()[0]
    assert s.input_since_last["total_count"] == 13
    assert s.input_since_last["mouse_move_count"] == 7
    # flick_count is computed inside the recorder from drained vectors,
    # not from the aggregates dict.
    assert s.input_since_last["flick_count"] == 2
    assert s.input_since_last["flick_mag_max"] == 13.0  # (-5,12) → 13


@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_info", return_value=(None, "", ""))
def test_capture_sample_grows_hw_recent_up_to_depth(_info, _shot, recorder):
    """Successive captures append, capped at _HW_RECENT_DEPTH."""
    for _ in range(_HW_RECENT_DEPTH + 2):
        recorder._capture_sample()
    s = recorder.get_samples()[-1]
    assert len(s.hw_recent) == _HW_RECENT_DEPTH


@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_info", return_value=(None, "", ""))
def test_capture_sample_empty_hw_without_hw_monitor(_info, _shot, monitor):
    rec = DataRecorder(input_monitor=monitor, hardware_monitor=None,
                       window_seconds=30, sample_interval=5)
    rec._capture_sample()
    s = rec.get_samples()[0]
    # Without a HW monitor, the snapshot is still populated (one slot per tick)
    # but all readings are None — feature engineering coerces them to zeros.
    assert len(s.hw_recent) == 1
    assert s.hw_recent[0]["cpu_percent"] is None
    assert s.hw_recent[0]["gpu_present"] == 0


@patch("src.dev.recorder.take_screenshot")
@patch("src.dev.recorder.get_active_window_info", return_value=((0, 0, 100, 100), "app", "title"))
def test_capture_sample_stores_screenshot(_info, mock_shot, recorder):
    img = _rgb_image()
    mock_shot.return_value = img
    recorder._capture_sample()
    assert recorder.get_samples()[0].screenshot is img


@patch("src.dev.recorder.take_screenshot", side_effect=RuntimeError("boom"))
@patch("src.dev.recorder.get_active_window_info", return_value=((0, 0, 1, 1), "app", "title"))
def test_capture_sample_swallows_exceptions(_info, _shot, recorder):
    recorder._hw.get_latest.side_effect = RuntimeError("hw boom")
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


def test_export_samples_jsonl_has_new_schema(tmp_path, recorder):
    _fill(recorder, n=2)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        lines = zf.read("samples.jsonl").decode().strip().split("\n")
    for line in lines:
        obj = json.loads(line)
        assert "hw_recent" in obj
        assert "input_since_last" in obj
        assert "app_name" in obj
        assert "window_title" in obj
        assert "screenshot" not in obj


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
        patch("src.dev.recorder.get_active_window_info", return_value=(None, "", "")),
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
    assert isinstance(s.hw_recent, list)
    assert isinstance(s.input_since_last, dict)


def test_export_roundtrip(tmp_path):
    monitor = _mock_input_monitor()
    hw = _mock_hw_monitor()
    with (
        patch("src.dev.recorder.get_active_window_info", return_value=(None, "", "")),
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
        meta = json.loads(zf.read("metadata.json"))
        lines = zf.read("samples.jsonl").decode().strip().split("\n")

    assert meta["sample_count"] == 4
    assert all(json.loads(l)["label"] == "Gaming" for l in lines)
    assert all("hw_recent" in json.loads(l) for l in lines)
