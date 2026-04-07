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
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_input_monitor(count: int = 3, flicks=None):
    m = MagicMock()
    m.get_and_reset_count.return_value = count
    m.get_flicks.return_value = flicks if flicks is not None else [(1, 2), (-3, 4)]
    return m


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
            DataSample(
                timestamp=1000.0 + i * 5,
                label="Gaming" if i % 2 == 0 else "Not Gaming",
                cpu_percent=float(i * 10),
                ram_percent=float(i * 5),
                input_count=i,
                flick_vectors=[(i, -i)],
                screenshot=_rgb_image() if with_screenshot else None,
            )
        )


# ---------------------------------------------------------------------------
# DataSample
# ---------------------------------------------------------------------------

def test_sample_to_dict_excludes_screenshot():
    s = DataSample(1000.0, "Gaming", 42.5, 60.0, 7, [(1, 2)], screenshot=_rgb_image())
    d = s.to_dict()
    assert "screenshot" not in d
    assert d == {
        "timestamp": 1000.0,
        "label": "Gaming",
        "cpu_percent": 42.5,
        "ram_percent": 60.0,
        "input_count": 7,
        "flick_vectors": [(1, 2)],
    }


def test_sample_to_dict_is_json_serialisable():
    s = DataSample(1234567890.0, "Not Gaming", 10.0, 30.0, 0, [])
    assert "Not Gaming" in json.dumps(s.to_dict())


def test_sample_screenshot_defaults_to_none():
    s = DataSample(1.0, "Gaming", 5.0, 5.0, 0, [])
    assert s.screenshot is None


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
    recorder._samples.append(DataSample(1.0, "Gaming", 10.0, 20.0, 1, []))
    assert recorder.sample_count == 1
    recorder.clear()
    assert recorder.sample_count == 0


# ---------------------------------------------------------------------------
# DataRecorder — buffer
# ---------------------------------------------------------------------------

def test_recorder_get_samples_returns_list_copy(recorder):
    recorder._samples.append(DataSample(1.0, "Gaming", 10.0, 20.0, 1, []))
    samples = recorder.get_samples()
    assert isinstance(samples, list)
    samples.clear()
    assert recorder.sample_count == 1  # internal deque unaffected


def test_recorder_rolling_buffer_eviction(monitor):
    rec = DataRecorder(input_monitor=monitor, window_seconds=10, sample_interval=5)
    assert rec.max_samples == 2
    for i in range(5):
        rec._samples.append(DataSample(float(i), "Gaming", 0.0, 0.0, 0, []))
    assert rec.sample_count == 2
    assert rec.get_samples()[0].timestamp == 3.0
    assert rec.get_samples()[1].timestamp == 4.0


# ---------------------------------------------------------------------------
# DataRecorder — capture
# ---------------------------------------------------------------------------

@patch("src.dev.recorder.threading")
@patch("src.dev.recorder.DataRecorder._capture_sample")
def test_schedule_next_calls_capture_and_timer(mock_capture, mock_threading, recorder):
    recorder._recording = True
    recorder._schedule_next()
    mock_capture.assert_called_once()
    mock_threading.Timer.assert_called_once()


@patch("src.dev.recorder.take_screenshot", return_value=None)
@patch("src.dev.recorder.get_active_window_rect", return_value=None)
@patch("src.dev.recorder.psutil")
def test_capture_sample_appends_to_buffer(mock_psutil, _rect, _shot, recorder):
    mock_psutil.cpu_percent.return_value = 55.0
    mock_psutil.virtual_memory.return_value.percent = 70.0
    recorder._current_label = "Gaming"

    recorder._capture_sample()

    assert recorder.sample_count == 1
    s = recorder.get_samples()[0]
    assert s.label == "Gaming"
    assert s.cpu_percent == 55.0
    assert s.ram_percent == 70.0
    assert s.screenshot is None


@patch("src.dev.recorder.take_screenshot")
@patch("src.dev.recorder.get_active_window_rect", return_value=(0, 0, 100, 100))
@patch("src.dev.recorder.psutil")
def test_capture_sample_stores_screenshot(mock_psutil, _rect, mock_shot, recorder):
    img = _rgb_image()
    mock_shot.return_value = img
    mock_psutil.cpu_percent.return_value = 10.0
    mock_psutil.virtual_memory.return_value.percent = 20.0

    recorder._capture_sample()

    assert recorder.get_samples()[0].screenshot is img


@patch("src.dev.recorder.take_screenshot", side_effect=RuntimeError("boom"))
@patch("src.dev.recorder.get_active_window_rect", return_value=(0, 0, 100, 100))
@patch("src.dev.recorder.psutil")
def test_capture_sample_swallows_exceptions(mock_psutil, _rect, _shot, recorder):
    mock_psutil.cpu_percent.return_value = 0.0
    mock_psutil.virtual_memory.return_value.percent = 0.0
    recorder._capture_sample()  # must not raise
    assert recorder.sample_count == 0


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


def test_export_samples_jsonl(tmp_path, recorder):
    _fill(recorder, n=3)
    out = tmp_path / "session.zip"
    recorder.export_zip(out)
    with zipfile.ZipFile(out) as zf:
        lines = zf.read("samples.jsonl").decode().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "label" in obj and "cpu_percent" in obj and "screenshot" not in obj


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
    out = tmp_path / "empty.zip"
    recorder.export_zip(out)
    assert not out.exists()


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


def test_export_roundtrip(tmp_path):
    monitor = _mock_input_monitor()
    with (
        patch("src.dev.recorder.get_active_window_rect", return_value=None),
        patch("src.dev.recorder.take_screenshot", return_value=None),
        patch("src.dev.recorder.psutil") as mp,
    ):
        mp.cpu_percent.return_value = 33.0
        mp.virtual_memory.return_value.percent = 55.0
        rec = DataRecorder(input_monitor=monitor, window_seconds=20, sample_interval=1)
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
    assert all(json.loads(l)["label"] == "Gaming" for l in lines)
