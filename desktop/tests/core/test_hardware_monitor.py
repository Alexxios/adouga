"""Tests for src.core.hardware_monitor.HardwareMonitor."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.hardware_monitor import HardwareMonitor, _sample_gpu_once


# ---------------------------------------------------------------------------
# _sample_gpu_once helper
# ---------------------------------------------------------------------------

def test_sample_gpu_once_returns_none_without_gputil():
    with patch.dict("sys.modules", {"GPUtil": None}):
        assert _sample_gpu_once() is None


def test_sample_gpu_once_returns_none_when_no_gpus():
    mock_gputil = MagicMock()
    mock_gputil.getGPUs.return_value = []
    with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
        assert _sample_gpu_once() is None


def test_sample_gpu_once_returns_dict_with_correct_keys():
    mock_gpu = MagicMock()
    mock_gpu.load = 0.5
    mock_gpu.memoryUsed = 2048
    mock_gpu.memoryTotal = 8192
    mock_gpu.temperature = 70
    mock_gputil = MagicMock()
    mock_gputil.getGPUs.return_value = [mock_gpu]
    with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
        result = _sample_gpu_once()
    assert result is not None
    assert result["load_percent"] == 50.0
    assert result["memory_used_gb"]  == pytest.approx(2.0,  abs=0.01)
    assert result["memory_total_gb"] == pytest.approx(8.0,  abs=0.01)
    assert result["temperature_c"]   == 70


def test_sample_gpu_once_returns_none_on_exception():
    mock_gputil = MagicMock()
    mock_gputil.getGPUs.side_effect = RuntimeError("driver error")
    with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
        assert _sample_gpu_once() is None


# ---------------------------------------------------------------------------
# HardwareMonitor — initialisation
# ---------------------------------------------------------------------------

def test_hardware_monitor_creates_empty_histories():
    hw = HardwareMonitor()
    assert hw.get_cpu_history()  == []
    assert hw.get_ram_history()  == []
    assert hw.get_gpu_history()  == []
    assert hw.get_disk_history() == []


def test_hardware_monitor_max_entries():
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=10)
    assert hw._cpu_hist.maxlen == 10


# ---------------------------------------------------------------------------
# HardwareMonitor — get_latest
# ---------------------------------------------------------------------------

def test_get_latest_returns_none_when_empty():
    hw = HardwareMonitor()
    latest = hw.get_latest()
    assert latest == {"cpu": None, "ram": None, "gpu": None, "disk": None}


def test_get_latest_returns_most_recent_entries():
    hw = HardwareMonitor()
    hw._cpu_hist.append({"timestamp": 1.0, "percent": 10.0})
    hw._cpu_hist.append({"timestamp": 2.0, "percent": 25.0})
    hw._ram_hist.append({"timestamp": 2.0, "percent": 60.0})
    hw._disk_hist.append({"timestamp": 2.0, "read_bps": 1024, "write_bps": 0})

    latest = hw.get_latest()
    assert latest["cpu"]["percent"] == 25.0
    assert latest["ram"]["percent"] == 60.0
    assert latest["gpu"] is None
    assert latest["disk"]["read_bps"] == 1024


# ---------------------------------------------------------------------------
# HardwareMonitor — _disk_delta
# ---------------------------------------------------------------------------

def test_disk_delta_computes_bytes_per_second():
    hw = HardwareMonitor(sample_interval=2)
    hw._last_disk_io = MagicMock(read_bytes=0, write_bytes=0)
    current = MagicMock(read_bytes=4000, write_bytes=2000)
    with patch("src.core.hardware_monitor.psutil") as mp:
        mp.disk_io_counters.return_value = current
        result = hw._disk_delta(1000.0)
    assert result["read_bps"]  == pytest.approx(2000.0)
    assert result["write_bps"] == pytest.approx(1000.0)
    assert result["timestamp"] == 1000.0


def test_disk_delta_returns_none_when_no_counters():
    hw = HardwareMonitor()
    hw._last_disk_io = None
    with patch("src.core.hardware_monitor.psutil") as mp:
        mp.disk_io_counters.return_value = None
        assert hw._disk_delta(1000.0) is None


def test_disk_delta_returns_none_on_exception():
    hw = HardwareMonitor()
    with patch("src.core.hardware_monitor.psutil") as mp:
        mp.disk_io_counters.side_effect = OSError("no disk")
        assert hw._disk_delta(1000.0) is None


# ---------------------------------------------------------------------------
# HardwareMonitor — _sample
# ---------------------------------------------------------------------------

def _make_psutil_mock(cpu=30.0, ram_pct=50.0, ram_used=4e9, ram_total=8e9, freq_mhz=2500.0):
    mp = MagicMock()
    mp.cpu_percent.return_value = cpu
    mp.cpu_freq.return_value = MagicMock(current=freq_mhz)
    mem = MagicMock()
    mem.percent = ram_pct
    mem.used    = ram_used
    mem.total   = ram_total
    mp.virtual_memory.return_value = mem
    mp.disk_io_counters.return_value = None
    return mp


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_sample_appends_cpu_entry(mock_gpu):
    hw = HardwareMonitor()
    hw._last_disk_io = None
    with patch("src.core.hardware_monitor.psutil", _make_psutil_mock(cpu=42.0, freq_mhz=3000.0)):
        hw._sample()
    hist = hw.get_cpu_history()
    assert len(hist) == 1
    assert hist[0]["percent"]  == 42.0
    assert hist[0]["freq_ghz"] == pytest.approx(3.0, abs=0.001)


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_sample_appends_ram_entry(mock_gpu):
    hw = HardwareMonitor()
    hw._last_disk_io = None
    with patch("src.core.hardware_monitor.psutil",
               _make_psutil_mock(ram_pct=75.0, ram_used=12e9, ram_total=16e9)):
        hw._sample()
    hist = hw.get_ram_history()
    assert hist[0]["percent"]  == 75.0
    assert hist[0]["used_gb"]  == pytest.approx(12.0, abs=0.01)
    assert hist[0]["total_gb"] == pytest.approx(16.0, abs=0.01)


@patch("src.core.hardware_monitor._sample_gpu_once")
def test_sample_appends_gpu_entry_when_available(mock_gpu):
    gpu_reading = {"load_percent": 55.0, "memory_used_gb": 4.0,
                   "memory_total_gb": 8.0, "temperature_c": 70}
    mock_gpu.return_value = gpu_reading
    hw = HardwareMonitor()
    hw._last_disk_io = None
    with patch("src.core.hardware_monitor.psutil", _make_psutil_mock()):
        hw._sample()
    hist = hw.get_gpu_history()
    assert len(hist) == 1
    assert hist[0]["load_percent"] == 55.0
    assert "timestamp" in hist[0]


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_sample_skips_gpu_entry_when_unavailable(mock_gpu):
    hw = HardwareMonitor()
    hw._last_disk_io = None
    with patch("src.core.hardware_monitor.psutil", _make_psutil_mock()):
        hw._sample()
    assert hw.get_gpu_history() == []


@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_sample_appends_disk_entry(mock_gpu):
    hw = HardwareMonitor(sample_interval=1)
    hw._last_disk_io = MagicMock(read_bytes=0, write_bytes=0)
    mp = _make_psutil_mock()
    mp.disk_io_counters.return_value = MagicMock(read_bytes=1000, write_bytes=500)
    with patch("src.core.hardware_monitor.psutil", mp):
        hw._sample()
    hist = hw.get_disk_history()
    assert len(hist) == 1
    assert hist[0]["read_bps"]  == pytest.approx(1000.0)
    assert hist[0]["write_bps"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# HardwareMonitor — rolling buffer
# ---------------------------------------------------------------------------

@patch("src.core.hardware_monitor._sample_gpu_once", return_value=None)
def test_buffer_evicts_old_entries(mock_gpu):
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=3)
    assert hw._cpu_hist.maxlen == 3
    hw._last_disk_io = None
    with patch("src.core.hardware_monitor.psutil", _make_psutil_mock()):
        for _ in range(5):
            hw._sample()
    assert len(hw.get_cpu_history()) == 3


# ---------------------------------------------------------------------------
# HardwareMonitor — lifecycle
# ---------------------------------------------------------------------------

def test_start_stop_does_not_raise():
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=5)
    with (
        patch("src.core.hardware_monitor.psutil") as mp,
        patch("src.core.hardware_monitor._sample_gpu_once", return_value=None),
    ):
        mp.cpu_percent.return_value = 10.0
        mp.cpu_freq.return_value = MagicMock(current=2000.0)
        mp.virtual_memory.return_value = MagicMock(percent=40.0, used=4e9, total=8e9)
        mp.disk_io_counters.return_value = None
        hw.start()
        time.sleep(0.1)
        hw.stop()


def test_double_start_is_safe():
    hw = HardwareMonitor()
    with (
        patch("src.core.hardware_monitor.psutil") as mp,
        patch("src.core.hardware_monitor._sample_gpu_once", return_value=None),
    ):
        mp.cpu_percent.return_value = 0.0
        mp.cpu_freq.return_value = None
        mp.virtual_memory.return_value = MagicMock(percent=0.0, used=0, total=1)
        mp.disk_io_counters.return_value = None
        hw.start()
        hw.start()  # must not raise or spawn a second thread
        hw.stop()


# ---------------------------------------------------------------------------
# Integration — real psutil (no mocking)
# ---------------------------------------------------------------------------

def test_hardware_monitor_collects_real_data():
    hw = HardwareMonitor(sample_interval=1, buffer_seconds=5)
    with patch("src.core.hardware_monitor._sample_gpu_once", return_value=None):
        hw.start()
        time.sleep(1.5)
        hw.stop()

    cpu = hw.get_cpu_history()
    ram = hw.get_ram_history()
    assert len(cpu) >= 1
    assert 0.0 <= cpu[0]["percent"] <= 100.0
    assert ram[0]["total_gb"] > 0
    assert "freq_ghz" in cpu[0]
