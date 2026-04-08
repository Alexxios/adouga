"""Tests for src.feature_engineering — tabular feature extraction."""

import math

import pytest

from src.feature_engineering import (
    TABULAR_DIM,
    _heatmap_stats,
    _shannon_entropy,
    _stats,
    extract_tabular_features,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CPU_HIST = [
    {"timestamp": 1.0, "percent": 20.0, "freq_ghz": 3.0},
    {"timestamp": 2.0, "percent": 40.0, "freq_ghz": 3.5},
    {"timestamp": 3.0, "percent": 60.0, "freq_ghz": 3.2},
]
_RAM_HIST = [
    {"timestamp": 1.0, "percent": 50.0, "used_gb": 8.0, "total_gb": 16.0},
    {"timestamp": 2.0, "percent": 55.0, "used_gb": 8.8, "total_gb": 16.0},
]
_GPU_HIST = [
    {"timestamp": 1.0, "load_percent": 70.0, "memory_used_gb": 4.0,
     "memory_total_gb": 8.0, "temperature_c": 65},
]
_DISK_HIST = [
    {"timestamp": 1.0, "read_bps": 1024.0, "write_bps": 512.0},
    {"timestamp": 2.0, "read_bps": 2048.0, "write_bps": 1024.0},
]
_SEQ = [
    {"timestamp": 999.0, "type": "key_press", "value": "a"},
    {"timestamp": 999.5, "type": "mouse_click", "value": "left"},
    {"timestamp": 999.8, "type": "mouse_scroll", "value": "scroll_up"},
]
_HEATMAPS = {
    "1s": {"a": 1}, "5s": {"a": 3}, "15s": {"a": 5},
    "30s": {"a": 8}, "1m": {"a": 12}, "3m": {"a": 20, "left": 5},
}

_FULL_SAMPLE = {
    "timestamp": 1000.0,
    "label": "Gaming",
    "cpu_history": _CPU_HIST,
    "ram_history": _RAM_HIST,
    "gpu_history": _GPU_HIST,
    "disk_history": _DISK_HIST,
    "input_count": 7,
    "flick_vectors": [(10, 20), (-5, 15)],
    "input_sequence": _SEQ,
    "key_heatmaps": _HEATMAPS,
}


# ---------------------------------------------------------------------------
# TABULAR_DIM constant
# ---------------------------------------------------------------------------

def test_tabular_dim_is_102():
    assert TABULAR_DIM == 102


# ---------------------------------------------------------------------------
# _stats helper
# ---------------------------------------------------------------------------

def test_stats_empty_returns_zeros():
    assert _stats([]) == [0.0] * 5


def test_stats_single_value():
    result = _stats([42.0])
    assert result == [42.0, 0.0, 42.0, 42.0, 42.0]


def test_stats_multiple_values():
    result = _stats([10.0, 20.0, 30.0])
    assert result[0] == pytest.approx(20.0)       # mean
    assert result[1] == pytest.approx(8.165, abs=0.01)  # std
    assert result[2] == 10.0                       # min
    assert result[3] == 30.0                       # max
    assert result[4] == 30.0                       # last


# ---------------------------------------------------------------------------
# _shannon_entropy
# ---------------------------------------------------------------------------

def test_entropy_single_key_is_zero():
    assert _shannon_entropy({"a": 10}) == 0.0


def test_entropy_uniform_two_keys():
    ent = _shannon_entropy({"a": 5, "b": 5})
    assert ent == pytest.approx(1.0, abs=0.001)


def test_entropy_empty_is_zero():
    assert _shannon_entropy({}) == 0.0


# ---------------------------------------------------------------------------
# _heatmap_stats
# ---------------------------------------------------------------------------

def test_heatmap_stats_length():
    result = _heatmap_stats(_HEATMAPS)
    assert len(result) == 32  # 6 * 4 + 8


def test_heatmap_stats_empty():
    result = _heatmap_stats({})
    assert len(result) == 32
    assert all(v == 0.0 for v in result)


# ---------------------------------------------------------------------------
# extract_tabular_features — full sample
# ---------------------------------------------------------------------------

def test_extract_full_sample_length():
    features = extract_tabular_features(_FULL_SAMPLE)
    assert len(features) == TABULAR_DIM


def test_extract_full_sample_all_finite():
    features = extract_tabular_features(_FULL_SAMPLE)
    assert all(math.isfinite(f) for f in features)


# ---------------------------------------------------------------------------
# extract_tabular_features — empty histories
# ---------------------------------------------------------------------------

def test_extract_empty_histories():
    sample = {
        "timestamp": 1000.0,
        "label": "Not Gaming",
        "cpu_history": [],
        "ram_history": [],
        "gpu_history": [],
        "disk_history": [],
        "input_count": 0,
        "flick_vectors": [],
        "input_sequence": [],
        "key_heatmaps": {},
    }
    features = extract_tabular_features(sample)
    assert len(features) == TABULAR_DIM
    assert all(math.isfinite(f) for f in features)


def test_extract_missing_gpu_flag_is_zero():
    sample = {
        "cpu_history": _CPU_HIST,
        "ram_history": _RAM_HIST,
        "gpu_history": [],
        "disk_history": _DISK_HIST,
        "input_count": 5,
        "flick_vectors": [],
        "input_sequence": [],
        "key_heatmaps": {},
    }
    features = extract_tabular_features(sample)
    # GPU: 10 (cpu) + 15 (ram) + 20 (gpu stats) = index 45 is gpu_available flag
    assert features[45] == 0.0


def test_extract_present_gpu_flag_is_one():
    features = extract_tabular_features(_FULL_SAMPLE)
    assert features[45] == 1.0


# ---------------------------------------------------------------------------
# extract_tabular_features — None freq_ghz
# ---------------------------------------------------------------------------

def test_extract_none_freq_ghz():
    sample = dict(_FULL_SAMPLE)
    sample["cpu_history"] = [{"timestamp": 1.0, "percent": 30.0, "freq_ghz": None}]
    features = extract_tabular_features(sample)
    assert len(features) == TABULAR_DIM
    assert math.isfinite(features[5])  # freq_ghz mean position
