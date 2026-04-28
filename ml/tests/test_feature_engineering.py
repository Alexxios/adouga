"""Tests for src.feature_engineering — tabular feature extraction."""

import math

import pytest

from src.feature_engineering import (
    TABULAR_DIM,
    _GAMING_KEYS,
    _HW_RECENT_DEPTH,
    _app_hash_block,
    _hw_recent_block,
    extract_tabular_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HW_SLOT_A = {
    "timestamp": 1.0,
    "cpu_percent": 25.0,
    "cpu_freq_ghz": 3.5,
    "ram_percent": 60.0,
    "ram_used_gb": 8.0,
    "gpu_present": 1,
    "gpu_load_percent": 70.0,
    "gpu_memory_used_gb": 4.0,
    "gpu_temperature_c": 65,
    "disk_read_bps": 1024.0,
    "disk_write_bps": 512.0,
}
_HW_SLOT_B = dict(_HW_SLOT_A, timestamp=2.0, cpu_percent=80.0, disk_read_bps=4096.0)

_INPUT_AGG = {
    "key_press_count": 5,
    "mouse_click_count": 1,
    "mouse_scroll_count": 0,
    "mouse_move_count": 12,
    "total_count": 18,
    "flick_count": 2,
    "flick_mag_mean": 22.36,
    "flick_mag_max": 30.0,
    "flick_dx_mean": 5.0,
    "flick_dy_mean": 7.5,
    "gaming_keys": {
        "w": 4, "a": 0, "s": 1, "d": 0,
        "space": 2, "shift": 0, "left": 0, "right": 0,
    },
}

_FULL_SAMPLE = {
    "timestamp": 1000.0,
    "label": "Gaming",
    "app_name": "valorant.exe",
    "window_title": "VALORANT",
    "hw_recent": [_HW_SLOT_A, _HW_SLOT_B],
    "input_since_last": _INPUT_AGG,
}


# ---------------------------------------------------------------------------
# TABULAR_DIM constant
# ---------------------------------------------------------------------------

def test_tabular_dim_is_85():
    assert TABULAR_DIM == 85


def test_block_sizes_sum_to_tabular_dim():
    """Sanity check the layout invariant in case any block size drifts."""
    hw_block_size = _HW_RECENT_DEPTH * 10
    input_block_size = 6
    flick_block_size = 5
    gaming_keys_size = len(_GAMING_KEYS)
    app_hash_size = 16
    total = (
        hw_block_size + input_block_size + flick_block_size
        + gaming_keys_size + app_hash_size
    )
    assert total == TABULAR_DIM


# ---------------------------------------------------------------------------
# extract_tabular_features — shape + finiteness
# ---------------------------------------------------------------------------

def test_extract_full_sample_length():
    features = extract_tabular_features(_FULL_SAMPLE)
    assert len(features) == TABULAR_DIM


def test_extract_full_sample_all_finite():
    features = extract_tabular_features(_FULL_SAMPLE)
    assert all(math.isfinite(f) for f in features)


def test_extract_empty_sample():
    sample = {
        "timestamp": 1000.0,
        "label": "Not Gaming",
        "app_name": "",
        "window_title": "",
        "hw_recent": [],
        "input_since_last": {},
    }
    features = extract_tabular_features(sample)
    assert len(features) == TABULAR_DIM
    assert all(math.isfinite(f) for f in features)
    assert all(f == 0.0 for f in features)


# ---------------------------------------------------------------------------
# Per-sample discrimination — the bug-fix invariant
# ---------------------------------------------------------------------------

def test_changing_hw_changes_features():
    """Two consecutive samples with different HW must produce different vectors."""
    a = dict(_FULL_SAMPLE, hw_recent=[_HW_SLOT_A])
    b = dict(_FULL_SAMPLE, hw_recent=[_HW_SLOT_B])
    assert extract_tabular_features(a) != extract_tabular_features(b)


def test_changing_input_changes_features():
    a = dict(_FULL_SAMPLE)
    b = dict(_FULL_SAMPLE, input_since_last=dict(_INPUT_AGG, total_count=999))
    assert extract_tabular_features(a) != extract_tabular_features(b)


def test_changing_app_name_changes_features():
    a = dict(_FULL_SAMPLE, app_name="valorant.exe")
    b = dict(_FULL_SAMPLE, app_name="chrome.exe")
    assert extract_tabular_features(a) != extract_tabular_features(b)


# ---------------------------------------------------------------------------
# HW recent block — padding and ordering
# ---------------------------------------------------------------------------

def test_hw_recent_block_pads_when_short():
    block = _hw_recent_block([_HW_SLOT_A])
    assert len(block) == _HW_RECENT_DEPTH * 10
    # First (depth-1) slots are the cold-start zero pad.
    pad_len = (_HW_RECENT_DEPTH - 1) * 10
    assert all(v == 0.0 for v in block[:pad_len])
    # Newest slot is at the end and reflects _HW_SLOT_A.
    assert block[pad_len + 0] == 25.0  # cpu_percent


def test_hw_recent_block_truncates_when_too_long():
    long_recent = [_HW_SLOT_A] * (_HW_RECENT_DEPTH + 3)
    block = _hw_recent_block(long_recent)
    assert len(block) == _HW_RECENT_DEPTH * 10


def test_hw_recent_block_log1p_on_disk_bps():
    block = _hw_recent_block([_HW_SLOT_A])
    pad_len = (_HW_RECENT_DEPTH - 1) * 10
    # disk_read_bps is field index 8 within a slot.
    assert block[pad_len + 8] == pytest.approx(math.log1p(1024.0), abs=1e-6)


# ---------------------------------------------------------------------------
# App-hash block
# ---------------------------------------------------------------------------

def test_app_hash_empty_string_is_all_zeros():
    block = _app_hash_block("")
    assert block == [0.0] * 16


def test_app_hash_one_hot_indicator():
    block = _app_hash_block("valorant.exe")
    assert sum(block) == 1.0
    assert all(v in (0.0, 1.0) for v in block)


def test_app_hash_is_case_insensitive():
    assert _app_hash_block("Valorant.EXE") == _app_hash_block("valorant.exe")


def test_app_hash_strips_whitespace():
    assert _app_hash_block("  valorant.exe  ") == _app_hash_block("valorant.exe")


def test_app_hash_distinct_apps_likely_differ():
    """Different app names should usually land in different buckets."""
    distinct = {
        tuple(_app_hash_block(name)) for name in (
            "valorant.exe", "chrome.exe", "rocket league.exe",
            "miro", "twitch.tv", "wuthering waves",
        )
    }
    # 6 inputs × 16 buckets — expect ≥ 4 distinct buckets in practice.
    assert len(distinct) >= 4
