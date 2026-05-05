"""Tests for src.feature_engineering — tabular feature extraction."""

import math

import pytest

from src.feature_engineering import (
    TABULAR_DIM,
    _HEATMAP_LABELS,
    _HW_RECENT_DEPTH,
    _TOP_KEYS_PER_WINDOW,
    _app_hash_block,
    _heatmap_block,
    _heatmap_window_stats,
    _hw_recent_block,
    _shannon_entropy,
    _top_n_block,
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

_HEATMAPS = {
    "1s":  {"w": 1},
    "5s":  {"w": 3, "a": 1},
    "15s": {"w": 5, "a": 2},
    "30s": {"w": 8, "a": 3, "space": 1},
    "1m":  {"w": 12, "a": 5, "space": 2},
    "3m":  {"w": 20, "a": 9, "space": 4, "shift": 1},
}

_FULL_SAMPLE = {
    "timestamp": 1000.0,
    "label": "Gaming",
    "app_name": "valorant.exe",
    "window_title": "VALORANT",
    "hw_recent": [_HW_SLOT_A, _HW_SLOT_B],
    "input_since_last": _INPUT_AGG,
    "key_heatmaps": _HEATMAPS,
}


# ---------------------------------------------------------------------------
# TABULAR_DIM constant
# ---------------------------------------------------------------------------

def test_tabular_dim_is_149():
    assert TABULAR_DIM == 149


def test_block_sizes_sum_to_tabular_dim():
    """Sanity check the layout invariant in case any block size drifts."""
    hw_block_size = _HW_RECENT_DEPTH * 10
    input_block_size = 6
    flick_block_size = 5
    app_hash_size = 16
    heatmap_block_size = (
        len(_HEATMAP_LABELS) * 4
        + len(_HEATMAP_LABELS) * _TOP_KEYS_PER_WINDOW
    )
    total = (
        hw_block_size + input_block_size + flick_block_size
        + app_hash_size + heatmap_block_size
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
        "key_heatmaps": {},
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


def test_changing_heatmaps_changes_features():
    a = dict(_FULL_SAMPLE)
    b = dict(_FULL_SAMPLE, key_heatmaps=dict(_HEATMAPS, **{
        "3m": {"w": 50, "a": 30, "space": 20, "shift": 10},
    }))
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


# ---------------------------------------------------------------------------
# Heatmap block
# ---------------------------------------------------------------------------

def test_shannon_entropy_uniform_is_log2_n():
    counts = {"w": 1, "a": 1, "s": 1, "d": 1}
    assert _shannon_entropy(counts) == pytest.approx(2.0, abs=1e-6)


def test_shannon_entropy_single_key_is_zero():
    assert _shannon_entropy({"w": 100}) == pytest.approx(0.0, abs=1e-6)


def test_shannon_entropy_empty_is_zero():
    assert _shannon_entropy({}) == 0.0


def test_heatmap_window_stats_empty():
    assert _heatmap_window_stats({}) == [0.0, 0.0, 0.0, 0.0]


def test_heatmap_window_stats_shape():
    stats = _heatmap_window_stats({"w": 3, "a": 1})
    assert len(stats) == 4
    assert stats[0] == pytest.approx(math.log1p(4.0), abs=1e-6)  # total
    assert stats[1] == 2.0                                       # unique
    assert stats[2] == pytest.approx(math.log1p(3.0), abs=1e-6)  # max


def test_heatmap_block_size():
    block = _heatmap_block({})
    expected = len(_HEATMAP_LABELS) * 4 + len(_HEATMAP_LABELS) * _TOP_KEYS_PER_WINDOW
    assert len(block) == expected


def test_heatmap_block_includes_all_windows():
    block = _heatmap_block({
        label: {"w": i + 1} for i, label in enumerate(_HEATMAP_LABELS)
    })
    # Each window's first stat is log1p(total); total here is i+1.
    for i in range(len(_HEATMAP_LABELS)):
        assert block[i * 4] == pytest.approx(math.log1p(i + 1), abs=1e-6)


def test_heatmap_block_top_n_section_after_stats():
    """Top-N block follows the stats block; first slot of each window is log1p(i+1)."""
    block = _heatmap_block({
        label: {"w": i + 1} for i, label in enumerate(_HEATMAP_LABELS)
    })
    stats_len = len(_HEATMAP_LABELS) * 4
    for i in range(len(_HEATMAP_LABELS)):
        slot = stats_len + i * _TOP_KEYS_PER_WINDOW
        assert block[slot] == pytest.approx(math.log1p(i + 1), abs=1e-6)
        # Remaining top-N slots for that window are zero-padded.
        for j in range(1, _TOP_KEYS_PER_WINDOW):
            assert block[slot + j] == 0.0


# ---------------------------------------------------------------------------
# Top-N block — sorted, padded, log1p
# ---------------------------------------------------------------------------

def test_top_n_block_empty_window_is_zeros():
    assert _top_n_block({}) == [0.0] * _TOP_KEYS_PER_WINDOW


def test_top_n_block_sorts_descending():
    block = _top_n_block({"a": 2, "b": 9, "c": 5})
    assert len(block) == _TOP_KEYS_PER_WINDOW
    assert block[0] == pytest.approx(math.log1p(9), abs=1e-6)
    assert block[1] == pytest.approx(math.log1p(5), abs=1e-6)
    assert block[2] == pytest.approx(math.log1p(2), abs=1e-6)
    # Remaining slots zero-padded.
    assert all(v == 0.0 for v in block[3:])


def test_top_n_block_truncates_when_more_keys_than_n():
    counts = {f"k{i}": i + 1 for i in range(_TOP_KEYS_PER_WINDOW + 4)}
    block = _top_n_block(counts)
    assert len(block) == _TOP_KEYS_PER_WINDOW
    # Largest count is _TOP_KEYS_PER_WINDOW + 4; smallest kept is 5 (i.e., counts[4]).
    assert block[0] == pytest.approx(math.log1p(_TOP_KEYS_PER_WINDOW + 4), abs=1e-6)
    assert block[-1] == pytest.approx(math.log1p(5), abs=1e-6)


def test_top_n_block_is_anonymous():
    """Permuting key names must not change the block — only counts matter."""
    a = _top_n_block({"w": 5, "a": 3, "s": 1})
    b = _top_n_block({"q": 5, "e": 3, "r": 1})
    assert a == b
