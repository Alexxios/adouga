"""Tests for src.core.input_monitor — drain methods and move-counter wiring."""

from unittest.mock import patch

import pytest

# Avoid actually starting pynput listeners during tests.
with patch("src.core.input_monitor.keyboard"), \
     patch("src.core.input_monitor.mouse"):
    from src.core.input_monitor import InputMonitor, _GAMING_KEYS  # noqa: E402


@pytest.fixture
def monitor():
    with patch("src.core.input_monitor.keyboard"), \
         patch("src.core.input_monitor.mouse"):
        m = InputMonitor()
    yield m


# ---------------------------------------------------------------------------
# get_and_reset_input_aggregates
# ---------------------------------------------------------------------------

def test_aggregates_initially_zero(monitor):
    agg = monitor.get_and_reset_input_aggregates()
    assert agg["key_press_count"] == 0
    assert agg["mouse_click_count"] == 0
    assert agg["mouse_scroll_count"] == 0
    assert agg["mouse_move_count"] == 0
    assert agg["total_count"] == 0
    for key in _GAMING_KEYS:
        assert agg["gaming_keys"][key] == 0


def test_aggregates_increment_on_events(monitor):
    monitor._key_press_count = 4
    monitor._mouse_click_count = 2
    monitor._mouse_scroll_count = 1
    monitor._mouse_move_count = 7
    monitor._gaming_key_counts["w"] = 3

    agg = monitor.get_and_reset_input_aggregates()

    assert agg["key_press_count"] == 4
    assert agg["mouse_click_count"] == 2
    assert agg["mouse_scroll_count"] == 1
    assert agg["mouse_move_count"] == 7
    assert agg["total_count"] == 14
    assert agg["gaming_keys"]["w"] == 3


def test_aggregates_reset_after_drain(monitor):
    monitor._key_press_count = 5
    monitor._gaming_key_counts["a"] = 2
    monitor.get_and_reset_input_aggregates()

    second = monitor.get_and_reset_input_aggregates()
    assert second["key_press_count"] == 0
    assert second["gaming_keys"]["a"] == 0


def test_gaming_key_dict_is_a_copy(monitor):
    """Mutating the returned gaming_keys must not corrupt internal state."""
    monitor._gaming_key_counts["space"] = 9
    agg = monitor.get_and_reset_input_aggregates()
    agg["gaming_keys"]["space"] = 999
    assert monitor._gaming_key_counts["space"] == 0


# ---------------------------------------------------------------------------
# get_and_reset_flicks
# ---------------------------------------------------------------------------

def test_flicks_initially_empty(monitor):
    assert monitor.get_and_reset_flicks() == []


def test_flicks_drain(monitor):
    monitor._flicks.extend([(1, 2), (-3, 4)])
    flicks = monitor.get_and_reset_flicks()
    assert flicks == [(1, 2), (-3, 4)]
    assert monitor.get_and_reset_flicks() == []


# ---------------------------------------------------------------------------
# Listener wiring — _on_move increments mouse_move_count when flick is recorded
# ---------------------------------------------------------------------------

def test_on_move_increments_move_counter_for_significant_motion(monitor):
    monitor._last_pos = (0, 0)
    monitor._on_move(50, 50)  # well above _FLICK_MIN_MAG (10)
    agg = monitor.get_and_reset_input_aggregates()
    assert agg["mouse_move_count"] == 1


def test_on_move_ignores_small_motion(monitor):
    monitor._last_pos = (0, 0)
    monitor._on_move(2, 2)  # below threshold → no flick, no count
    agg = monitor.get_and_reset_input_aggregates()
    assert agg["mouse_move_count"] == 0


def test_on_move_does_not_inflate_counter(monitor):
    """The legacy ``counter`` (used by the UI) must NOT track mouse moves."""
    monitor._last_pos = (0, 0)
    monitor._on_move(50, 50)
    assert monitor.get_and_reset_count() == 0


# ---------------------------------------------------------------------------
# Listener wiring — _on_mouse_click / _on_mouse_scroll
# ---------------------------------------------------------------------------

def test_mouse_click_updates_aggregates(monitor):
    class _Btn:
        def __str__(self) -> str:
            return "Button.left"
    monitor._on_mouse_click(0, 0, _Btn(), True)
    agg = monitor.get_and_reset_input_aggregates()
    assert agg["mouse_click_count"] == 1


def test_mouse_scroll_updates_aggregates(monitor):
    monitor._on_mouse_scroll(0, 0, 0, 1)
    agg = monitor.get_and_reset_input_aggregates()
    assert agg["mouse_scroll_count"] == 1


def test_mouse_click_release_is_ignored(monitor):
    class _Btn:
        def __str__(self) -> str:
            return "Button.left"
    monitor._on_mouse_click(0, 0, _Btn(), False)
    assert monitor.get_and_reset_input_aggregates()["mouse_click_count"] == 0
