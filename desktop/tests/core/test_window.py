"""Tests for src.core.window — get_active_window_info."""

from unittest.mock import patch

from src.core.window import get_active_window_info


@patch("src.core.window.platform.system", return_value="Linux")
def test_get_active_window_info_unsupported_os_returns_empty(_sys):
    rect, app_name, title = get_active_window_info()
    assert rect is None
    assert app_name == ""
    assert title == ""


@patch("src.core.window.platform.system", return_value="Darwin")
@patch("src.core.window._get_macos_info_native",
       return_value=((0, 0, 1280, 720), "Valorant", "VALORANT"))
def test_get_active_window_info_dispatches_to_macos(_macos, _sys):
    rect, app_name, title = get_active_window_info()
    assert rect == (0, 0, 1280, 720)
    assert app_name == "Valorant"
    assert title == "VALORANT"


@patch("src.core.window.platform.system", return_value="Windows")
@patch("src.core.window._get_windows_info",
       return_value=((10, 20, 800, 600), "chrome.exe", "Twitch"))
def test_get_active_window_info_dispatches_to_windows(_win, _sys):
    rect, app_name, title = get_active_window_info()
    assert rect == (10, 20, 800, 600)
    assert app_name == "chrome.exe"
    assert title == "Twitch"
