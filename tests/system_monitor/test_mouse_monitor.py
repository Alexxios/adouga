import time

import pytest
import unittest.mock
from pynput.mouse import Controller, Button

from src.system_monitor.mouse_monitor import MouseMonitor

def test_click():
    controller = Controller()
    controller.position = (0, 0)
    monitor = MouseMonitor()

    monitor.start()

    controller.click(Button.left)
    time.sleep(1)

    monitor.stop()

    assert {Button.left: [0, 0, False]} == monitor.button_states

def test_long_press():
    controller = Controller()
    controller.position = (0, 0)
    monitor = MouseMonitor()

    monitor.start()

    controller.press(Button.left)
    assert {Button.left: [0, 0, True]} == monitor.button_states
    time.sleep(1)
    assert {Button.left: [0, 0, True]} == monitor.button_states
    controller.release(Button.left)
    assert {Button.left: [0, 0, False]} == monitor.button_states

    monitor.stop()