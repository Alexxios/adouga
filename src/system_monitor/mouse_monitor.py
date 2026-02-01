from pynput import mouse

import logging

logger = logging.getLogger(__name__)

class EventStatistics(object):
    def __init__(self):
        pass

class MouseMonitor(object):
    def __init__(self):
        self.mouse = mouse.Listener(on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll)
        self.mouse_pos = None
        self.button_states = {}

    def start(self):
        self.mouse.start()

    def stop(self):
        self.mouse.stop()

    def _on_move(self, x, y):
        if self.mouse_pos:
            vec = (x - self.mouse_pos[0], y - self.mouse_pos[1])
            # logger.debug(f"Mouse moved on vector: {repr((x, y))}")
            # logger.debug(f"Mouse moved on vector: {repr(vec)}")
        self.mouse_pos = (x, y)

    def _on_click(self, x, y, button, pressed):
        self.button_states[button] = [x, y, pressed]

    def _on_scroll(self, x, y, dx, dy):
        pass