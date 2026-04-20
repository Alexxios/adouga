# ==========================================
# 5. MAIN APP CONTROLLER
# ==========================================
import time as _time
import tkinter as tk
from collections import deque

from src.core.models import DataSample
from src.core.window import get_active_window_rect
from src.core.hardware_monitor import HardwareMonitor
from src.core.input_monitor import InputMonitor
from src.core.theme import ModernTheme
from src.core.screenshot import take_screenshot
from src.app.ui.monitor_page import MonitorPage
from src.app.ui.ai_page import AIPage
from src.app.ui.flicks_page import FlicksPage
from src.app.ui.distributions_page import DistributionsPage
from src.app.ui.navbar import ModernNavbar
from src.app.inference import ONNXClassifier

class SystemMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Adouga - Automatic Detection of User Gaming Activity")
        self.geometry("1200x800")
        self.configure(bg=ModernTheme.BACKGROUND_DARK)

        # --- Data Stores ---
        self.history_len = 30
        # Rich sample buffer (matching dev app's DataSample)
        self._samples: deque = deque(maxlen=self.history_len)
        # Legacy deques for MonitorPage graphs
        self.cpu_data = deque([0]*self.history_len, maxlen=self.history_len)
        self.ram_data = deque([0]*self.history_len, maxlen=self.history_len)
        self.input_data = deque([0]*self.history_len, maxlen=self.history_len)
        self.current_image = None

        # --- Initialize ONNX Classifier ---
        try:
            print("Loading ONNX model...")
            self.classifier = ONNXClassifier(use_gpu=True)
            print("✓ ONNX model loaded successfully")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.classifier = None

        # --- Start Input Monitor ---
        # Note: On macOS, this triggers "Input Monitoring" permission request
        try:
            self.input_monitor = InputMonitor()
        except Exception as e:
            print(f"Failed to start Input Monitor: {e}")
            self.input_monitor = None

        # --- Start Hardware Monitor ---
        try:
            self.hw_monitor = HardwareMonitor()
            self.hw_monitor.start()
            print("HardwareMonitor started")
        except Exception as e:
            print(f"Failed to start HardwareMonitor: {e}")
            self.hw_monitor = None

        # --- UI Setup ---
        # Create modern navbar
        pages = [
            ("AIPage", "AI Analysis"),
            ("MonitorPage", "Live Stats"),
            ("FlicksPage", "Mouse Flicks"),
            ("DistributionsPage", "Distributions"),
        ]
        self.navbar = ModernNavbar(
            self,
            pages=pages,
            on_page_change=self.show,
            on_theme_change=self._apply_theme,
            backend_url="http://0.0.0.0:7999",
        )
        self.navbar.pack(side=tk.TOP, fill=tk.X)

        # Separator line below navbar
        separator = tk.Frame(self, bg=ModernTheme.BORDER, height=1)
        separator.pack(side=tk.TOP, fill=tk.X)

        # Main container for pages
        self.container = tk.Frame(self, bg=ModernTheme.BACKGROUND_DARK)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Initialize pages
        self.frames = {}
        for F in (MonitorPage, AIPage, FlicksPage, DistributionsPage):
            fname = F.__name__
            frame = F(self.container, self)
            self.frames[fname] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.curr_page = "AIPage"
        self.show("AIPage")

        # Start Loop - 5 Seconds interval
        self.update_interval = 5000
        self.after(2000, self.loop)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def show(self, name):
        """Switch to a different page."""
        self.curr_page = name
        self.navbar.set_active_page(name)

        # Show the page
        f = self.frames[name]
        f.tkraise()
        f.update_view()

    def _apply_theme(self):
        """Re-apply theme colours to root window and all pages."""
        self.configure(bg=ModernTheme.BACKGROUND_DARK)
        self.container.configure(bg=ModernTheme.BACKGROUND_DARK)
        ModernTheme.recolor_widget_tree(self)
        self.navbar.update_view()
        for frame in self.frames.values():
            frame.update_view()

    def loop(self):
        timestamp = _time.time()

        # 1. Hardware histories from HardwareMonitor
        cpu_history = self.hw_monitor.get_cpu_history() if self.hw_monitor else []
        ram_history = self.hw_monitor.get_ram_history() if self.hw_monitor else []
        gpu_history = self.hw_monitor.get_gpu_history() if self.hw_monitor else []
        disk_history = self.hw_monitor.get_disk_history() if self.hw_monitor else []

        # 2. Input data
        input_count = 0
        flick_vectors = []
        input_sequence = []
        key_heatmaps = {}
        if self.input_monitor:
            input_count = self.input_monitor.get_and_reset_count()
            flick_vectors = self.input_monitor.get_flicks()
            input_sequence = self.input_monitor.get_input_sequence()
            key_heatmaps = self.input_monitor.get_key_heatmaps()

        # 3. Capture Screen
        rect = get_active_window_rect()
        self.current_image = take_screenshot(rect)

        # 4. Build DataSample (matching dev app structure)
        sample = DataSample(
            timestamp=timestamp,
            label="",
            cpu_history=cpu_history,
            ram_history=ram_history,
            gpu_history=gpu_history,
            disk_history=disk_history,
            input_count=input_count,
            flick_vectors=list(flick_vectors),
            input_sequence=input_sequence,
            key_heatmaps=key_heatmaps,
            screenshot=self.current_image,
        )
        self._samples.append(sample)

        # 5. Populate legacy deques for MonitorPage graphs
        latest_cpu = cpu_history[-1]["percent"] if cpu_history else 0
        latest_ram = ram_history[-1]["percent"] if ram_history else 0
        self.cpu_data.append(float(latest_cpu))
        self.ram_data.append(float(latest_ram))
        self.input_data.append(input_count)

        # 6. Update UI
        self.frames[self.curr_page].update_view()
        self.after(self.update_interval, self.loop)

    def _on_close(self) -> None:
        if self.hw_monitor:
            self.hw_monitor.stop()
        if self.input_monitor:
            self.input_monitor.stop()
        self.destroy()


if __name__ == "__main__":
    app = SystemMonitorApp()
    app.mainloop()
