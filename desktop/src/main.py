# ==========================================
# 5. MAIN APP CONTROLLER
# ==========================================
import tkinter as tk
from collections import deque

# External libraries
import psutil

from src.system_monitor import get_active_window_rect, InputMonitor
from src.ui import MonitorPage, AIPage
from src.utils import take_screenshot
from src.inference import ONNXClassifier

class SystemMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Active App Observer v4 - AI Edition")
        self.geometry("1000x800")

        # --- Data Stores ---
        self.history_len = 30
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

        # --- UI Setup ---
        nav = tk.Frame(self, bg="#333", height=50)
        nav.pack(side=tk.TOP, fill=tk.X)

        style = {"bg": "#555", "fg": "white", "bd": 0, "font": ("Arial", 11), "padx": 15, "pady": 8}
        tk.Button(nav, text="Live Stats", command=lambda: self.show("MonitorPage"), **style).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(nav, text="AI Analysis", command=lambda: self.show("AIPage"), **style).pack(side=tk.LEFT, padx=5, pady=5)

        self.container = tk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MonitorPage, AIPage):
            fname = F.__name__
            frame = F(self.container, self)
            self.frames[fname] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.curr_page = "AIPage"
        self.show("AIPage")

        # Start Loop - 5 Seconds interval for snappier input updates
        self.update_interval = 5000
        self.after(2000, self.loop)

    def show(self, name):
        self.curr_page = name
        f = self.frames[name]
        f.tkraise()
        f.update_view()

    def loop(self):
        # 1. Capture Stats
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_data.append(cpu)
        self.ram_data.append(ram)

        # 2. Capture Inputs (Atomic get and reset)
        input_count = 0
        if self.input_monitor:
            input_count = self.input_monitor.get_and_reset_count()
        self.input_data.append(input_count)

        # 3. Capture Screen
        rect = get_active_window_rect()
        self.current_image = take_screenshot(rect)

        # 4. Update UI
        self.frames[self.curr_page].update_view()
        self.after(self.update_interval, self.loop)


if __name__ == "__main__":
    app = SystemMonitorApp()
    app.mainloop()
