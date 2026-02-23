# ==========================================
# 5. MAIN APP CONTROLLER
# ==========================================
import tkinter as tk
from collections import deque

# External libraries
import psutil

from src.system_monitor import get_active_window_rect, InputMonitor
from src.ui import MonitorPage, AIPage
from src.ui.theme import ModernTheme
from src.ui.navbar import ModernNavbar
from src.utils import take_screenshot
from src.inference import ONNXClassifier

class SystemMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Adouga - Activity Detection & Observation")
        self.geometry("1200x800")
        self.configure(bg=ModernTheme.BACKGROUND_DARK)

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
        # Create modern navbar
        pages = [("AIPage", "AI Analysis"), ("MonitorPage", "Live Stats")]
        self.navbar = ModernNavbar(
            self,
            pages=pages,
            on_page_change=self.show,
            backend_url="http://0.0.0.0:7999"
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
        """Switch to a different page."""
        self.curr_page = name
        self.navbar.set_active_page(name)

        # Show the page
        f = self.frames[name]
        f.tkraise()
        f.update_view()

    def loop(self):
        # 1. Capture Stats
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_data.append(float(cpu))
        self.ram_data.append(float(ram))

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
