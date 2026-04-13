"""
Network Connection Indicator Widget
Shows real-time connection status to backend server.
"""

import tkinter as tk
import threading
import time
from typing import Optional
import urllib.request
import urllib.error

from src.core.theme import ModernTheme


class NetworkIndicator(tk.Frame):
    """Widget displaying network connection status."""

    def __init__(self, parent, backend_url: str = "http://localhost:8000", **kwargs):
        """Initialize network indicator.

        Args:
            parent: Parent widget
            backend_url: Backend server URL to check
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, bg=ModernTheme.SURFACE_ELEVATED, **kwargs)

        self.backend_url = backend_url
        self.check_interval = 5  # seconds
        self.is_running = True
        self.current_status = "checking"

        self._setup_ui()
        self._start_monitoring()

    def _setup_ui(self):
        """Setup the indicator UI - just a status dot."""
        # Status indicator dot (small circle)
        self.status_canvas = tk.Canvas(
            self,
            width=10,
            height=10,
            bg=ModernTheme.SURFACE_ELEVATED,
            highlightthickness=0
        )
        self.status_canvas.pack()

        # Draw initial dot
        self.status_dot = self.status_canvas.create_oval(
            1, 1, 9, 9,
            fill=ModernTheme.STATUS_CONNECTING,
            outline=""
        )

        # Tooltip on hover
        self._create_tooltip()

    def _create_tooltip(self):
        """Create tooltip showing backend URL."""
        self.tooltip = None
        self.bind("<Enter>", self._show_tooltip)
        self.bind("<Leave>", self._hide_tooltip)

    def _show_tooltip(self, event):
        """Show tooltip with backend status."""
        x = self.winfo_rootx() + 15
        y = self.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        # Status text based on current status
        status_text = {
            "online": "Backend: Online",
            "offline": "Backend: Offline",
            "checking": "Backend: Checking..."
        }.get(self.current_status, "Backend: Unknown")

        label = tk.Label(
            self.tooltip,
            text=f"{status_text}\n{self.backend_url}",
            bg=ModernTheme.BACKGROUND_DARK,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL),
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=4
        )
        label.pack()

    def _hide_tooltip(self, event):
        """Hide tooltip."""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def _start_monitoring(self):
        """Start background thread to monitor connection."""
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Background loop to check connection status."""
        while self.is_running:
            status = self._check_connection()

            # Update UI in main thread
            try:
                self.after(0, self._update_status, status)
            except:
                break

            time.sleep(self.check_interval)

    def _check_connection(self) -> str:
        """Check if backend is reachable.

        Returns:
            Status string: "online", "offline", or "checking"
        """
        try:
            # Try to connect with short timeout
            req = urllib.request.Request(
                self.backend_url,
                headers={'User-Agent': 'Adouga-Desktop/1.0'}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return "online"
                else:
                    return "offline"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return "offline"
        except Exception as e:
            print(f"Connection check error: {e}")
            return "offline"

    def _update_status(self, status: str):
        """Update the indicator display.

        Args:
            status: New status ("online", "offline", "checking")
        """
        self.current_status = status

        # Update colors based on status
        if status == "online":
            color = ModernTheme.STATUS_ONLINE
        elif status == "offline":
            color = ModernTheme.STATUS_OFFLINE
        else:
            color = ModernTheme.STATUS_CONNECTING

        # Update dot color
        self.status_canvas.itemconfig(self.status_dot, fill=color)

    def stop(self):
        """Stop the monitoring thread."""
        self.is_running = False

    def destroy(self):
        """Clean up resources."""
        self.stop()
        super().destroy()
