"""Shared data models used by both the user app and dev app."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataSample:
    """Single time-stamped data point captured during recording.

    All time-series fields are aligned to *this* sample (no session-rolling
    deques). ``hw_recent`` carries the last few hardware snapshots (one per
    capture tick), and ``input_since_last`` aggregates input events that
    occurred between the previous capture and this one.
    """

    timestamp: float
    label: str

    # Foreground app metadata (empty strings when unavailable).
    app_name: str
    window_title: str

    # Last N hardware snapshots, oldest first. Each entry is a dict with
    # the keys produced by HardwareMonitor (cpu/ram/gpu/disk merged):
    #   {"cpu_percent", "cpu_freq_ghz", "ram_percent", "ram_used_gb",
    #    "gpu_load_percent", "gpu_memory_used_gb", "gpu_temperature_c",
    #    "gpu_present", "disk_read_bps", "disk_write_bps"}
    # Cold-start padding entries are empty dicts.
    hw_recent: list

    # Input aggregates strictly since the previous sample tick.
    # Shape:
    #   {"key_press_count", "mouse_click_count", "mouse_scroll_count",
    #    "mouse_move_count", "total_count",
    #    "flick_count", "flick_mag_mean", "flick_mag_max",
    #    "flick_dx_mean", "flick_dy_mean",
    #    "gaming_keys": {"w","a","s","d","space","shift","left","right"}}
    input_since_last: dict

    # Visual (excluded from JSON export)
    screenshot: Optional[object] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """JSON-serialisable representation (screenshot excluded)."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "hw_recent": self.hw_recent,
            "input_since_last": self.input_since_last,
        }
