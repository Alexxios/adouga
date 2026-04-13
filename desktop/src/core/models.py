"""Shared data models used by both the user app and dev app."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataSample:
    """Single time-stamped data point captured during recording."""

    timestamp: float
    label: str

    # Hardware histories — rolling 3-min time series from HardwareMonitor
    cpu_history: list   # [{"timestamp", "percent", "freq_ghz"}, ...]
    ram_history: list   # [{"timestamp", "percent", "used_gb", "total_gb"}, ...]
    gpu_history: list   # [{"timestamp", "load_percent", "memory_used_gb", ...}, ...]
    disk_history: list  # [{"timestamp", "read_bps", "write_bps"}, ...]

    # Input aggregate (since last sample)
    input_count: int
    flick_vectors: list  # [(dx, dy), ...]

    # Input detail — rolling 3-min window
    input_sequence: list  # [{"timestamp", "type", "value"}, ...]
    key_heatmaps: dict    # {"1s": {key: count}, "5s": ..., ...}

    # Visual (excluded from JSON export)
    screenshot: Optional[object] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """JSON-serialisable representation (screenshot excluded)."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "cpu_history": self.cpu_history,
            "ram_history": self.ram_history,
            "gpu_history": self.gpu_history,
            "disk_history": self.disk_history,
            "input_count": self.input_count,
            "flick_vectors": self.flick_vectors,
            "input_sequence": self.input_sequence,
            "key_heatmaps": self.key_heatmaps,
        }
