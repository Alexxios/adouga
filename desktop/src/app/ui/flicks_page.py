"""Mouse flick analysis page — polar chart of recent mouse movement vectors."""

import math
import tkinter as tk

from src.core.theme import ModernTheme
from src.app.ui.flick_dashboard import FlickDashboard


class FlicksPage(tk.Frame):
    """Dedicated page for visualizing mouse flick vectors."""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=ModernTheme.BACKGROUND_DARK)
        self.controller = controller

        # Header
        tk.Label(
            self,
            text="Mouse Flick Analysis",
            bg=ModernTheme.BACKGROUND_DARK,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_TITLE, "bold"),
        ).pack(pady=(15, 5))

        tk.Label(
            self,
            text="Polar chart of recent mouse movement vectors (magnitude \u2265 10 px)",
            bg=ModernTheme.BACKGROUND_DARK,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL),
        ).pack(pady=(0, 10))

        # Main row: dashboard on left, stats on right
        row = tk.Frame(self, bg=ModernTheme.BACKGROUND_DARK)
        row.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.flick_dashboard = FlickDashboard(row)
        self.flick_dashboard.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Stats panel
        stats_frame = tk.Frame(row, bg=ModernTheme.SURFACE, width=220)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        stats_frame.pack_propagate(False)

        tk.Label(
            stats_frame,
            text="Stats",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_LARGE, "bold"),
        ).pack(pady=(20, 10))

        self.stat_count = self._make_stat(stats_frame, "Flicks recorded", "0")
        self.stat_avg_mag = self._make_stat(stats_frame, "Avg magnitude", "\u2014")
        self.stat_max_mag = self._make_stat(stats_frame, "Max magnitude", "\u2014")
        self.stat_direction = self._make_stat(stats_frame, "Dominant direction", "\u2014")

    def _make_stat(self, parent, label: str, value: str):
        tk.Label(
            parent,
            text=label,
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL),
        ).pack(anchor="w", padx=15, pady=(10, 0))
        val_label = tk.Label(
            parent,
            text=value,
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_MEDIUM, "bold"),
        )
        val_label.pack(anchor="w", padx=15)
        return val_label

    def update_view(self):
        flicks = []
        if self.controller.input_monitor:
            flicks = self.controller.input_monitor.get_flicks()

        self.flick_dashboard.update_view(flicks)
        self._update_stats(flicks)

    def _update_stats(self, flicks: list):
        self.stat_count.config(text=str(len(flicks)))
        if not flicks:
            self.stat_avg_mag.config(text="\u2014")
            self.stat_max_mag.config(text="\u2014")
            self.stat_direction.config(text="\u2014")
            return

        mags = [math.sqrt(dx * dx + dy * dy) for dx, dy in flicks]
        avg_mag = sum(mags) / len(mags)
        max_mag = max(mags)
        self.stat_avg_mag.config(text=f"{avg_mag:.1f} px")
        self.stat_max_mag.config(text=f"{max_mag:.1f} px")

        # Dominant direction: bin angles into 8 sectors
        sectors = ["\u2192", "\u2197", "\u2191", "\u2196", "\u2190", "\u2199", "\u2193", "\u2198"]
        bins = [0] * 8
        for dx, dy in flicks:
            angle = math.atan2(-dy, dx)  # screen-space angle
            idx = int((angle + math.pi) / (2 * math.pi) * 8 + 0.5) % 8
            bins[idx] += 1
        dominant = sectors[bins.index(max(bins))]
        self.stat_direction.config(text=dominant)
