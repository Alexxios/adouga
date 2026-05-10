"""Pie chart (key distribution) + polar histogram (flick directions)."""

import math
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.core.theme import ModernTheme


class DistributionChartsView(tk.Frame):
    """Side-by-side pie chart and polar histogram for input distributions."""

    _TOP_N = 10  # number of keys shown individually in the pie chart

    def __init__(self, parent, page, **kwargs):
        super().__init__(parent, bg=ModernTheme.BACKGROUND_DARK, **kwargs)
        self.page = page

        self.fig = Figure(figsize=(10, 5), dpi=100,
                          facecolor=ModernTheme.BACKGROUND_DARK)

        self.ax_pie = self.fig.add_subplot(121)
        self.ax_polar = self.fig.add_subplot(122, projection="polar")
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    # ---- Public API ---------------------------------------------------

    def update_data(self, key_counts: dict[str, int], flicks: list[tuple[int, int]]) -> None:
        ModernTheme.style_matplotlib_figure(self.fig)
        self._draw_pie(key_counts)
        self._draw_polar(flicks)
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    # ---- Pie chart ----------------------------------------------------

    def _draw_pie(self, counts: dict[str, int]) -> None:
        ax = self.ax_pie
        ax.clear()
        ax.set_facecolor(ModernTheme.BACKGROUND_DARK)

        if not counts:
            ax.text(
                0.5, 0.5, "No key data",
                ha="center", va="center",
                color=ModernTheme.TEXT_SECONDARY,
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_title(
                "Key Press Distribution",
                color=ModernTheme.TEXT_PRIMARY, fontsize=12, fontweight="bold",
            )
            return

        sorted_keys = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = sorted_keys[: self._TOP_N]
        other = sum(v for _, v in sorted_keys[self._TOP_N :])

        labels = [k.upper() if len(k) == 1 else k.capitalize() for k, _ in top]
        values = [v for _, v in top]
        if other > 0:
            labels.append("Other")
            values.append(other)

        colors = self._pie_colors(len(labels))
        wedges, texts, autotexts = ax.pie(
            values,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": ModernTheme.TEXT_PRIMARY, "fontsize": 8},
            pctdistance=0.8,
        )
        for t in autotexts:
            t.set_fontsize(7)
            t.set_color(ModernTheme.TEXT_SECONDARY)

        ax.legend(
            wedges, labels,
            loc="center left", bbox_to_anchor=(-0.35, 0.5),
            fontsize=7, framealpha=0.5,
            facecolor=ModernTheme.BACKGROUND_DARK,
            edgecolor=ModernTheme.BORDER,
            labelcolor=ModernTheme.TEXT_SECONDARY,
        )

        ax.set_title(
            "Key Press Distribution",
            color=ModernTheme.TEXT_PRIMARY, fontsize=12, fontweight="bold",
        )

    # ---- Polar histogram (flick directions) ---------------------------

    _N_BINS = 16

    def _draw_polar(self, flicks: list[tuple[int, int]]) -> None:
        ax = self.ax_polar
        ax.clear()
        ax.set_facecolor(ModernTheme.BACKGROUND_DARK)
        ModernTheme.style_matplotlib_axes(ax, polar=True)
        ax.set_yticklabels([])

        if not flicks:
            ax.set_title(
                "Flick Directions",
                color=ModernTheme.TEXT_PRIMARY, fontsize=12, fontweight="bold",
            )
            return

        # Bin flick angles into N_BINS sectors
        bin_width = 2 * math.pi / self._N_BINS
        bins = [0] * self._N_BINS
        for dx, dy in flicks:
            angle = math.atan2(-dy, dx)  # screen-space
            if angle < 0:
                angle += 2 * math.pi
            idx = int(angle / bin_width) % self._N_BINS
            bins[idx] += 1

        angles = [i * bin_width for i in range(self._N_BINS)]
        max_bin = max(bins) or 1

        colors = [self._polar_bar_color(b, max_bin) for b in bins]
        bars = ax.bar(
            angles, bins,
            width=bin_width * 0.85,
            color=colors,
            edgecolor=ModernTheme.BORDER,
            linewidth=0.5,
            alpha=0.85,
        )

        # Count labels on bars
        for angle, count, bar in zip(angles, bins, bars):
            if count > 0:
                ax.text(
                    angle, count + max_bin * 0.05,
                    str(count),
                    ha="center", va="bottom",
                    color=ModernTheme.TEXT_SECONDARY,
                    fontsize=7,
                )

        # Cardinal labels
        ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2])
        ax.set_xticklabels(
            ["\u2192", "\u2191", "\u2190", "\u2193"],
            color=ModernTheme.TEXT_SECONDARY, fontsize=11,
        )
        ax.set_title(
            f"Flick Directions ({len(flicks)} total)",
            color=ModernTheme.TEXT_PRIMARY, fontsize=12, fontweight="bold",
        )

    # ---- Colour helpers -----------------------------------------------

    @staticmethod
    def _pie_colors(n: int) -> list[str]:
        """Generate *n* visually distinct colours on a blue-magenta ramp."""
        palette = [
            "#3b82f6", "#8b5cf6", "#d946ef", "#ec4899", "#f43f5e",
            "#f97316", "#eab308", "#22c55e", "#06b6d4", "#6366f1",
            "#a855f7", "#14b8a6",
        ]
        return [palette[i % len(palette)] for i in range(n)]

    @staticmethod
    def _polar_bar_color(count: int, max_count: int) -> str:
        if count == 0:
            return "#2a2a2a"
        t = count / max_count
        r = int(59 + t * (139 - 59))
        g = int(130 + t * (92 - 130))
        b = int(246 + t * (246 - 246))
        return f"#{r:02x}{g:02x}{b:02x}"
