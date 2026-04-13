"""
Circular Mouse Flick Dashboard
Visualizes mouse movement vectors on a polar plot.
"""
import math
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.core.theme import ModernTheme


class FlickDashboard(tk.Frame):
    """Polar chart showing recent mouse flick vectors.

    Directions are mapped to screen space: right=\u2192, down=\u2193, left=\u2190, up=\u2191.
    Newer flicks are brighter; the most recent is highlighted in orange.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.BACKGROUND_MEDIUM, **kwargs)

        tk.Label(
            self,
            text="Mouse Flick Vectors",
            bg=ModernTheme.BACKGROUND_MEDIUM,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_MEDIUM, "bold"),
        ).pack(pady=(10, 2))

        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor=ModernTheme.BACKGROUND_MEDIUM)
        self.ax = self.fig.add_subplot(111, projection="polar")
        self._style_axes()
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(ModernTheme.BACKGROUND_DARK)
        ax.tick_params(colors=ModernTheme.TEXT_SECONDARY, labelsize=8)
        ax.spines["polar"].set_color(ModernTheme.BORDER)
        ax.grid(color=ModernTheme.BORDER, alpha=0.4)
        ax.set_yticklabels([])
        # Screen-space cardinal directions: negate dy so down=\u2193 at plot bottom
        ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2])
        ax.set_xticklabels(["\u2192", "\u2191", "\u2190", "\u2193"], color=ModernTheme.TEXT_SECONDARY, fontsize=11)

    def update_view(self, flicks: list):
        """Redraw with new flick vectors.

        Args:
            flicks: list of (dx, dy) tuples in screen coords, oldest first
        """
        ModernTheme.style_matplotlib_figure(self.fig)
        self.ax.cla()
        self._style_axes()

        if flicks:
            n = len(flicks)
            # Negate dy so the polar plot matches screen directions (y-axis flipped)
            angles = [math.atan2(-dy, dx) for dx, dy in flicks]
            mags = [math.sqrt(dx * dx + dy * dy) for dx, dy in flicks]
            max_mag = max(mags)
            norm_mags = [m / max_mag for m in mags]

            for i, (angle, r) in enumerate(zip(angles, norm_mags)):
                alpha = 0.1 + 0.9 * (i / max(n - 1, 1))
                self.ax.plot([0, angle], [0, r], color=ModernTheme.PRIMARY, alpha=alpha, lw=1.2)

            # Highlight the most recent flick
            self.ax.plot(
                angles[-1], norm_mags[-1], "o",
                color=ModernTheme.WARNING, markersize=6, zorder=5
            )

        self.ax.set_ylim(0, 1)
        self.canvas.draw()
