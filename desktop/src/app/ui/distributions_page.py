"""Distributions page — keyboard heatmap and input distribution charts."""

import tkinter as tk
from tkinter import ttk, filedialog

from src.core.theme import ModernTheme
from src.core.input_monitor import _HEATMAP_INTERVALS
from src.app.ui.keyboard_heatmap import KeyboardHeatmapView
from src.app.ui.distribution_charts import DistributionChartsView


class DistributionsPage(tk.Frame):
    """Page with two selectable sub-views: keyboard heatmap and distribution charts."""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=ModernTheme.BACKGROUND_DARK)
        self.controller = controller
        self._active_view = "heatmap"

        # ---- Top bar: toggle buttons + interval selector ----
        top_bar = tk.Frame(self, bg=ModernTheme.BACKGROUND_DARK)
        top_bar.pack(fill=tk.X, padx=20, pady=(15, 5))

        # Toggle buttons
        toggle_frame = tk.Frame(top_bar, bg=ModernTheme.BACKGROUND_DARK)
        toggle_frame.pack(side=tk.LEFT)

        self._heatmap_btn = tk.Button(
            toggle_frame,
            text="Heatmap",
            command=lambda: self._switch_view("heatmap"),
            bd=0, relief="flat", cursor="hand2",
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL, "bold"),
            padx=16, pady=6,
        )
        self._heatmap_btn.pack(side=tk.LEFT, padx=(0, 2))

        self._charts_btn = tk.Button(
            toggle_frame,
            text="Charts",
            command=lambda: self._switch_view("charts"),
            bd=0, relief="flat", cursor="hand2",
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL, "bold"),
            padx=16, pady=6,
        )
        self._charts_btn.pack(side=tk.LEFT)

        # Export button (right side, before interval)
        tk.Button(
            top_bar,
            text="Export",
            command=self._export,
            bd=0, relief="flat", cursor="hand2",
            bg=ModernTheme.SURFACE_ELEVATED,
            fg=ModernTheme.TEXT_PRIMARY,
            activebackground=ModernTheme.BACKGROUND_LIGHT,
            activeforeground=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL),
            padx=10, pady=3,
        ).pack(side=tk.RIGHT, padx=(0, 15))

        # Interval selector (right side)
        interval_frame = tk.Frame(top_bar, bg=ModernTheme.BACKGROUND_DARK)
        interval_frame.pack(side=tk.RIGHT)

        tk.Label(
            interval_frame,
            text="Interval:",
            bg=ModernTheme.BACKGROUND_DARK,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL),
        ).pack(side=tk.LEFT, padx=(0, 5))

        self._interval_var = tk.StringVar(value="5s")
        interval_values = [label for _, label in _HEATMAP_INTERVALS]
        self._interval_menu = ttk.Combobox(
            interval_frame,
            textvariable=self._interval_var,
            values=interval_values,
            state="readonly",
            width=6,
        )
        self._interval_menu.pack(side=tk.LEFT)
        self._interval_menu.bind("<<ComboboxSelected>>", lambda _: self.update_view())

        # ---- Content area: stacked views ----
        self._content = tk.Frame(self, bg=ModernTheme.BACKGROUND_DARK)
        self._content.pack(fill=tk.BOTH, expand=True)
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)

        self._heatmap_view = KeyboardHeatmapView(self._content, self)
        self._charts_view = DistributionChartsView(self._content, self)

        for view in (self._heatmap_view, self._charts_view):
            view.grid(row=0, column=0, sticky="nsew")

        # Show default view
        self._update_toggle_styles()
        self._heatmap_view.tkraise()

    # ---- View switching -----------------------------------------------

    def _switch_view(self, name: str) -> None:
        self._active_view = name
        self._update_toggle_styles()
        if name == "heatmap":
            self._heatmap_view.tkraise()
        else:
            self._charts_view.tkraise()
        self.update_view()

    def _update_toggle_styles(self) -> None:
        active_cfg = {
            "bg": ModernTheme.PRIMARY,
            "fg": "#ffffff",
            "activebackground": ModernTheme.PRIMARY_HOVER,
            "activeforeground": "#ffffff",
        }
        inactive_cfg = {
            "bg": ModernTheme.SURFACE,
            "fg": ModernTheme.TEXT_PRIMARY,
            "activebackground": ModernTheme.SURFACE_ELEVATED,
            "activeforeground": ModernTheme.TEXT_PRIMARY,
        }

        if self._active_view == "heatmap":
            self._heatmap_btn.config(**active_cfg)
            self._charts_btn.config(**inactive_cfg)
        else:
            self._heatmap_btn.config(**inactive_cfg)
            self._charts_btn.config(**active_cfg)

    # ---- Export -------------------------------------------------------

    def _export(self) -> None:
        if self._active_view == "charts":
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF Document", "*.pdf")],
                title="Export Distribution Charts",
            )
            if path:
                fig = self._charts_view.fig
                fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".eps",
                filetypes=[("EPS Vector", "*.eps"), ("PostScript", "*.ps")],
                title="Export Keyboard Heatmap",
            )
            if path:
                self._heatmap_view.export(path)

    # ---- Data refresh -------------------------------------------------

    def update_view(self) -> None:
        """Called by controller on page show and on each tick."""
        heatmaps: dict[str, dict[str, int]] = {}
        flicks: list[tuple[int, int]] = []
        if self.controller.input_monitor:
            heatmaps = self.controller.input_monitor.get_key_heatmaps()
            flicks = self.controller.input_monitor.get_flicks()

        interval = self._interval_var.get()
        counts = heatmaps.get(interval, {})

        if self._active_view == "heatmap":
            self._heatmap_view.update_data(counts)
        else:
            self._charts_view.update_data(counts, flicks)
