"""
Modern Navigation Bar Component
Provides clean navigation with page links, settings, and profile.
"""

import tkinter as tk
from typing import Callable, Dict

from .theme import ModernTheme
from .network_indicator import NetworkIndicator


class ModernNavbar(tk.Frame):
    """Modern navigation bar with page links and user controls."""

    def __init__(self, parent, pages: list, on_page_change: Callable, backend_url: str = "http://localhost:8000", **kwargs):
        """Initialize navbar.

        Args:
            parent: Parent widget
            pages: List of tuples (page_id, display_name)
            on_page_change: Callback function when page changes
            backend_url: Backend server URL for network indicator
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, bg=ModernTheme.SURFACE_ELEVATED, height=50, **kwargs)
        self.pack_propagate(False)

        self.on_page_change = on_page_change
        self.nav_labels: Dict[str, tk.Label] = {}

        self._create_left_nav(pages)
        self._create_right_nav(backend_url)

    def _create_left_nav(self, pages: list):
        """Create left side navigation with page links."""
        left_nav = tk.Frame(self, bg=ModernTheme.SURFACE_ELEVATED)
        left_nav.pack(side=tk.LEFT, padx=ModernTheme.PADDING_LARGE, pady=ModernTheme.PADDING_MEDIUM)

        for i, (page_id, display_name) in enumerate(pages):
            if i > 0:
                # Add spacing between links
                tk.Frame(left_nav, bg=ModernTheme.SURFACE_ELEVATED, width=20).pack(side=tk.LEFT)

            label = tk.Label(
                left_nav,
                text=display_name,
                bg=ModernTheme.SURFACE_ELEVATED,
                fg=ModernTheme.TEXT_PRIMARY,
                font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL),
                cursor="hand2",
                padx=10,
                pady=5
            )
            label.pack(side=tk.LEFT)

            # Bind click event
            label.bind("<Button-1>", lambda e, p=page_id: self.on_page_change(p))

            # Bind hover events
            label.bind("<Enter>", lambda e, l=label: self._on_hover(l))
            label.bind("<Leave>", lambda e, l=label, p=page_id: self._on_leave(l, p))

            self.nav_labels[page_id] = label

    def _create_right_nav(self, backend_url: str):
        """Create right side with settings and profile."""
        right_nav = tk.Frame(self, bg=ModernTheme.SURFACE_ELEVATED)
        right_nav.pack(side=tk.RIGHT, padx=ModernTheme.PADDING_LARGE, pady=ModernTheme.PADDING_MEDIUM)

        # Settings icon
        settings_label = tk.Label(
            right_nav,
            text="⚙",
            bg=ModernTheme.SURFACE_ELEVATED,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, 16),
            cursor="hand2"
        )
        settings_label.pack(side=tk.LEFT, padx=(0, 15))

        # Profile icon container with network indicator
        profile_container = tk.Frame(right_nav, bg=ModernTheme.SURFACE_ELEVATED)
        profile_container.pack(side=tk.LEFT)

        # Profile icon
        profile_icon = tk.Label(
            profile_container,
            text="👤",
            bg=ModernTheme.SURFACE_ELEVATED,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, 20),
            cursor="hand2"
        )
        profile_icon.pack()

        # Network indicator at bottom-right of profile
        self.network_indicator = NetworkIndicator(
            profile_container,
            backend_url=backend_url
        )
        self.network_indicator.place(relx=1.0, rely=1.0, anchor="se")

    def _on_hover(self, label: tk.Label):
        """Handle mouse hover over nav link."""
        # Don't change color if it's the active page (will be PRIMARY)
        if label.cget("fg") != ModernTheme.PRIMARY:
            label.config(fg=ModernTheme.PRIMARY)

    def _on_leave(self, label: tk.Label, page_id: str):
        """Handle mouse leave from nav link."""
        # Restore color based on whether it's active
        if page_id == getattr(self, '_active_page', None):
            label.config(fg=ModernTheme.PRIMARY)
        else:
            label.config(fg=ModernTheme.TEXT_PRIMARY)

    def set_active_page(self, page_id: str):
        """Highlight the active page.

        Args:
            page_id: ID of the active page
        """
        self._active_page = page_id

        for pid, label in self.nav_labels.items():
            if pid == page_id:
                label.config(
                    fg=ModernTheme.PRIMARY,
                    font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL, "bold")
                )
            else:
                label.config(
                    fg=ModernTheme.TEXT_PRIMARY,
                    font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL)
                )
