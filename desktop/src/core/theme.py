"""
Modern UI Theme Configuration
Provides unified color scheme and styling for the application.
Supports dark and light themes with runtime switching.
"""


class ModernTheme:
    """Theme system with dark and light palettes.

    All pages read ``ModernTheme.X`` each time they render, so swapping the
    palette via :meth:`set_theme` is enough — no observer pattern needed.
    """

    # ---- Palette definitions (private) --------------------------------
    _PALETTES: dict[str, dict[str, str]] = {
        "dark": {
            "BACKGROUND_DARK":   "#0f0f0f",
            "BACKGROUND_MEDIUM": "#1a1a1a",
            "BACKGROUND_LIGHT":  "#252525",
            "SURFACE":           "#2a2a2a",
            "SURFACE_ELEVATED":  "#333333",
            "PRIMARY":           "#3b82f6",
            "PRIMARY_HOVER":     "#2563eb",
            "SUCCESS":           "#10b981",
            "WARNING":           "#f59e0b",
            "ERROR":             "#ef4444",
            "INFO":              "#06b6d4",
            "TEXT_PRIMARY":      "#ffffff",
            "TEXT_SECONDARY":    "#a1a1aa",
            "TEXT_DISABLED":     "#52525b",
            "BORDER":            "#3f3f46",
            "BORDER_LIGHT":      "#52525b",
            "STATUS_ONLINE":     "#10b981",
            "STATUS_OFFLINE":    "#ef4444",
            "STATUS_CONNECTING": "#f59e0b",
        },
        "light": {
            "BACKGROUND_DARK":   "#f5f5f5",
            "BACKGROUND_MEDIUM": "#e8e8e8",
            "BACKGROUND_LIGHT":  "#dedede",
            "SURFACE":           "#ffffff",
            "SURFACE_ELEVATED":  "#f0f0f0",
            "PRIMARY":           "#2563eb",
            "PRIMARY_HOVER":     "#1d4ed8",
            "SUCCESS":           "#059669",
            "WARNING":           "#d97706",
            "ERROR":             "#dc2626",
            "INFO":              "#0891b2",
            "TEXT_PRIMARY":      "#18181b",
            "TEXT_SECONDARY":    "#52525b",
            "TEXT_DISABLED":     "#a1a1aa",
            "BORDER":            "#d4d4d8",
            "BORDER_LIGHT":      "#e4e4e7",
            "STATUS_ONLINE":     "#059669",
            "STATUS_OFFLINE":    "#dc2626",
            "STATUS_CONNECTING": "#d97706",
        },
    }

    _current_theme: str = "dark"

    # ---- Color attributes (initialised from dark palette) -------------
    BACKGROUND_DARK = "#0f0f0f"
    BACKGROUND_MEDIUM = "#1a1a1a"
    BACKGROUND_LIGHT = "#252525"
    SURFACE = "#2a2a2a"
    SURFACE_ELEVATED = "#333333"

    PRIMARY = "#3b82f6"
    PRIMARY_HOVER = "#2563eb"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    INFO = "#06b6d4"

    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a1a1aa"
    TEXT_DISABLED = "#52525b"

    BORDER = "#3f3f46"
    BORDER_LIGHT = "#52525b"

    STATUS_ONLINE = "#10b981"
    STATUS_OFFLINE = "#ef4444"
    STATUS_CONNECTING = "#f59e0b"

    # ---- Theme-independent constants ----------------------------------
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_SMALL = 9
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_XLARGE = 16
    FONT_SIZE_TITLE = 20

    PADDING_SMALL = 5
    PADDING_MEDIUM = 10
    PADDING_LARGE = 15
    PADDING_XLARGE = 20

    BORDER_WIDTH = 1

    # ---- Theme switching ----------------------------------------------

    @classmethod
    def set_theme(cls, name: str) -> None:
        """Swap all color attributes to the named palette."""
        palette = cls._PALETTES[name]
        for attr, value in palette.items():
            setattr(cls, attr, value)
        cls._current_theme = name

    @classmethod
    def current_theme(cls) -> str:
        return cls._current_theme

    @classmethod
    def toggle_theme(cls) -> str:
        """Switch between dark and light. Returns the new theme name."""
        new = "light" if cls._current_theme == "dark" else "dark"
        cls.set_theme(new)
        return new

    # ---- Widget recoloring --------------------------------------------

    @classmethod
    def recolor_widget_tree(cls, widget) -> None:
        """Recursively update bg/fg of a widget tree to match the current theme."""
        # Build a mapping of all old-palette colours -> their semantic attribute name
        color_to_attr: dict[str, str] = {}
        for palette in cls._PALETTES.values():
            for attr, color in palette.items():
                color_to_attr[color.lower()] = attr

        cls._recolor_recursive(widget, color_to_attr)

    @classmethod
    def _recolor_recursive(cls, widget, color_to_attr: dict) -> None:
        try:
            bg = str(widget.cget("bg")).lower()
            if bg in color_to_attr:
                widget.configure(bg=getattr(cls, color_to_attr[bg]))
        except Exception:
            pass
        try:
            fg = str(widget.cget("fg")).lower()
            if fg in color_to_attr:
                widget.configure(fg=getattr(cls, color_to_attr[fg]))
        except Exception:
            pass
        try:
            hlbg = str(widget.cget("highlightbackground")).lower()
            if hlbg in color_to_attr:
                widget.configure(highlightbackground=getattr(cls, color_to_attr[hlbg]))
        except Exception:
            pass
        for child in widget.winfo_children():
            cls._recolor_recursive(child, color_to_attr)

    # ---- Matplotlib helpers -------------------------------------------

    @classmethod
    def style_matplotlib_figure(cls, fig) -> None:
        """Apply current theme facecolor to a matplotlib Figure."""
        fig.set_facecolor(cls.BACKGROUND_DARK)

    @classmethod
    def style_matplotlib_axes(cls, ax, *, polar: bool = False) -> None:
        """Apply current theme colors to a matplotlib Axes."""
        ax.set_facecolor(cls.BACKGROUND_MEDIUM)
        ax.tick_params(colors=cls.TEXT_SECONDARY, labelsize=9)
        if polar:
            ax.spines["polar"].set_color(cls.BORDER)
            ax.grid(color=cls.BORDER, alpha=0.4)
        else:
            ax.grid(True, alpha=0.2, color=cls.BORDER)
            for spine in ax.spines.values():
                spine.set_color(cls.BORDER)

    # ---- Style helpers ------------------------------------------------

    @classmethod
    def button_style(cls, variant="primary"):
        """Get button style configuration."""
        styles = {
            "primary": {
                "bg": cls.PRIMARY,
                "fg": "#ffffff",
                "activebackground": cls.PRIMARY_HOVER,
                "activeforeground": "#ffffff",
            },
            "secondary": {
                "bg": cls.SURFACE_ELEVATED,
                "fg": cls.TEXT_PRIMARY,
                "activebackground": cls.BACKGROUND_LIGHT,
                "activeforeground": cls.TEXT_PRIMARY,
            },
            "success": {
                "bg": cls.SUCCESS,
                "fg": "#ffffff",
                "activebackground": "#059669",
                "activeforeground": "#ffffff",
            },
        }

        base_style = {
            "bd": 0,
            "relief": "flat",
            "font": (cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL, "bold"),
            "padx": cls.PADDING_LARGE,
            "pady": cls.PADDING_MEDIUM,
            "cursor": "hand2",
        }

        return {**base_style, **styles.get(variant, styles["primary"])}

    @classmethod
    def label_style(cls, variant="primary"):
        """Get label style configuration."""
        styles = {
            "primary": {
                "bg": cls.SURFACE,
                "fg": cls.TEXT_PRIMARY,
            },
            "secondary": {
                "bg": cls.SURFACE,
                "fg": cls.TEXT_SECONDARY,
            },
            "title": {
                "bg": cls.SURFACE,
                "fg": cls.TEXT_PRIMARY,
                "font": (cls.FONT_FAMILY, cls.FONT_SIZE_TITLE, "bold"),
            },
        }

        base_style = {
            "font": (cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL),
        }

        return {**base_style, **styles.get(variant, styles["primary"])}

    @classmethod
    def frame_style(cls, variant="default"):
        """Get frame style configuration."""
        styles = {
            "default": {
                "bg": cls.SURFACE,
            },
            "elevated": {
                "bg": cls.SURFACE_ELEVATED,
            },
            "dark": {
                "bg": cls.BACKGROUND_DARK,
            },
        }

        return styles.get(variant, styles["default"])

    @classmethod
    def card_style(cls):
        """Return kwargs for a card-like frame with subtle border."""
        return {
            "bg": cls.SURFACE,
            "highlightthickness": 1,
            "highlightbackground": cls.BORDER,
            "bd": 0,
        }
