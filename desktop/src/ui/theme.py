"""
Modern UI Theme Configuration
Provides unified color scheme and styling for the application.
"""

class ModernTheme:
    """Modern dark theme with consistent styling."""

    # Color Palette
    BACKGROUND_DARK = "#0f0f0f"
    BACKGROUND_MEDIUM = "#1a1a1a"
    BACKGROUND_LIGHT = "#252525"
    SURFACE = "#2a2a2a"
    SURFACE_ELEVATED = "#333333"

    # Accent Colors
    PRIMARY = "#3b82f6"  # Blue
    PRIMARY_HOVER = "#2563eb"
    SUCCESS = "#10b981"  # Green
    WARNING = "#f59e0b"  # Orange
    ERROR = "#ef4444"  # Red
    INFO = "#06b6d4"  # Cyan

    # Text Colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a1a1aa"
    TEXT_DISABLED = "#52525b"

    # Border Colors
    BORDER = "#3f3f46"
    BORDER_LIGHT = "#52525b"

    # Status Colors
    STATUS_ONLINE = "#10b981"
    STATUS_OFFLINE = "#ef4444"
    STATUS_CONNECTING = "#f59e0b"

    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_SMALL = 9
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_XLARGE = 16
    FONT_SIZE_TITLE = 20

    # Spacing
    PADDING_SMALL = 5
    PADDING_MEDIUM = 10
    PADDING_LARGE = 15
    PADDING_XLARGE = 20

    # Border Radius (simulated with relief)
    BORDER_WIDTH = 1

    @classmethod
    def button_style(cls, variant="primary"):
        """Get button style configuration."""
        styles = {
            "primary": {
                "bg": cls.PRIMARY,
                "fg": cls.TEXT_PRIMARY,
                "activebackground": cls.PRIMARY_HOVER,
                "activeforeground": cls.TEXT_PRIMARY,
            },
            "secondary": {
                "bg": cls.SURFACE_ELEVATED,
                "fg": cls.TEXT_PRIMARY,
                "activebackground": cls.BACKGROUND_LIGHT,
                "activeforeground": cls.TEXT_PRIMARY,
            },
            "success": {
                "bg": cls.SUCCESS,
                "fg": cls.TEXT_PRIMARY,
                "activebackground": "#059669",
                "activeforeground": cls.TEXT_PRIMARY,
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
