"""Canvas-drawn ANSI keyboard heatmap coloured by per-key press frequency."""

import tkinter as tk

from src.core.theme import ModernTheme

# ---------------------------------------------------------------------------
# ANSI keyboard layout
# ---------------------------------------------------------------------------
# Each entry: (key_name, display_label, width_units)
#   key_name  — matches pynput's _key_name() output (or None for a spacer)
#   width_units — 1.0 = standard key width

ANSI_LAYOUT: list[list[tuple[str | None, str, float]]] = [
    # Function row
    [
        ("esc", "Esc", 1.0),
        (None, "", 0.5),
        ("f1", "F1", 1.0), ("f2", "F2", 1.0), ("f3", "F3", 1.0), ("f4", "F4", 1.0),
        (None, "", 0.25),
        ("f5", "F5", 1.0), ("f6", "F6", 1.0), ("f7", "F7", 1.0), ("f8", "F8", 1.0),
        (None, "", 0.25),
        ("f9", "F9", 1.0), ("f10", "F10", 1.0), ("f11", "F11", 1.0), ("f12", "F12", 1.0),
    ],
    # Number row
    [
        ("`", "`", 1.0),
        ("1", "1", 1.0), ("2", "2", 1.0), ("3", "3", 1.0), ("4", "4", 1.0),
        ("5", "5", 1.0), ("6", "6", 1.0), ("7", "7", 1.0), ("8", "8", 1.0),
        ("9", "9", 1.0), ("0", "0", 1.0), ("-", "-", 1.0), ("=", "=", 1.0),
        ("backspace", "Bksp", 2.0),
    ],
    # QWERTY row
    [
        ("tab", "Tab", 1.5),
        ("q", "Q", 1.0), ("w", "W", 1.0), ("e", "E", 1.0), ("r", "R", 1.0),
        ("t", "T", 1.0), ("y", "Y", 1.0), ("u", "U", 1.0), ("i", "I", 1.0),
        ("o", "O", 1.0), ("p", "P", 1.0), ("[", "[", 1.0), ("]", "]", 1.0),
        ("\\", "\\", 1.5),
    ],
    # Home row
    [
        ("caps_lock", "Caps", 1.75),
        ("a", "A", 1.0), ("s", "S", 1.0), ("d", "D", 1.0), ("f", "F", 1.0),
        ("g", "G", 1.0), ("h", "H", 1.0), ("j", "J", 1.0), ("k", "K", 1.0),
        ("l", "L", 1.0), (";", ";", 1.0), ("'", "'", 1.0),
        ("enter", "Enter", 2.25),
    ],
    # Shift row
    [
        ("shift", "Shift", 2.25),
        ("z", "Z", 1.0), ("x", "X", 1.0), ("c", "C", 1.0), ("v", "V", 1.0),
        ("b", "B", 1.0), ("n", "N", 1.0), ("m", "M", 1.0),
        (",", ",", 1.0), (".", ".", 1.0), ("/", "/", 1.0),
        ("shift_r", "Shift", 2.75),
    ],
    # Bottom row
    [
        ("ctrl_l", "Ctrl", 1.25),
        ("cmd", "Cmd", 1.25),
        ("alt_l", "Alt", 1.25),
        ("space", "", 6.25),
        ("alt_r", "Alt", 1.25),
        ("cmd_r", "Cmd", 1.25),
        ("menu", "Fn", 1.0),
        ("ctrl_r", "Ctrl", 1.25),
    ],
]

# Mouse buttons — drawn to the right of the keyboard
MOUSE_BUTTONS: list[tuple[str, str]] = [
    ("left", "LMB"),
    ("middle", "MMB"),
    ("right", "RMB"),
    ("scroll_up", "Scrl\u2191"),
    ("scroll_down", "Scrl\u2193"),
]

# Key name aliases so both forms match
_KEY_ALIASES: dict[str, list[str]] = {
    "cmd":     ["cmd", "cmd_l"],
    "cmd_r":   ["cmd_r"],
    "shift":   ["shift", "shift_l"],
    "shift_r": ["shift_r"],
    "ctrl_l":  ["ctrl_l", "ctrl"],
    "ctrl_r":  ["ctrl_r"],
    "alt_l":   ["alt_l", "alt"],
    "alt_r":   ["alt_r"],
}


def _count_for_key(key_name: str, counts: dict[str, int]) -> int:
    """Sum counts across all aliases for *key_name*."""
    aliases = _KEY_ALIASES.get(key_name, [key_name])
    return sum(counts.get(a, 0) for a in aliases)


# ---------------------------------------------------------------------------
# Heatmap view
# ---------------------------------------------------------------------------

class KeyboardHeatmapView(tk.Frame):
    """Canvas-based ANSI keyboard heatmap."""

    _KEY_UNIT = 48       # px per 1.0-width key (before scaling)
    _KEY_HEIGHT = 42
    _KEY_GAP = 3
    _ROW_GAP = 6
    _MARGIN = 20

    def __init__(self, parent, page, **kwargs):
        super().__init__(parent, bg=ModernTheme.BACKGROUND_DARK, **kwargs)
        self.page = page
        self._counts: dict[str, int] = {}

        self.canvas = tk.Canvas(
            self,
            bg=ModernTheme.BACKGROUND_DARK,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda _: self._draw())

    def update_data(self, counts: dict[str, int]) -> None:
        self._counts = counts
        self._draw()

    def export(self, path: str) -> None:
        """Export the heatmap canvas as a PostScript / EPS file."""
        self.canvas.postscript(file=path, colormode="color")

    # ---- Drawing ------------------------------------------------------

    def _draw(self) -> None:
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        self.canvas.configure(bg=ModernTheme.BACKGROUND_DARK)

        # Calculate scale to fit
        max_row_units = max(
            sum(w for _, _, w in row) + (len(row) - 1) * self._KEY_GAP / self._KEY_UNIT
            for row in ANSI_LAYOUT
        )
        # Reserve space on the right for mouse buttons
        mouse_col_width_units = 3.5
        total_width_units = max_row_units + mouse_col_width_units
        available = cw - 2 * self._MARGIN
        scale = available / (total_width_units * self._KEY_UNIT)

        unit = self._KEY_UNIT * scale
        h = self._KEY_HEIGHT * scale
        gap = self._KEY_GAP * scale
        row_gap = self._ROW_GAP * scale

        max_count = max(
            (_count_for_key(kn, self._counts) for row in ANSI_LAYOUT for kn, _, _ in row if kn),
            default=0,
        )
        # Include mouse buttons in max
        for btn_name, _ in MOUSE_BUTTONS:
            max_count = max(max_count, self._counts.get(btn_name, 0))
        max_count = max_count or 1

        # Draw keyboard rows
        y = self._MARGIN
        for row in ANSI_LAYOUT:
            x = self._MARGIN
            for key_name, label, width_units in row:
                w = width_units * unit
                if key_name is None:
                    x += w + gap
                    continue

                count = _count_for_key(key_name, self._counts)
                color = self._intensity_color(count, max_count)
                self._rounded_rect(x, y, w, h, 4 * scale, color)

                # Key label
                self.canvas.create_text(
                    x + w / 2, y + h / 2 - (4 * scale if count > 0 else 0),
                    text=label,
                    fill=ModernTheme.TEXT_PRIMARY,
                    font=(ModernTheme.FONT_FAMILY, max(7, int(9 * scale))),
                )
                # Count
                if count > 0:
                    self.canvas.create_text(
                        x + w / 2, y + h - 7 * scale,
                        text=str(count),
                        fill=ModernTheme.TEXT_SECONDARY,
                        font=(ModernTheme.FONT_FAMILY, max(6, int(7 * scale))),
                    )

                x += w + gap
            y += h + row_gap

        # Draw mouse buttons column to the right
        kb_right = self._MARGIN + max_row_units * unit + gap * 4
        self._draw_mouse_buttons(kb_right, self._MARGIN, unit, h, gap, row_gap, scale, max_count)

    def _draw_mouse_buttons(
        self, x_start: float, y_start: float,
        unit: float, h: float, gap: float, row_gap: float,
        scale: float, max_count: int,
    ) -> None:
        w = 2.5 * unit
        y = y_start
        # Section label
        self.canvas.create_text(
            x_start + w / 2, y + h / 2,
            text="Mouse",
            fill=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, max(8, int(10 * scale)), "bold"),
        )
        y += h + row_gap

        for btn_name, label in MOUSE_BUTTONS:
            count = self._counts.get(btn_name, 0)
            color = self._intensity_color(count, max_count)
            self._rounded_rect(x_start, y, w, h, 4 * scale, color)
            self.canvas.create_text(
                x_start + w / 2, y + h / 2 - (4 * scale if count > 0 else 0),
                text=label,
                fill=ModernTheme.TEXT_PRIMARY,
                font=(ModernTheme.FONT_FAMILY, max(7, int(9 * scale))),
            )
            if count > 0:
                self.canvas.create_text(
                    x_start + w / 2, y + h - 7 * scale,
                    text=str(count),
                    fill=ModernTheme.TEXT_SECONDARY,
                    font=(ModernTheme.FONT_FAMILY, max(6, int(7 * scale))),
                )
            y += h + row_gap

    # ---- Helpers ------------------------------------------------------

    def _intensity_color(self, count: int, max_count: int) -> str:
        """Map count to a colour on a magenta spectrum."""
        if count == 0:
            return ModernTheme.SURFACE
        t = count / max_count
        # Interpolate: dim purple (#2a1a2e) -> bright magenta (#e91e8c)
        r = int(42 + t * (233 - 42))
        g = int(26 + t * (30 - 26))
        b = int(46 + t * (140 - 46))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _rounded_rect(
        self, x: float, y: float, w: float, h: float, r: float, fill: str,
    ) -> None:
        """Draw a rounded rectangle on the canvas."""
        # Clamp radius so it doesn't exceed half of the smaller dimension
        r = min(r, w / 2, h / 2)
        points = [
            x + r, y,
            x + w - r, y,
            x + w, y + r,
            x + w, y + h - r,
            x + w - r, y + h,
            x + r, y + h,
            x, y + h - r,
            x, y + r,
        ]
        self.canvas.create_polygon(
            points,
            fill=fill,
            outline=ModernTheme.BORDER,
            width=1,
            smooth=True,
        )
