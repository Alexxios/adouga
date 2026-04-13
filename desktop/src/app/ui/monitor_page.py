"""Live stats page — CPU, RAM, and input activity graphs."""

import tkinter as tk

from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.core.theme import ModernTheme


class MonitorPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=ModernTheme.BACKGROUND_DARK)
        self.controller = controller

        # Main content area (no header)
        content = tk.Frame(self, bg=ModernTheme.BACKGROUND_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Image
        self.left_frame = tk.Frame(content, bg=ModernTheme.BACKGROUND_MEDIUM, width=500)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.left_frame.pack_propagate(False)

        # Image title
        tk.Label(
            self.left_frame,
            text="Active Window Screenshot",
            bg=ModernTheme.BACKGROUND_MEDIUM,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_MEDIUM, "bold")
        ).pack(pady=(10, 5))

        self.img_label = tk.Label(
            self.left_frame,
            text="Waiting for screenshot...",
            bg=ModernTheme.BACKGROUND_MEDIUM,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL)
        )
        self.img_label.pack(expand=True)

        # Right: Graphs
        self.right_frame = tk.Frame(content, bg=ModernTheme.SURFACE, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Graph title
        tk.Label(
            self.right_frame,
            text="System Metrics",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_MEDIUM, "bold")
        ).pack(pady=(10, 5))

        # Configure matplotlib for crisp text rendering
        self.fig = Figure(figsize=(4, 6), dpi=120, facecolor=ModernTheme.SURFACE)

        self.ax_cpu = self.fig.add_subplot(311)
        self.ax_cpu.set_title("CPU Usage (%)", color=ModernTheme.TEXT_PRIMARY, fontsize=11, fontweight='bold')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_facecolor(ModernTheme.BACKGROUND_MEDIUM)
        self.ax_cpu.tick_params(colors=ModernTheme.TEXT_SECONDARY, labelsize=9)
        self.ax_cpu.grid(True, alpha=0.2, color=ModernTheme.BORDER)
        self.line_cpu, = self.ax_cpu.plot([], [], color=ModernTheme.ERROR, lw=2, antialiased=True)

        self.ax_ram = self.fig.add_subplot(312)
        self.ax_ram.set_title("RAM Usage (%)", color=ModernTheme.TEXT_PRIMARY, fontsize=11, fontweight='bold')
        self.ax_ram.set_ylim(0, 100)
        self.ax_ram.set_facecolor(ModernTheme.BACKGROUND_MEDIUM)
        self.ax_ram.tick_params(colors=ModernTheme.TEXT_SECONDARY, labelsize=9)
        self.ax_ram.grid(True, alpha=0.2, color=ModernTheme.BORDER)
        self.line_ram, = self.ax_ram.plot([], [], color=ModernTheme.PRIMARY, lw=2, antialiased=True)

        self.ax_inp = self.fig.add_subplot(313)
        self.ax_inp.set_title("User Activity (Inputs/5s)", color=ModernTheme.TEXT_PRIMARY, fontsize=11, fontweight='bold')
        self.ax_inp.set_ylim(0, 300)
        self.ax_inp.set_facecolor(ModernTheme.BACKGROUND_MEDIUM)
        self.ax_inp.tick_params(colors=ModernTheme.TEXT_SECONDARY, labelsize=9)
        self.ax_inp.grid(True, alpha=0.2, color=ModernTheme.BORDER)
        self.line_inp, = self.ax_inp.plot([], [], color=ModernTheme.SUCCESS, lw=2, antialiased=True)

        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_view(self):
        # 1. Update Image
        img = self.controller.current_image
        if img:
            display_w = self.left_frame.winfo_width()
            display_h = self.left_frame.winfo_height()
            if display_w > 10: # avoid startup glitch
                img_ratio = img.width / img.height
                frame_ratio = display_w / display_h
                if frame_ratio > img_ratio:
                    new_h = int(display_h * 0.9)
                    new_w = int(new_h * img_ratio)
                else:
                    new_w = int(display_w * 0.9)
                    new_h = int(new_w / img_ratio)
                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img_resized)
                self.img_label.config(image=tk_img, text="")
                self.img_label.image = tk_img

        # 2. Update Graphs — reapply theme colours each tick
        ModernTheme.style_matplotlib_figure(self.fig)
        for ax, title, color in [
            (self.ax_cpu, "CPU Usage (%)", ModernTheme.ERROR),
            (self.ax_ram, "RAM Usage (%)", ModernTheme.PRIMARY),
            (self.ax_inp, "User Activity (Inputs/5s)", ModernTheme.SUCCESS),
        ]:
            ModernTheme.style_matplotlib_axes(ax)
            ax.set_title(title, color=ModernTheme.TEXT_PRIMARY, fontsize=11, fontweight='bold')

        c_d = self.controller.cpu_data
        r_d = self.controller.ram_data
        i_d = self.controller.input_data

        x_axis = range(len(c_d))
        self.line_cpu.set_data(x_axis, c_d)
        self.line_ram.set_data(x_axis, r_d)
        self.line_inp.set_data(x_axis, i_d)

        self.ax_cpu.set_xlim(0, max(10, len(c_d)))
        self.ax_ram.set_xlim(0, max(10, len(r_d)))
        self.ax_inp.set_xlim(0, max(10, len(i_d)))

        # Dynamic Y-limit for input graph if inputs are huge
        max_inp = max(i_d) if i_d else 10
        self.ax_inp.set_ylim(0, max(300, max_inp + 50))

        self.canvas.draw()
