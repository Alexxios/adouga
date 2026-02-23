# ==========================================
# 4. GUI PAGES
# ==========================================
import tkinter as tk


# External libraries
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .theme import ModernTheme
from .network_indicator import NetworkIndicator


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

        # 2. Update Graphs
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


class AIPage(tk.Frame):
    """AI prediction page showing screenshot and classification results."""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=ModernTheme.BACKGROUND_DARK)
        self.controller = controller
        self.last_prediction = None

        # Main container with two sections (no header)
        main_container = tk.Frame(self, bg=ModernTheme.BACKGROUND_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Screenshot display
        self.left_frame = tk.Frame(main_container, bg=ModernTheme.BACKGROUND_MEDIUM, width=600)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Screenshot title
        tk.Label(
            self.left_frame,
            text="Foreground Application Screenshot",
            bg=ModernTheme.BACKGROUND_MEDIUM,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_MEDIUM, "bold")
        ).pack(pady=(10, 5))

        # Screenshot display
        self.img_label = tk.Label(
            self.left_frame,
            text="Waiting for screenshot...",
            bg=ModernTheme.BACKGROUND_MEDIUM,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL)
        )
        self.img_label.pack(expand=True, pady=10)

        # Right: Prediction results
        self.right_frame = tk.Frame(main_container, bg=ModernTheme.SURFACE, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.right_frame.pack_propagate(False)

        # Results title
        tk.Label(
            self.right_frame,
            text="Prediction Results",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_XLARGE, "bold")
        ).pack(pady=(20, 10))

        # Prediction class label
        self.class_label = tk.Label(
            self.right_frame,
            text="Class: -",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.SUCCESS,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_TITLE, "bold"),
            wraplength=350
        )
        self.class_label.pack(pady=10)

        # Confidence label
        self.confidence_label = tk.Label(
            self.right_frame,
            text="Confidence: -",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_MEDIUM)
        )
        self.confidence_label.pack(pady=5)

        # Separator
        tk.Frame(self.right_frame, bg=ModernTheme.BORDER, height=1).pack(fill=tk.X, pady=20, padx=20)

        # Detailed probabilities title
        tk.Label(
            self.right_frame,
            text="Class Probabilities:",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL, "bold")
        ).pack(pady=(10, 5))

        # Probabilities frame
        self.prob_frame = tk.Frame(self.right_frame, bg=ModernTheme.SURFACE)
        self.prob_frame.pack(pady=5, padx=20, fill=tk.X)

        # Provider info
        self.provider_label = tk.Label(
            self.right_frame,
            text="Provider: -",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL)
        )
        self.provider_label.pack(side=tk.BOTTOM, pady=10)

        # Status label
        self.status_label = tk.Label(
            self.right_frame,
            text="Status: Initializing...",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.WARNING,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL)
        )
        self.status_label.pack(side=tk.BOTTOM, pady=5)

    def update_view(self):
        """Update the AI page with latest screenshot and prediction."""
        img = self.controller.current_image

        if img:
            # Update screenshot display
            display_w = self.left_frame.winfo_width()
            display_h = self.left_frame.winfo_height()

            if display_w > 10:  # Avoid startup glitch
                # Calculate aspect ratio preserving resize
                img_ratio = img.width / img.height
                frame_ratio = display_w / display_h

                if frame_ratio > img_ratio:
                    new_h = int(display_h * 0.8)
                    new_w = int(new_h * img_ratio)
                else:
                    new_w = int(display_w * 0.8)
                    new_h = int(new_w / img_ratio)

                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img_resized)
                self.img_label.config(image=tk_img, text="")
                self.img_label.image = tk_img

            # Run prediction
            try:
                if not hasattr(self.controller, 'classifier'):
                    self.status_label.config(
                        text="Status: Model not loaded",
                        fg=ModernTheme.ERROR
                    )
                    return

                # Get prediction with details
                prediction = self.controller.classifier.predict_with_details(img)
                self.last_prediction = prediction

                # Update prediction display
                pred_class = prediction['predicted_class']
                confidence = prediction['confidence']

                # Color code based on class
                class_color = ModernTheme.SUCCESS if pred_class == "Gaming" else ModernTheme.WARNING

                self.class_label.config(
                    text=f"Class: {pred_class}",
                    fg=class_color
                )

                self.confidence_label.config(
                    text=f"Confidence: {confidence:.1%}"
                )

                # Update probabilities
                self._update_probabilities(prediction['probabilities'])

                # Update provider info
                self.provider_label.config(
                    text=f"Provider: {prediction['provider']}"
                )

                self.status_label.config(
                    text="Status: Ready",
                    fg=ModernTheme.SUCCESS
                )

            except Exception as e:
                self.status_label.config(
                    text=f"Status: Error - {str(e)}",
                    fg=ModernTheme.ERROR
                )
                print(f"Prediction error: {e}")
        else:
            self.img_label.config(text="No screenshot available")

    def _update_probabilities(self, probabilities: dict):
        """Update the probability bars display.

        Args:
            probabilities: Dictionary of class -> probability
        """
        # Clear existing widgets
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

        # Create probability bars for each class
        for class_name, prob in probabilities.items():
            # Class name and percentage
            label_frame = tk.Frame(self.prob_frame, bg=ModernTheme.SURFACE)
            label_frame.pack(fill=tk.X, pady=3)

            tk.Label(
                label_frame,
                text=f"{class_name}:",
                bg=ModernTheme.SURFACE,
                fg=ModernTheme.TEXT_PRIMARY,
                font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL),
                width=15,
                anchor="w"
            ).pack(side=tk.LEFT)

            tk.Label(
                label_frame,
                text=f"{prob:.1%}",
                bg=ModernTheme.SURFACE,
                fg=ModernTheme.TEXT_PRIMARY,
                font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL, "bold"),
                width=8,
                anchor="e"
            ).pack(side=tk.RIGHT)

            # Progress bar background
            bar_bg = tk.Frame(self.prob_frame, bg=ModernTheme.BACKGROUND_LIGHT, height=20)
            bar_bg.pack(fill=tk.X, pady=(0, 8))

            # Progress bar fill
            bar_width = int(prob * 350)  # Max width 350px
            bar_color = ModernTheme.SUCCESS if class_name == "Gaming" else ModernTheme.PRIMARY

            if bar_width > 0:
                tk.Frame(
                    bar_bg,
                    bg=bar_color,
                    width=bar_width,
                    height=20
                ).pack(side=tk.LEFT)
