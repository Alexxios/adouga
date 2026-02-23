# ==========================================
# 4. GUI PAGES
# ==========================================
import tkinter as tk


# External libraries
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MonitorPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Left: Image
        self.left_frame = tk.Frame(self, bg="#222", width=500)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)
        self.img_label = tk.Label(self.left_frame, text="Waiting...", bg="#222", fg="#888")
        self.img_label.pack(expand=True)

        # Right: Graphs
        self.right_frame = tk.Frame(self, bg="white", width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # We increase the figure size for 3 graphs
        self.fig = Figure(figsize=(4, 6), dpi=100)

        self.ax_cpu = self.fig.add_subplot(311)
        self.ax_cpu.set_title("CPU %")
        self.ax_cpu.set_ylim(0, 100)
        self.line_cpu, = self.ax_cpu.plot([], [], 'r-', lw=2)

        self.ax_ram = self.fig.add_subplot(312)
        self.ax_ram.set_title("RAM %")
        self.ax_ram.set_ylim(0, 100)
        self.line_ram, = self.ax_ram.plot([], [], 'b-', lw=2)

        self.ax_inp = self.fig.add_subplot(313)
        self.ax_inp.set_title("Activity (Inputs/10s)")
        self.ax_inp.set_ylim(0, 300) # Scale for inputs
        self.line_inp, = self.ax_inp.plot([], [], 'g-', lw=2)

        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
                self.img_label.config(image=tk_img)
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
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.last_prediction = None

        # Main container with two sections
        main_container = tk.Frame(self, bg="#1a1a1a")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Screenshot display
        self.left_frame = tk.Frame(main_container, bg="#222", width=600)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Screenshot title
        tk.Label(
            self.left_frame,
            text="Foreground Application Screenshot",
            bg="#222",
            fg="white",
            font=("Arial", 14, "bold")
        ).pack(pady=(10, 5))

        # Screenshot display
        self.img_label = tk.Label(
            self.left_frame,
            text="Waiting for screenshot...",
            bg="#222",
            fg="#888",
            font=("Arial", 12)
        )
        self.img_label.pack(expand=True, pady=10)

        # Right: Prediction results
        self.right_frame = tk.Frame(main_container, bg="#2a2a2a", width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.right_frame.pack_propagate(False)

        # Results title
        tk.Label(
            self.right_frame,
            text="AI Prediction Results",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 16, "bold")
        ).pack(pady=(20, 10))

        # Prediction class label
        self.class_label = tk.Label(
            self.right_frame,
            text="Class: -",
            bg="#2a2a2a",
            fg="#00ff00",
            font=("Arial", 20, "bold"),
            wraplength=350
        )
        self.class_label.pack(pady=10)

        # Confidence label
        self.confidence_label = tk.Label(
            self.right_frame,
            text="Confidence: -",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=5)

        # Separator
        tk.Frame(self.right_frame, bg="#444", height=2).pack(fill=tk.X, pady=20, padx=20)

        # Detailed probabilities title
        tk.Label(
            self.right_frame,
            text="Class Probabilities:",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 5))

        # Probabilities frame
        self.prob_frame = tk.Frame(self.right_frame, bg="#2a2a2a")
        self.prob_frame.pack(pady=5, padx=20, fill=tk.X)

        # Provider info
        self.provider_label = tk.Label(
            self.right_frame,
            text="Provider: -",
            bg="#2a2a2a",
            fg="#888",
            font=("Arial", 9)
        )
        self.provider_label.pack(side=tk.BOTTOM, pady=10)

        # Status label
        self.status_label = tk.Label(
            self.right_frame,
            text="Status: Initializing...",
            bg="#2a2a2a",
            fg="#ffaa00",
            font=("Arial", 10)
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
                        fg="#ff0000"
                    )
                    return

                # Get prediction with details
                prediction = self.controller.classifier.predict_with_details(img)
                self.last_prediction = prediction

                # Update prediction display
                pred_class = prediction['predicted_class']
                confidence = prediction['confidence']

                # Color code based on class
                class_color = "#00ff00" if pred_class == "Gaming" else "#ffaa00"

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
                    fg="#00ff00"
                )

            except Exception as e:
                self.status_label.config(
                    text=f"Status: Error - {str(e)}",
                    fg="#ff0000"
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
            label_frame = tk.Frame(self.prob_frame, bg="#2a2a2a")
            label_frame.pack(fill=tk.X, pady=3)

            tk.Label(
                label_frame,
                text=f"{class_name}:",
                bg="#2a2a2a",
                fg="white",
                font=("Arial", 10),
                width=15,
                anchor="w"
            ).pack(side=tk.LEFT)

            tk.Label(
                label_frame,
                text=f"{prob:.1%}",
                bg="#2a2a2a",
                fg="white",
                font=("Arial", 10, "bold"),
                width=8,
                anchor="e"
            ).pack(side=tk.RIGHT)

            # Progress bar
            bar_frame = tk.Frame(self.prob_frame, bg="#444", height=20)
            bar_frame.pack(fill=tk.X, pady=(0, 8))

            bar_width = int(prob * 350)  # Max width 350px
            bar_color = "#00ff00" if class_name == "Gaming" else "#ffaa00"

            tk.Frame(
                bar_frame,
                bg=bar_color,
                width=bar_width,
                height=20
            ).pack(side=tk.LEFT)
