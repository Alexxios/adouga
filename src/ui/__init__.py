# ==========================================
# 4. GUI PAGES
# ==========================================
import tkinter as tk


# External libraries
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ai import GameSeeAI

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
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#f0f0f5")
        self.controller = controller
        self.ai = GameSeeAI()

        tk.Label(self, text="AI Analysis", font=("Helvetica", 20, "bold"), bg="#f0f0f5").pack(pady=15)

        self.img_frame = tk.Frame(self, bg="black", width=300, height=200)
        self.img_frame.pack(pady=5)
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, text="No Image", bg="black", fg="white")
        self.img_label.pack(expand=True, fill=tk.BOTH)

        self.result_label = tk.Label(self, text="...", font=("Helvetica", 36, "bold"), bg="#f0f0f5", fg="#555")
        self.result_label.pack(pady=10)

        self.stats_label = tk.Label(self, text="", font=("Helvetica", 12), bg="#f0f0f5")
        self.stats_label.pack()

    def update_view(self):
        img = self.controller.current_image
        cpu = self.controller.cpu_data[-1] if self.controller.cpu_data else 0
        inp = self.controller.input_data[-1] if self.controller.input_data else 0

        if img:
            img_small = img.resize((300, 200), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img_small)
            self.img_label.config(image=tk_img)
            self.img_label.image = tk_img

            label, prob = self.ai.predict(img, cpu, inp)

            color = "#555"
            if "GAME" in label: color = "#e74c3c" # Red
            elif "WORK" in label: color = "#f39c12" # Yellow/Orange
            elif "APP" in label: color = "#27ae60" # Green

            self.result_label.config(text=label, fg=color)
            self.stats_label.config(text=f"Game Probability: {prob:.0f}%\nInputs (10s): {inp} | CPU: {cpu}%")
