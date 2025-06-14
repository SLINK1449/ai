import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing as mp
from datamodel import train_model  # Asegúrate de que train_model(pipe) tenga el pipe.send

class NeuronWindow:
    def __init__(self, master, process, pipe):
        self.master = master
        self.process = process
        self.pipe = pipe

        self.master.title("Transformer Training Monitor")
        self.master.geometry("1280x720")

        self.epochs = []
        self.losses = []

        # Configurar gráfico
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.set_facecolor('black')
        self.setup_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().configure(bg='black')
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barra de progreso y porcentaje
        self.progress_label = tk.Label(self.master, text="Training Progress", font=("Arial", 12))
        self.progress_label.pack(pady=(5, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.master, orient="horizontal", length=500, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=(0, 10))

        self.percent_label = tk.Label(self.master, text="0%", font=("Arial", 12))
        self.percent_label.pack()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_plot()

    def setup_plot(self):
        self.ax.set_xlabel('Epoch', fontsize=12)
        self.ax.set_ylabel('Loss', fontsize=12)
        self.ax.set_title('Training Progress', fontsize=14)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.line, = self.ax.plot([], [], color='cyan', lw=2, label='Loss')
        self.ax.legend(loc='upper right')
        self.info_text = self.ax.text(
            0.95, 0.90, "Starting...",
            transform=self.ax.transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='black', alpha=0.8), fontsize=10
        )

    def update_plot(self):
        try:
            if self.pipe.poll():
                data = self.pipe.recv()
                epoch = data['epoch']
                loss = data['loss']

                self.epochs.append(epoch)
                self.losses.append(loss)

                self.line.set_data(self.epochs, self.losses)
                self.ax.relim()
                self.ax.autoscale_view()
                self.info_text.set_text(f"Epoch: {epoch}\nLoss: {loss:.4f}")
                self.canvas.draw_idle()

                # Simulación de progreso de 0 a 100
                progress_percent = (epoch % 100)
                self.progress_var.set(progress_percent)
                self.percent_label.config(text=f"{progress_percent:.0f}%")

            self.master.after(100, self.update_plot)

        except (BrokenPipeError, EOFError):
            print("[INFO] Pipe closed")
            self.on_close()
        except Exception as e:
            print(f"[ERROR] {e}")
            self.on_close()

    def on_close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.pipe.close()
        self.master.destroy()

class AutoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()
        self.launch_neuron()

    def launch_neuron(self):
        parent_pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(target=train_model, args=(child_pipe,))
        self.process.start()

        window = tk.Toplevel()
        NeuronWindow(window, self.process, parent_pipe)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    root = tk.Tk()
    app = AutoLauncher(root)
    root.mainloop()
