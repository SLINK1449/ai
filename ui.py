import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing as mp
from training import setup_training

class NeuronWindow:
    def __init__(self, master, process, pipe):
        self.master = master
        self.process = process
        self.pipe = pipe
        
        # window configuration
        self.master.title("Real time training")
        self.master.geometry("1920x1080")
        
        # data arrays
        self.epochs = []
        self.losses = []
        self.weights = []
        self.biases = []
        
        # grafich configuration
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.set_facecolor('black')
        self.setup_plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().configure(bg='black')  
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # close window event
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # initiate plot update
        self.update_plot()

    def setup_plot(self):
        """Configurate initial plot"""
        self.ax.set_xlabel('Epoch', fontsize=10)
        self.ax.set_ylabel('Loss', fontsize=10)
        self.ax.set_title('Training progress', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.line, = self.ax.plot([], [], color='#FF0000', lw=2, label='current loss')
        self.ax.legend(loc='upper right')
        
        # informative text
        self.info_text = self.ax.text(
            0.95, 0.90,
            "Starting...",
            transform=self.ax.transAxes,
            ha='right',
            va='top',
            bbox=dict(facecolor='black', alpha=0.8),
            fontsize=9
        )

    def update_plot(self):
        """Actualizar gráfica con nuevos datos"""
        try:
            if self.pipe.poll():
                data = self.pipe.recv()
                
                # update data arrays
                self.epochs.append(data['epoch'])
                self.losses.append(data['loss'])
                self.weights.append(data['weight'])
                self.biases.append(data['bias'])
                
                # update graphic
                self.line.set_data(self.epochs, self.losses)
                self.ax.relim()
                self.ax.autoscale_view()
                
                # update text
                self.info_text.set_text(
                    f"Epoch: {data['epoch']}\n"
                    f"Loss: {data['loss']:.4f}\n"
                    f"Best loss: {data['best_loss']:.4f}\n"
                    f"Weight: {data['weight']:.4f}\n"
                    f"Bias: {data['bias']:.4f}"
                )
                
                self.canvas.draw_idle()
            
            # Programing the next update
            self.master.after(50, self.update_plot)
            
        except (BrokenPipeError, EOFError):
            print("closed connection")
            self.on_close()
        except Exception as e:
            print(f"Error: {str(e)}")
            self.on_close()

    def on_close(self):
        """Handle window close event"""
        if self.process.is_alive():
            self.process.terminate()
        self.pipe.close()
        self.master.destroy()

class AutoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()  # hide the root window
        
        # auto-launch the neuron training
        self.launch_neuron()

    def launch_neuron(self):
        """Start training process and create the window"""
        parent_pipe, child_pipe = mp.Pipe()
        self.process = setup_training(child_pipe)
        self.process.start()
        
        # create wiev window
        window = tk.Toplevel()
        NeuronWindow(window, self.process, parent_pipe)

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoLauncher(root)
    root.mainloop()