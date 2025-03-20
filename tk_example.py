import tkinter as tk
import pyvisa
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
import numpy as np

class FuncGen():
    def __init__(self):
        # Create a resource manager
        rm = pyvisa.ResourceManager()

        instrument_address = 'GPIB0::28::INSTR'  # Replace with your instrument's address
        try:
            self.hp33120a = rm.open_resource(instrument_address)
            print("Connected to:", self.hp33120a.query("*IDN?"))  # Query the instrument identification
        except pyvisa.errors.VisaIOError:
            print("Could not connect to the instrument. Check the address and connection.")
            exit()

    def generate_sinusoidal(self, amplitude, frequency, time):
        """
        Generate a sinusoidal signal.
        :param amplitude: Amplitude of the signal
        :param frequency: Frequency of the signal (Hz)
        :param time: Time array
        :return: Sinusoidal signal
        """
        print(f"Amp: {amplitude} | Type: {type(amplitude)}\nFreq: {frequency} | Type: {type(frequency)}")
        return amplitude * np.sin(2 * np.pi * frequency * time)

    def generate_square(self, amplitude, frequency, time):
        """
        Generate a square wave signal.
        :param amplitude: Amplitude of the signal
        :param frequency: Frequency of the signal (Hz)
        :param time: Time array
        :return: Square wave signal
        """
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * time))

    def generate_ramp(self, amplitude, frequency, time):
        """
        Generate a sawtooth wave signal.
        :param amplitude: Amplitude of the signal
        :param frequency: Frequency of the signal (Hz)
        :param time: Time array
        :return: Sawtooth wave signal
        """
        return amplitude * signal.sawtooth(2 * np.pi * frequency * time, width=1)

    def generate_triangular(self, amplitude, frequency, time, width=0.5):
        """
        Generate a triangular wave using scipy.signal.sawtooth.
        :param amplitude: Amplitude of the signal
        :param frequency: Frequency of the signal (Hz)
        :param time: Time array
        :param width: Width of the rising ramp (0.5 for symmetric triangular wave)
        :return: Triangular wave signal
        """
        return amplitude * signal.sawtooth(2 * np.pi * frequency * time, width=width)

    def on_confirm(self):
        # Get values from dropdowns and text inputs
        shape_value = shape_dropdown.get()
        freq_value = float(freq_input.get())
        amp_value = float(amp_input.get())
        
        # Display the results
        display_text = (
            f"Shape: {shape_value}\n"
            f"Frequency: {freq_value} Hz\n"
            f"Amplitude: {amp_value} V"
        )
        display_label.config(text=display_text)

        params = {"shape" : shape_value,
                  "freq": freq_value,
                  "ampl": amp_value
                  }
        self.update_plot(params)
        self.output(params)
        # update_plot()

    def update_plot(self, params:dict):
        # Clear the current plot
        ax.clear()
        
        # Generate some data based on the text inputs
        period = 1/params["freq"]
        time = np.linspace(0, 3*period, 1000)
        try:
            if params["shape"] == "SIN":
                signal = self.generate_sinusoidal(params["ampl"], params["freq"], time)

            elif params["shape"] == "SQU":
                signal = self.generate_square(params["ampl"], params["freq"], time)

            elif params["shape"] == "TRI":
                signal = self.generate_triangular(params["ampl"], params["freq"], time)

            elif params["shape"] == "RAMP":
                signal = self.generate_ramp(params["ampl"], params["freq"], time)

            ax.plot(time, signal, label=f"{params["shape"]} Wave")
            ax.set_title("Output signal preview")
        except ValueError:
            ax.set_title("Invalid Input: Enter Numbers")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [v]")
        ax.grid(True)
        
        # Redraw the canvas
        canvas.draw()

    def output(self, params:dict):
        shape = params["shape"]
        freq = params["freq"]
        ampl = params["ampl"]

        instr = f"FUNC {shape}; FREQ {freq}; VOLT {ampl}"

        self.hp33120a.write(instr)
    
if __name__ == "__main__":
    print("Starting software...")
    instrument = FuncGen()
# Create the main window
    root = tk.Tk()
    root.title("Dropdowns and Text Inputs on Same Line")

# Create a toolbar frame at the top
    toolbar = ttk.Frame(root)
    toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

# Create variables for dropdowns
    shape_dropdown = tk.StringVar()

# Dropdown options
    options = ["SIN", "SQU", "TRI", "RAMP", "PULS"]

# Create dropdowns and text inputs in the toolbar
# Dropdown 1
    ttk.Label(toolbar, text="Shape:").grid(row=0, column=0, padx=5, pady=5)
    dropdown1 = ttk.Combobox(toolbar, textvariable=shape_dropdown, values=options, width=15)
    dropdown1.grid(row=0, column=1, padx=5, pady=5)
    dropdown1.set(options[0])  # Set default value

# Text Input 1
    ttk.Label(toolbar, text="Frequency:").grid(row=0, column=2, padx=5, pady=5)
    freq_input = ttk.Entry(toolbar, width=20)
    freq_input.grid(row=0, column=3, padx=5, pady=5)

# Text Input 2
    ttk.Label(toolbar, text="Amplitude:").grid(row=0, column=6, padx=5, pady=5)
    amp_input = ttk.Entry(toolbar, width=20)
    amp_input.grid(row=0, column=7, padx=5, pady=5)

# Confirm button
    confirm_button = ttk.Button(toolbar, text="Confirm", command=instrument.on_confirm)
    confirm_button.grid(row=0, column=8, padx=5, pady=5)

# Create a display area below the toolbar
    display_frame = ttk.Frame(root)
    display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Add a label to the display area
    display_label = ttk.Label(display_frame, text="Selected values and inputs will appear here", justify=tk.LEFT)
    display_label.pack(padx=10, pady=10)

# Create a display frame below the toolbar
    display_frame = ttk.Frame(root)
    display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Add a label to the display area
    display_label = ttk.Label(display_frame, text="Selected values and inputs will appear here", justify=tk.LEFT)
    display_label.pack(padx=10, pady=10)

# Create a matplotlib figure and embed it in the display frame
    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=display_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Start the main event loop
    root.mainloop()
