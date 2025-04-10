import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import nidaqmx.errors
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time


class Oscilloscope:
    def __init__(self, root):
        self.root = root
        self.root.title("NI DAQ Oscilloscope")
        self.lock = threading.Lock()

        # DAQ Configuration
        self.device_name = "Dev2/ai4"
        self.sample_rate = 10000.0  # Hz
        self.num_samples = 1000
        self.running = True

        # Initialize scales
        self.time_per_div = 0.1  # seconds/div
        self.volts_per_div = 1.0  # volts/div

        # Pre-calculate time axis
        self.time = np.linspace(0, self.num_samples / self.sample_rate, self.num_samples)
        self.data_buffer = np.zeros(self.num_samples)

        # Create DAQ Task
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            self.device_name,
            min_val=-10,
            max_val=10,
            terminal_config=TerminalConfiguration.RSE
        )
        self.task.timing.cfg_samp_clk_timing(
            self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.num_samples
        )
        self.task.in_stream.input_buf_size = self.num_samples * 10

        # GUI and plot setup
        self.create_controls()
        self.create_plot()


        # Queue for thread-safe data passing
        self.data_queue = queue.Queue(maxsize=1)

        # Start background data acquisition thread
        self.reader_thread = threading.Thread(target=self.daq_reader, daemon=True)
        self.reader_thread.start()

        # Start update loop for GUI
        self.update_plot()

    def create_controls(self):
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Time/Div
        ttk.Label(self.toolbar, text="Time/Div:").grid(row=0, column=0, padx=5)
        self.time_scale_label = ttk.Label(self.toolbar, text=f"{self.time_per_div:.3f} s/div")
        self.time_scale_label.grid(row=0, column=1, padx=5)
        ttk.Button(self.toolbar, text="+", command=self.increase_time_scale, width=3).grid(row=0, column=2)
        ttk.Button(self.toolbar, text="-", command=self.decrease_time_scale, width=3).grid(row=0, column=3)

        # Volts/Div
        ttk.Label(self.toolbar, text="Volts/Div:").grid(row=0, column=4, padx=5)
        self.volt_scale_label = ttk.Label(self.toolbar, text=f"{self.volts_per_div:.3f} V/div")
        self.volt_scale_label.grid(row=0, column=5, padx=5)
        ttk.Button(self.toolbar, text="+", command=self.increase_volt_scale, width=3).grid(row=0, column=6)
        ttk.Button(self.toolbar, text="-", command=self.decrease_volt_scale, width=3).grid(row=0, column=7)

        # Autoscale and Start/Stop
        ttk.Button(self.toolbar, text="Autoscale", command=self.autoscale).grid(row=0, column=8, padx=5)
        self.start_stop_button = ttk.Button(self.toolbar, text="Stop", command=self.toggle_start_stop)
        self.start_stop_button.grid(row=0, column=9, padx=5)

        # Frequency/Period Display
        self.freq_label = ttk.Label(self.toolbar, text="Freq: -- Hz   Period: -- s")
        self.freq_label.grid(row=0, column=10, padx=10)


    def create_plot(self):
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.line, = self.ax.plot(self.time, self.data_buffer, label="Live Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (V)")
        self.ax.set_title("NI 6251 - Digital Oscilloscope")
        self.ax.grid(True)
        self.ax.legend()
        self.update_plot_limits()

    def update_plot_limits(self):
        x_span = self.time_per_div * 10
        y_span = self.volts_per_div * 8

        new_num_samples = int(self.sample_rate * x_span)
        with self.lock:
            self.num_samples = new_num_samples
            self.time = np.linspace(0, x_span, self.num_samples)
            self.data_buffer = np.zeros(self.num_samples)

        self.reconfigure_daq()

        self.ax.set_xlim(0, x_span)
        self.ax.set_ylim(-y_span / 2, y_span / 2)
        self.line.set_xdata(self.time)
        self.line.set_ydata(self.data_buffer)

        self.time_scale_label.config(text=f"{self.time_per_div:.3f} s/div")
        self.volt_scale_label.config(text=f"{self.volts_per_div:.3f} V/div")
        self.canvas.draw()


    def reconfigure_daq(self):
        try:
            # Stop current task to reconfigure it
            self.task.stop()
            
            # Reconfigure timing with the updated sample count
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.num_samples
            )

            # Optionally resize input buffer (helps with overflow)
            self.task.in_stream.input_buf_size = self.num_samples * 10

            # Restart the task
            self.task.start()
        except nidaqmx.errors.DaqError as e:
            print("⚠️ DAQ reconfiguration failed:", e)



    def update_plot(self):
        if self.running and self.root.winfo_exists():
            try:
                if not self.data_queue.empty():
                    new_data = self.data_queue.get_nowait()
                    new_data = np.asarray(new_data)

                    if new_data.ndim != 1:
                        print(f"Invalid data shape: {new_data.shape}, skipping frame")
                        self.root.after(50, self.update_plot)
                        return

                    # Align on trigger
                    trigger_idx = self.find_trigger_index(new_data)
                    if trigger_idx is not None:
                        start = max(0, trigger_idx - self.num_samples // 2)
                        end = start + self.num_samples
                        if end <= len(new_data):
                            new_data = new_data[start:end]

                    if new_data.shape[0] == self.num_samples:
                        self.data_buffer[:] = new_data
                        self.line.set_ydata(self.data_buffer)
                        self.line.set_xdata(self.time)
                        self.canvas.draw_idle()
                    else:
                        print(f"Invalid data shape: {new_data.shape}, expected: ({self.num_samples},)")

            except Exception as e:
                print(f"update_plot exception: {e}")

        self.root.after(50, self.update_plot)




    def daq_reader(self):
        while True:
            if self.running:
                try:
                    # Safely capture the current num_samples under lock
                    with self.lock:
                        num_samples = self.num_samples

                    # Read from the DAQ
                    data = self.task.read(
                        number_of_samples_per_channel=num_samples,
                        timeout=1.0
                    )
                    data = np.asarray(data)

                    # Check shape against the current expected length
                    with self.lock:
                        if data.shape[0] == self.num_samples:
                            if self.data_queue.full():
                                self.data_queue.get_nowait()
                            self.data_queue.put(data)
                        else:
                            print(f"⚠️ DAQ returned wrong shape: {data.shape}, expected: ({self.num_samples},)")
                except nidaqmx.errors.DaqError as e:
                    print("DAQ Read Error:", e)
            else:
                time.sleep(0.05)  # Sleep briefly if not running

    def increase_time_scale(self):
        self.time_per_div *= 1.25
        self.update_plot_limits()

    def decrease_time_scale(self):
        self.time_per_div = max(0.001, self.time_per_div * 0.8)
        self.update_plot_limits()

    def increase_volt_scale(self):
        self.volts_per_div *= 1.25
        self.update_plot_limits()

    def decrease_volt_scale(self):
        self.volts_per_div = max(0.001, self.volts_per_div * 0.8)
        self.update_plot_limits()

    def round_to_125(self, value):
        decade = 10 ** np.floor(np.log10(value))
        normalized = value / decade
        if normalized <= 1.5:
            return 1 * decade
        elif normalized <= 3:
            return 2 * decade
        else:
            return 5 * decade

    def autoscale(self):
        try:
            self.running = False
            self.task.close()
            time.sleep(0.1)

            with nidaqmx.Task() as temp_task:
                temp_task.ai_channels.add_ai_voltage_chan(
                    self.device_name,
                    min_val=-10,
                    max_val=10,
                    terminal_config=TerminalConfiguration.RSE
                )
                temp_task.timing.cfg_samp_clk_timing(
                    self.sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=self.num_samples
                )
                temp_task.start()
                data = temp_task.read(number_of_samples_per_channel=self.num_samples, timeout=1.0)
                data = np.asarray(data)

            # Volts/Div autoscale
            peak_to_peak = np.max(data) - np.min(data)
            if peak_to_peak > 0:
                self.volts_per_div = self.round_to_125(peak_to_peak / 6)

            # Time/Div autoscale via zero-crossing frequency estimate
            mean_val = np.mean(data)
            zero_crossings = np.where(np.diff(np.signbit(data - mean_val)))[0]

            if len(zero_crossings) >= 2:
                periods = np.diff(zero_crossings[::2]) / self.sample_rate
                if len(periods) > 0:
                    avg_period = np.mean(periods)
                    self.time_per_div = self.round_to_125((avg_period * 3) / 10)
                    self.freq_label.config(text=f"Freq: {1/avg_period:.2f} Hz   Period: {avg_period:.4f} s")
                else:
                    self.freq_label.config(text="Freq: -- Hz   Period: -- s")
            else:
                self.freq_label.config(text="Freq: -- Hz   Period: -- s")

            # Update sample size and plot data
            self.num_samples = int(self.sample_rate * self.time_per_div * 10)
            self.time = np.linspace(0, self.time_per_div * 10, self.num_samples)
            self.data_buffer = np.zeros(self.num_samples)

            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(
                self.device_name,
                min_val=-10,
                max_val=10,
                terminal_config=TerminalConfiguration.RSE
            )
            self.task.timing.cfg_samp_clk_timing(
                self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.num_samples
            )
            self.task.in_stream.input_buf_size = self.num_samples * 10
            self.task.start()

            self.update_plot_limits()
            self.running = True

        except nidaqmx.errors.DaqError as e:
            print("Autoscale failed:", e)


    def find_trigger_index(self, data):
        """
        Finds the index of the first rising zero-crossing near the center of the buffer.
        """
        midpoint = len(data) // 2
        threshold = np.mean(data)

        for i in range(midpoint - 100, midpoint + 100):
            if 0 < i < len(data) and data[i - 1] < threshold and data[i] >= threshold:
                return i
        return None



    def toggle_start_stop(self):
        self.running = not self.running
        self.start_stop_button.config(text="Stop" if self.running else "Start")

    def close(self):
        self.running = False
        time.sleep(0.2)  # Let the thread finish
        self.task.close()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = Oscilloscope(root)

    def on_closing():
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
