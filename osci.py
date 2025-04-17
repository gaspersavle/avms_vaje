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
        self.root.title("NI DAQ Dual-Channel Oscilloscope")
        self.lock = threading.Lock()

        # DAQ Configuration - now with two channels
        self.device_names = ["Dev2/ai4", "Dev2/ai1"]  # Add your second channel here
        self.sample_rate = 500000.0  # Hz
        self.num_samples = 100000
        self.running = True
        self.task_running = False  # New flag to track task state
        self.after_id = None

        # Initialize scales
        self.time_per_div = 0.1  # seconds/div
        self.volts_per_div = [1.0, 1.0]  # volts/div for each channel

        self.channel_offsets = [1.0, -1.0]  # Vertical offsets for each channel
        self.separation_factor = 1.5  # How much to separate the channels

        # Ensure time and data buffers are always the same length
        self.time = np.linspace(0, self.num_samples / self.sample_rate, self.num_samples)
        self.data_buffers = [np.zeros(self.num_samples), np.zeros(self.num_samples)]

        # Pre-calculate time axis
        self.time = np.linspace(0, self.num_samples / self.sample_rate, self.num_samples)
        self.data_buffers = [np.zeros(self.num_samples), np.zeros(self.num_samples)]  # One buffer per channel

        # Create DAQ Task
        self.task = nidaqmx.Task()
        for device_name in self.device_names:
            self.task.ai_channels.add_ai_voltage_chan(
                device_name,
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

        # Main layout: left = plot, right = control panel
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Now call creators
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
        # Time/Div slider
        ttk.Label(self.controls_frame, text="Time/Div (s)").pack()
        self.time_scale_slider = tk.Scale(
            self.controls_frame, 
            from_=0.0001, to=1.0, 
            resolution=0.0001,
            orient=tk.HORIZONTAL,
            command=lambda val: self.set_time_scale(float(val))
        )
        self.time_scale_slider.set(self.time_per_div)
        self.time_scale_slider.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Ch1 Volts/Div slider
        ttk.Label(self.controls_frame, text="Ch1 V/Div").pack()
        self.volt_scale_slider1 = tk.Scale(
            self.controls_frame, 
            from_=0.001, to=10.0, 
            resolution=0.001,
            orient=tk.HORIZONTAL,
            command=lambda val: self.set_volt_scale(0, float(val))
        )
        self.volt_scale_slider1.set(self.volts_per_div[0])
        self.volt_scale_slider1.pack(fill=tk.X, padx=5, pady=5)

        # Ch2 Volts/Div slider
        ttk.Label(self.controls_frame, text="Ch2 V/Div").pack(pady=(10, 0))
        self.volt_scale_slider2 = tk.Scale(
            self.controls_frame, 
            from_=0.001, to=10.0, 
            resolution=0.001,
            orient=tk.HORIZONTAL,
            command=lambda val: self.set_volt_scale(1, float(val))
        )
        self.volt_scale_slider2.set(self.volts_per_div[1])
        self.volt_scale_slider2.pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Create horizontal frame to hold both vertical sliders
        offset_frame = ttk.Frame(self.controls_frame)
        offset_frame.pack(pady=10)

        # Ch1 vertical slider
        slider1_frame = ttk.Frame(offset_frame)
        slider1_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(slider1_frame, text="Ch1 Offset").pack()
        self.offset_slider1 = tk.Scale(
            slider1_frame, from_=8, to=-8, resolution=0.01,
            orient=tk.VERTICAL, command=lambda val: self.set_offset(0, float(val))
        )
        self.offset_slider1.set(self.channel_offsets[0])
        self.offset_slider1.pack()

        # Ch2 vertical slider
        slider2_frame = ttk.Frame(offset_frame)
        slider2_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(slider2_frame, text="Ch2 Offset").pack()
        self.offset_slider2 = tk.Scale(
            slider2_frame, from_=8, to=-8, resolution=0.01,
            orient=tk.VERTICAL, command=lambda val: self.set_offset(1, float(val))
        )
        self.offset_slider2.set(self.channel_offsets[1])
        self.offset_slider2.pack()
        

        # Autoscale & Start/Stop
        ttk.Button(self.controls_frame, text="Autoscale Both", command=self.autoscale_both).pack(pady=10)
        self.start_stop_button = ttk.Button(self.controls_frame, text="Stop", command=self.toggle_start_stop)
        self.start_stop_button.pack(pady=10)

        self.freq_label = ttk.Label(self.controls_frame, text="Ch1 Freq: -- Hz\nCh2 Freq: -- Hz")
        self.freq_label.pack(pady=10)

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.lines = [
            self.ax.plot(self.time, self.data_buffers[0], label="Channel 1", color='blue')[0],
            self.ax.plot(self.time, self.data_buffers[1], label="Channel 2", color='red')[0]
        ]
        self.offset_lines = [
            self.ax.axhline(self.channel_offsets[0], color='blue', linestyle=':', alpha=0.5),
            self.ax.axhline(self.channel_offsets[1], color='red', linestyle=':', alpha=0.5)
        ]

        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax.grid(True)
        self.ax.legend()
        self.update_plot_limits()

    def update_plot_limits(self):
        x_span = self.time_per_div * 10
        with self.lock:
            # Keep num_samples consistent with time array
            self.num_samples = min(int(self.sample_rate * x_span), 100000)  # Increased max to 100k
            self.time = np.linspace(0, x_span, self.num_samples)
            self.data_buffers = [np.zeros(self.num_samples), np.zeros(self.num_samples)]

        self.reconfigure_daq()

        self.ax.set_ylim(-8, 8)
        self.ax.set_xlim(0, x_span)
        
        # Update plot elements
        for i, line in enumerate(self.lines):
            line.set_xdata(self.time)
            line.set_ydata(self.data_buffers[i])
        
        # Update slider ranges
        self.time_scale_slider.set(self.time_per_div)
        #self.volt_scale_slider1.set(self.volts_per_div[0])
        #self.volt_scale_slider2.set(self.volts_per_div[1])

        # Update vertical slider ranges
        for ch, slider in enumerate([self.offset_slider1, self.offset_slider2]):
            slider.config(from_=8, to=-8, resolution=0.01)

            # Only update slider *if value has actually changed* (no trigger)
            current_val = slider.get()
            new_val = self.channel_offsets[ch]
            if abs(current_val - new_val) > 1e-3:
                slider.set(new_val)

        self.canvas.draw()

    def reconfigure_daq(self):
        try:
            if self.task is not None:
                self.task.stop()
                self.task.in_stream.input_buf_size = self.num_samples * 100
                self.task.timing.cfg_samp_clk_timing(
                    rate=self.sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.num_samples * 10
                )
                self.task.start()
                self.task_running = True
                time.sleep(0.1)
        except nidaqmx.errors.DaqError as e:
            print("DAQ reconfiguration failed:", e)

    def update_plot(self):
        if not self.running or not self.root.winfo_exists():
            return

        try:
            if not self.data_queue.empty():
                new_data = self.data_queue.get_nowait()
                new_data = np.asarray(new_data)

                # Accept any data size and adjust display accordingly
                if new_data.ndim == 2:
                    received_samples = new_data.shape[1]
                    display_samples = min(received_samples, self.num_samples)
                    
                    # Update time axis for the received samples
                    display_time = np.linspace(0, display_samples/self.sample_rate, display_samples)
                    
                    for ch in range(2):
                        # Take the most recent samples that fit our display
                        channel_data = new_data[ch, -display_samples:]
                        
                        # Apply vertical scale and offset
                        channel_data = channel_data / self.volts_per_div[ch] + self.channel_offsets[ch]

                        
                        # Update plot data
                        self.lines[ch].set_data(display_time, channel_data)

                    for ch in range(2):
                        self.offset_lines[ch].set_ydata([self.channel_offsets[ch]] * 2)
                    
                    self.canvas.draw_idle()
                else:
                    print(f"Unexpected data shape: {new_data.shape}")

        except Exception as e:
            print(f"update_plot exception: {e}")
        
        self.after_id = self.root.after(50, self.update_plot)

    def daq_reader(self):
        while True:
            try:
                if self.running and self.task is not None and self.task_running:
                    try:
                        if not self.task.is_task_done():
                            available = self.task.in_stream.avail_samp_per_chan
                            if available > 0:
                                data = self.task.read(
                                    number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,
                                    timeout=1.0
                                )
                                data = np.asarray(data)
                                if data.size > 0:
                                    if self.data_queue.full():
                                        self.data_queue.get_nowait()
                                    self.data_queue.put(data)
                        time.sleep(0.01)
                    except nidaqmx.errors.DaqError as e:
                        # Only print meaningful errors
                        if e.error_code != -200279 and e.error_code != -200983:
                            print(f"DAQ Read Error: {e}")
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)
            except Exception as ex:
                print(f"daq_reader exception: {ex}")
                time.sleep(0.1)

    def set_time_scale(self, value):
        self.time_per_div = value
        self.update_plot_limits()

    def set_volt_scale(self, channel, value):
        self.volts_per_div[channel] = value
        
        # Temporarily disable callback to prevent recursive updates
        self.volt_scale_slider1.configure(command='')
        self.volt_scale_slider2.configure(command='')

        # Only update the changed one
        if channel == 0:
            self.volt_scale_slider1.set(value)
        else:
            self.volt_scale_slider2.set(value)

        self.update_plot_limits()

        # Restore callbacks
        self.volt_scale_slider1.configure(command=lambda val: self.set_volt_scale(0, float(val)))
        self.volt_scale_slider2.configure(command=lambda val: self.set_volt_scale(1, float(val)))

    def round_to_125(self, value):
        decade = 10 ** np.floor(np.log10(value))
        normalized = value / decade
        if normalized <= 1.5:
            return 1 * decade
        elif normalized <= 3:
            return 2 * decade
        else:
            return 5 * decade

    def autoscale_both(self):
        try:
            freq_texts = ["Ch1 Freq: -- Hz", "Ch2 Freq: -- Hz"]
            
            # Step 1: Fully stop DAQ system
            self.running = False
            self.task_running = False
            time.sleep(0.1)  # Let DAQ settle

            # Disable sliders temporarily to prevent interference
            self.volt_scale_slider1.configure(state=tk.DISABLED)
            self.volt_scale_slider2.configure(state=tk.DISABLED)

            if self.task:
                self.task.stop()
                self.task.close()
                self.task = None
                time.sleep(0.1)

            # Step 2: Create temporary task to measure signal
            task_name = f"AutoscaleTempTask_{int(time.time() * 1000)}"
            with nidaqmx.Task(task_name) as temp_task:
                for device_name in self.device_names:
                    temp_task.ai_channels.add_ai_voltage_chan(
                        device_name,
                        min_val=-10,
                        max_val=10,
                        terminal_config=TerminalConfiguration.RSE
                    )
                temp_task.timing.cfg_samp_clk_timing(
                    self.sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=min(50000, self.num_samples)
                )
                temp_task.start()
                data = temp_task.read(
                    number_of_samples_per_channel=min(50000, self.num_samples),
                    timeout=5.0
                )
                data = np.asarray(data)

                for ch in [0, 1]:
                    channel_data = data[ch]
                    peak_to_peak = np.max(channel_data) - np.min(channel_data)
                    if peak_to_peak > 0:
                        self.volts_per_div[ch] = max(0.001, self.round_to_125(peak_to_peak / 6))
                    
                    mean_val = np.mean(channel_data)
                    zero_crossings = np.where(np.diff(np.sign(channel_data - mean_val)))[0]
                    if len(zero_crossings) >= 2:
                        periods = np.diff(zero_crossings[::2]) / self.sample_rate
                        if len(periods) > 0:
                            avg_period = np.mean(periods)
                            freq = 1 / avg_period
                            freq_texts[ch] = f"Ch{ch+1} Freq: {freq:.2f} Hz"
                            if ch == 0:
                                desired_time_span = avg_period * 3
                                self.time_per_div = self.round_to_125(desired_time_span / 10)

                    self.channel_offsets[ch] = (1 - 2*ch) * self.volts_per_div[ch] * 2

            # Step 3: Update GUI & Restart DAQ
            self.freq_label.config(text="   ".join(freq_texts))
            self.update_plot_limits()
            self.start_daq_task()
            self.task_running = True
            self.running = True

        except Exception as e:
            print(f"Autoscale failed: {str(e)}")
            self.running = True
            if self.task is None:
                self.start_daq_task()

        finally:
            # Re-enable the sliders after autoscale
            self.volt_scale_slider1.configure(state=tk.NORMAL)
            self.volt_scale_slider2.configure(state=tk.NORMAL)

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

    def start_daq_task(self):
        """Helper method to start DAQ task with proper configuration"""
        try:
            self.task = nidaqmx.Task()
            for device_name in self.device_names:
                self.task.ai_channels.add_ai_voltage_chan(
                    device_name,
                    min_val=-10,
                    max_val=10,
                    terminal_config=TerminalConfiguration.RSE
                )
            self.task.timing.cfg_samp_clk_timing(
                self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.num_samples * 5  # Larger buffer
            )
            self.task.in_stream.input_buf_size = self.num_samples * 100  # Very large buffer
            self.task.start()
            self.task_running = True
            time.sleep(0.1)  # Initialization delay
        except Exception as e:
            print("DAQ start failed:", e)
            raise

    def set_offset(self, channel, value):
        self.channel_offsets[channel] = value
        if hasattr(self, 'offset_lines'):
            self.offset_lines[channel].set_ydata([value, value])
        if hasattr(self, 'lines'):
            ydata = self.lines[channel].get_ydata() - self.channel_offsets[channel] + value
            self.lines[channel].set_ydata(ydata)
        self.canvas.draw_idle()

    def toggle_start_stop(self):
        self.running = not self.running
        self.start_stop_button.config(text="Stop" if self.running else "Start")

    def close(self):
        self.running = False
        time.sleep(0.2)
        try:
            if self.task:
                self.task.close()
        except Exception as e:
            print("Error while closing DAQ task:", e)

        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = Oscilloscope(root)

    def on_closing():
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()