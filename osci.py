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
        self.single_mode = False

        # Initialize scales
        self.time_per_div = 0.1  # seconds/div
        self.volts_per_div = [1.0, 1.0]  # volts/div for each channel

        self.channel_offsets = [1.0, -1.0]  # Vertical offsets for each channel
        self.separation_factor = 1.5  # How much to separate the channels

        # Trigger
        self.single_mode = False
        self.single_mode_var = tk.BooleanVar(value=False)

        # Cursor variables
        self.cursors_active = False
        self.cursor_source = 0  # 0 for Channel 1, 1 for Channel 2
        self.h_cursors = [None, None]  # Horizontal (time) cursors
        self.v_cursors = [None, None]  # Vertical (voltage) cursors
        self.cursor_lines = []  # To store reference to cursor lines
        self.cursor_text = None  # For measurement display

        # Cursor flag variables
        self.cursor_flags = []  # To store flag handles
        self.flag_size = 0.02  # Size of flags relative to axis size

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
            from_=0.0001, to=0.5, 
            resolution=0.00001,
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

        # Add cursor controls
        ttk.Separator(self.controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Cursor source selection
        self.cursor_source_frame = ttk.Frame(self.controls_frame)
        self.cursor_source_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.cursor_source_frame, text="Cursor Source:").pack(side=tk.LEFT)
        self.cursor_source_var = tk.IntVar(value=0)
        ttk.Radiobutton(self.cursor_source_frame, text="Ch1", variable=self.cursor_source_var, 
                    value=0, command=self.update_cursor_source).pack(side=tk.LEFT)
        ttk.Radiobutton(self.cursor_source_frame, text="Ch2", variable=self.cursor_source_var, 
                    value=1, command=self.update_cursor_source).pack(side=tk.LEFT)
        
        self.cursor_button = ttk.Button(self.controls_frame, text="Enable Cursors", 
                                    command=self.toggle_cursors)
        self.cursor_button.pack(pady=5)


        ttk.Label(self.controls_frame, text="Trigger Level (Ch1)").pack(pady=(10, 0))
        self.trigger_level = tk.DoubleVar(value=0.0)
        self.trigger_slider = tk.Scale(
            self.controls_frame, from_=-10, to=10, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.trigger_level,
            command=self.on_trigger_change
        )

        self.trigger_slider.pack(fill=tk.X, padx=5, pady=5)

        self.single_mode_var = tk.BooleanVar(value=False)
        self.single_button = ttk.Checkbutton(
            self.controls_frame,
            text="Single Mode",
            variable=self.single_mode_var,
            command=self.toggle_single_mode
        )
        self.single_button.pack(pady=5)

        
        # Measurement display
        self.measurement_frame = ttk.Frame(self.controls_frame)
        self.measurement_frame.pack(fill=tk.X, pady=5)
        self.measurement_label = ttk.Label(self.measurement_frame, text="Measurements:")
        self.measurement_label.pack()
        self.measurement_text = tk.Text(self.measurement_frame, height=4, width=25)
        self.measurement_text.pack()
        self.measurement_text.config(state=tk.DISABLED)


        # Single Mode Toggle
        self.single_button = ttk.Checkbutton(
            self.controls_frame,
            text="Single Mode",
            variable=self.single_mode_var,
            command=self.toggle_single_mode
        )
        self.single_button.pack(pady=5)

    def on_trigger_change(self, val):
        """Update trigger level and line position"""
        trigger_val = float(val)
        self.trigger_level.set(trigger_val)
        
        # Update the trigger line position
        if hasattr(self, 'trigger_line'):
            self.trigger_line.set_ydata([trigger_val, trigger_val])
            self.canvas.draw_idle()
        
        # If in single mode, capture new data with this trigger level
        if self.single_mode:
            self.capture_and_display_once()


    def toggle_cursors(self):
        self.cursors_active = not self.cursors_active
        self.cursor_button.config(text="Disable Cursors" if self.cursors_active else "Enable Cursors")
        
        if self.cursors_active:
            self.activate_cursors()
        else:
            self.deactivate_cursors()
        
            
        self.canvas.draw()

    def activate_cursors(self):
        # Create cursor lines if they don't exist
        if not self.cursor_lines:
            color = 'blue' if self.cursor_source == 0 else 'red'
            
            # Initialize horizontal cursors at 25% and 75% of time window
            x_span = self.time_per_div * 10
            h1_pos = x_span * 0.25
            h2_pos = x_span * 0.75
            
            # Initialize vertical cursors at 25% and 75% of voltage range
            v1_pos = -3  # 25% of -8 to 8 range
            v2_pos = 3   # 75% of -8 to 8 range
            
            # Horizontal (time) cursors - make them taller for better visibility
            self.h_cursors[0] = self.ax.axvline(h1_pos, color='green', linestyle='--', 
                                            alpha=0.7, linewidth=1.5, ymin=0, ymax=1)
            self.h_cursors[1] = self.ax.axvline(h2_pos, color='green', linestyle='--', 
                                            alpha=0.7, linewidth=1.5, ymin=0, ymax=1)
            
            # Vertical (voltage) cursors - make them wider for better visibility
            self.v_cursors[0] = self.ax.axhline(v1_pos, color=color, linestyle=':', 
                                            alpha=0.7, linewidth=1.5, xmin=0, xmax=1)
            self.v_cursors[1] = self.ax.axhline(v2_pos, color=color, linestyle=':', 
                                            alpha=0.7, linewidth=1.5, xmin=0, xmax=1)
            
            self.cursor_lines = self.h_cursors + self.v_cursors
            
            # Add text for measurements
            self.cursor_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,
                                        bbox=dict(facecolor='white', alpha=0.7))
            self.create_cursor_flags()

        # Make cursors visible
        for line in self.cursor_lines:
            line.set_visible(True)
        if self.cursor_text:
            self.cursor_text.set_visible(True)
        
        # Update cursor appearance based on source
        self.update_cursor_appearance()
        
        # Connect mouse events
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.update_measurements()

    def create_cursor_flags(self):
        """Create visual flags at the ends of cursors for easier grabbing"""
        color = 'blue' if self.cursor_source == 0 else 'red'
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Remove existing flags if any
        for flag in self.cursor_flags:
            flag.remove()
        self.cursor_flags = []
        
        # Create flags for horizontal (time) cursors
        for i, line in enumerate(self.h_cursors):
            x = line.get_xdata()[0]
            # Top flag for first cursor, bottom for second
            y_flag = ylim[1] - (ylim[1]-ylim[0])*0.1 if i == 0 else ylim[0] + (ylim[1]-ylim[0])*0.1
            flag = self.ax.plot([x, x], [y_flag, line.get_ydata()[0]], 
                            color='green', linewidth=2, alpha=0.7)[0]
            
            #flag = self.ax.
            self.cursor_flags.append(flag)
        
        # Create flags for vertical (voltage) cursors
        for i, line in enumerate(self.v_cursors):
            y = line.get_ydata()[0]
            # Left flag for first cursor, right for second
            x_flag = xlim[0] + (xlim[1]-xlim[0])*0.1 if i == 0 else xlim[1] - (xlim[1]-xlim[0])*0.1
            flag = self.ax.plot([x_flag, line.get_xdata()[0]], [y, y], 
                            color=color, linewidth=2, alpha=0.7)[0]
            self.cursor_flags.append(flag)
        
        self.canvas.draw()

    def update_cursor_source(self):
        self.cursor_source = self.cursor_source_var.get()
        if self.cursors_active:
            self.update_cursor_appearance()
            self.update_measurements()

    def update_cursor_appearance(self):
        """Update cursor and flag colors based on selected source channel"""
        if not self.cursors_active:
            return
        
        color = 'blue' if self.cursor_source == 0 else 'red'
        
        # Update vertical cursor colors
        for line in self.v_cursors:
            if line:
                line.set_color(color)
        
        # Update vertical flag colors (flags 2 and 3)
        for flag in self.cursor_flags[2:]:
            if flag:
                flag.set_color(color)
        
        self.canvas.draw()

    def deactivate_cursors(self):
        # Hide cursor lines and flags
        for line in self.cursor_lines:
            line.set_visible(False)
        for flag in self.cursor_flags:
            flag.set_visible(False)
        if self.cursor_text:
            self.cursor_text.set_visible(False)
        
        # Disconnect mouse events
        if hasattr(self, 'cid_press'):
            self.canvas.mpl_disconnect(self.cid_press)
        if hasattr(self, 'cid_motion'):
            self.canvas.mpl_disconnect(self.cid_motion)
        if hasattr(self, 'cid_release'):
            self.canvas.mpl_disconnect(self.cid_release)
        
        self.canvas.draw()


    def on_press(self, event):
        if not self.cursors_active or event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Check cursor flags first (easier to hit)
        for i, flag in enumerate(self.cursor_flags):
            flag_x = flag.get_xdata()
            flag_y = flag.get_ydata()
            
            # Calculate distance to flag line
            if i < 2:  # Horizontal cursor flags (vertical lines)
                dist = abs(flag_x[0] - x) / x_range
                if dist < 0.02:  # Very sensitive for flags
                    self.dragging = ('h', i)
                    break
            else:  # Vertical cursor flags (horizontal lines)
                dist = abs(flag_y[0] - y) / y_range
                if dist < 0.02:  # Very sensitive for flags
                    self.dragging = ('v', i-2)  # Adjust index for v_cursors
                    break
        
        # If no flag clicked, check cursor lines directly
        if self.dragging is None:
            for i, line in enumerate(self.v_cursors):
                cursor_y = line.get_ydata()[0]
                dist = abs(cursor_y - y) / y_range
                if dist < min_dist and dist < 0.05:  # 5% of y-range
                    min_dist = dist
                    self.dragging = ('v', i)
                    break  # Found closest cursor
        
        # If we found a cursor to drag, store its initial position
        if self.dragging is not None:
            self.drag_start = (x, y)

    def on_motion(self, event):
        if not hasattr(self, 'dragging') or self.dragging is None or not event.inaxes:
            return
        
        x, y = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        if self.dragging[0] == 'h':  # Moving horizontal cursor
            idx = self.dragging[1]
            x = max(xlim[0], min(xlim[1], x))
            self.h_cursors[idx].set_xdata([x, x])
            # Update corresponding flag
            flag_idx = idx  # First two flags are for horizontal cursors
            flag = self.cursor_flags[flag_idx]
            flag.set_xdata([x, x])
            
        elif self.dragging[0] == 'v':  # Moving vertical cursor
            idx = self.dragging[1]
            y = max(ylim[0], min(ylim[1], y))
            self.v_cursors[idx].set_ydata([y, y])
            # Update corresponding flag
            flag_idx = idx + 2  # Last two flags are for vertical cursors
            flag = self.cursor_flags[flag_idx]
            flag.set_ydata([y, y])
        
        self.update_measurements()
        self.canvas.draw()

    def on_release(self, event):
        if hasattr(self, 'dragging'):
            del self.dragging
        if hasattr(self, 'drag_start'):
            del self.drag_start
        self.update_measurements()

    def update_measurements(self):
        if not self.cursors_active or not all(line.get_visible() for line in self.cursor_lines):
            return
        
        # Get cursor positions
        h1, h2 = [line.get_xdata()[0] for line in self.h_cursors]
        v1, v2 = [line.get_ydata()[0] for line in self.v_cursors]
        
        # Calculate time difference
        delta_t = abs(h2 - h1)
        freq = 1/delta_t if delta_t > 0 else 0
        
        # Calculate voltage difference for selected channel
        delta_v = abs(v2 - v1)
        
        # Convert to real voltage (remove scaling/offset)
        real_v1 = (v1 - self.channel_offsets[self.cursor_source]) * self.volts_per_div[self.cursor_source]
        real_v2 = (v2 - self.channel_offsets[self.cursor_source]) * self.volts_per_div[self.cursor_source]
        real_delta_v = abs(real_v2 - real_v1)
        
        # Update text display
        text = (f"Source: Ch{self.cursor_source+1}\n"
                f"Time Δ: {delta_t*1e3:.2f} ms\n"
                f"Freq: {freq:.2f} Hz\n"
                f"Voltage Δ: {delta_v:.2f} div\n"
                f"({real_delta_v:.3f} V)")
        
        if self.cursor_text:
            self.cursor_text.set_text(text)
        
        # Update measurement text box
        self.measurement_text.config(state=tk.NORMAL)
        self.measurement_text.delete(1.0, tk.END)
        self.measurement_text.insert(tk.END, text)
        self.measurement_text.config(state=tk.DISABLED)
        
        self.canvas.draw()

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

        # Trigger line
        self.trigger_line = self.ax.axhline(
            y=self.trigger_level.get(), color='magenta',
            linestyle='--', linewidth=1.5, alpha=0.7, label='Trigger'
        )

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
        if self.cursors_active:
            self.create_cursor_flags()  # Recreate flags with new limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            for line in self.h_cursors:
                x = line.get_xdata()[0]
                if x < xlim[0] or x > xlim[1]:
                    line.set_xdata([xlim[0] + 0.1*(xlim[1]-xlim[0])])
            
            for line in self.v_cursors:
                y = line.get_ydata()[0]
                if y < ylim[0] or y > ylim[1]:
                    line.set_ydata([ylim[0] + 0.1*(ylim[1]-ylim[0])])

            
            self.update_measurements()

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
        if not self.root.winfo_exists() or (not self.running and not self.single_mode):
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

                    # Update offset lines
                    for ch in range(2):
                        self.offset_lines[ch].set_ydata([self.channel_offsets[ch]] * 2)
                    
                    # Update trigger line (ensure it's visible)
                    trigger_val = self.trigger_level.get()
                    scaled_trigger = trigger_val / self.volts_per_div[0] + self.channel_offsets[0]
                    self.trigger_line.set_ydata([scaled_trigger, scaled_trigger])
                    
                    self.canvas.draw_idle()

        except Exception as e:
            print(f"update_plot exception: {e}")
        
        self.after_id = self.root.after(50, self.update_plot)


    def toggle_single_mode(self):
        self.single_mode = self.single_mode_var.get()

        if self.single_mode:
            self.running = False
            if self.task_running:
                try:
                    self.task.stop()
                    self.task_running = False
                except Exception as e:
                    print(f"Error stopping task for single mode: {e}")
            self.capture_and_display_once()
        else:
            self.running = True
            if not self.task_running:
                self.start_daq_task()


    def capture_and_display_once(self):
        try:
            with nidaqmx.Task() as temp_task:
                for device_name in self.device_names:
                    temp_task.ai_channels.add_ai_voltage_chan(
                        device_name, min_val=-10, max_val=10,
                        terminal_config=TerminalConfiguration.RSE
                    )
                temp_task.timing.cfg_samp_clk_timing(
                    self.sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=self.num_samples * 2
                )
                temp_task.start()
                data = temp_task.read(
                    number_of_samples_per_channel=self.num_samples * 2,
                    timeout=2.0
                )
                data = np.asarray(data)

            trigger_val = self.trigger_level.get()
            
            # Find trigger point on channel 1 (but apply to both channels)
            trigger_idx = self.find_trigger_index_with_threshold(data[0], trigger_val)

            if trigger_idx and self.num_samples // 2 < trigger_idx < data.shape[1] - self.num_samples // 2:
                start_idx = trigger_idx - self.num_samples // 2
                data = data[:, start_idx:start_idx + self.num_samples]
            else:
                data = data[:, -self.num_samples:]

            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put(data)

        except Exception as e:
            print(f"Single capture failed: {e}")

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

    def find_trigger_index_with_threshold(self, data, threshold):
    
    #Returns index of first rising edge crossing the given threshold,
    #near the middle of the buffer.
    
        midpoint = len(data) // 2
        search_range = 5000

        start = max(1, midpoint - search_range)
        end = min(len(data) - 1, midpoint + search_range)

        for i in range(start, end):
            if data[i - 1] < threshold <= data[i]:
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

        # Safely cancel any scheduled updates
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception as e:
                print(f"Error canceling after_id: {e}")

        # Properly destroy the root to avoid idle_draw issues
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Oscilloscope(root)

    def on_closing():
        app.close()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()