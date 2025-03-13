import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt
import nidaqmx.constants
import numpy as np


# Create a DAQ Task
with nidaqmx.Task() as task:
    # Add a single AI channel (change "Dev2/ai4" to your actual channel)
    task.ai_channels.add_ai_voltage_chan("Dev2/ai4", min_val=-10, max_val=10, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
    
    # Configure sampling: 10 kHz, 1000 samples
    sample_rate = 10000.0  # 10 kHz
    num_samples = 1000
    task.timing.cfg_samp_clk_timing(sample_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=num_samples)

    # Explicitly read the required number of samples
    data = task.read(number_of_samples_per_channel=num_samples)

    # Convert to NumPy array for better plotting
    data = np.array(data)
    print(data)

    # Generate time axis (in seconds)
    time = np.linspace(0, num_samples / sample_rate, num_samples)

    # Plot the waveform correctly
    plt.figure(figsize=(10, 5))
    plt.plot(time, data, label="Acquired Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title("NI 6251 - Acquired Waveform")
    plt.legend()
    plt.grid()
    plt.show()
