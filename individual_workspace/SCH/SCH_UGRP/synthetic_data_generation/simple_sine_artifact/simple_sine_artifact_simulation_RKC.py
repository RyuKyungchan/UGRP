# Import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import sim functions
from neurodsp.sim.combined import sim_combined, sim_peak_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.utils import set_random_seed

# Import function to compute power spectra
from neurodsp.spectral import compute_spectrum

# Import utilities for plotting data
from neurodsp.utils import create_times
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

# Set some general settings, to be used across all simulations
num_signals = 1000
fs = 2000 #sampling 주파수
n_seconds = 2 #simulation 지속시간
times = create_times(n_seconds, fs) #시간 백터

# Define the components of the combined signal to simulate
n_neurons_range = (800, 1200)  # range for number of neurons
firing_rate_range = (2, 4)     # range for firing rate
t_ker_range = (0.8, 1.2)       # range for t_ker
tau_r_range = (0.001, 0.003)   # range for tau_r
tau_d_range = (0.015, 0.025)   # range for tau_d
freq_range = (15, 25)          # range for oscillation frequency
amplitude_range = (5,6)   # range for amplitude_range

# Sample values from the specified ranges
n_neurons = np.random.randint(n_neurons_range[0], n_neurons_range[1] + 1)
firing_rate = np.random.uniform(firing_rate_range[0], firing_rate_range[1])
t_ker = np.random.uniform(t_ker_range[0], t_ker_range[1])
tau_r = np.random.uniform(tau_r_range[0], tau_r_range[1])
tau_d = np.random.uniform(tau_d_range[0], tau_d_range[1])
freq = np.random.uniform(freq_range[0], freq_range[1])


data_signal = pd.DataFrame()
data_signal_with_sine = pd.DataFrame()
data_sine_wave = pd.DataFrame()

for idx in range(num_signals):
    components = {
        'sim_synaptic_current': {
        'n_neurons': n_neurons,
        'firing_rate': firing_rate,
        't_ker': t_ker,
        'tau_r': tau_r,
        'tau_d': tau_d
        },
        'sim_oscillation': {
        'freq': freq
        }
    }
    # Simulate an oscillation over an aperiodic component
    signal = sim_combined(n_seconds, fs, components)
    # Plot the simulated data, in the time domain
    ##plot_time_series(times, signal)
    
    # Generate a 130 Hz sine wave
    t = np.arange(0, n_seconds, 1/fs)
    amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
    sine_wave = amplitude * np.sin(2 * np.pi * 130 * t)
    # Plot the artifact, in the time domain
    ##plot_time_series(times[:100], sine_wave[:100])

    # Add the sine wave to the simulated signal
    signal_with_sine = signal + sine_wave
    # Plot the simulated data, in the time domain
    ##plot_time_series(times, signal_with_sine)
    
    import matplotlib.pyplot as plt
    ##plt.figure()
    ##plt.plot(times, signal_with_sine,'k')
    ##plt.plot(times, signal, 'r')
        
    # Add the signal as a new column in the DataFrame
    data_signal[f'sim_{idx+1}'] = signal
    data_sine_wave[f'sin_{idx+1}'] = sine_wave
    data_signal_with_sine[f'atf_{idx+1}'] = signal_with_sine
    
    
# Add the time vector as the first column (time flow column-wise)
data_signal.insert(0, 'Time', times)
data_sine_wave.insert(0, 'Time', times)
data_signal_with_sine.insert(0, 'Time', times)

# 현재 스크립트 파일의 디렉토리 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f'Script directory: {script_dir}')

# 파일 경로 설정 (CSV)
file_path_signal_csv = os.path.join(script_dir, 'simulatesignal.csv')
file_path_sine_wave_csv = os.path.join(script_dir, 'sine_wave.csv')
file_path_signal_with_sine_csv = os.path.join(script_dir, 'simulatesignalwithartifact.csv')

# 파일 경로 설정 (NPY)
file_path_signal_npy = os.path.join(script_dir, 'data_signal.npy')
file_path_sine_wave_npy = os.path.join(script_dir, 'data_sine_wave.npy')
file_path_signal_with_sine_npy = os.path.join(script_dir, 'data_signal_with_sine.npy')

# Save DataFrames to CSV files
file_path1 = '../data/DNN_data/simulatesignal.csv'
# os.makedirs(os.path.dirname(file_path1), exist_ok=True)
data_signal.to_csv(file_path1, index=False)
print(f'DataFrame (signal) has been saved to {file_path1}')

file_path2 = '../data/DNN_data/sine_wave.csv'
# os.makedirs(os.path.dirname(file_path2), exist_ok=True)
data_signal.to_csv(file_path2, index=False)
print(f'DataFrame (signal) has been saved to {file_path2}')

file_path3 = '../data/DNN_data/simulatesignalwithartifact.csv'
# os.makedirs(os.path.dirname(file_path3), exist_ok=True)
data_signal.to_csv(file_path3, index=False)
print(f'DataFrame (signal_with_sine) has been saved to {file_path3}')

# # Save arrays to NPY files
# np.save(file_path_signal_npy, data_signal.to_numpy())
# np.save(file_path_sine_wave_npy, data_sine_wave.to_numpy())
# np.save(file_path_signal_with_sine_npy, data_signal_with_sine.to_numpy())

# print(f'Numpy array (data_signal) has been saved to {file_path_signal_npy}')
# print(f'Numpy array (data_sine_wave) has been saved to {file_path_sine_wave_npy}')
# print(f'Numpy array (data_signal_with_sine) has been saved to {file_path_signal_with_sine_npy}')

# # example for loading data

# # Load arrays from NPY files
# loaded_data_signal = np.load(file_path_signal_npy)
# loaded_data_sine_wave = np.load(file_path_sine_wave_npy)
# loaded_data_signal_with_sine = np.load(file_path_signal_with_sine_npy)

# # Display shape of the loaded data
# print(loaded_data_signal.shape)
# print(loaded_data_sine_wave.shape)
# print(loaded_data_signal_with_sine.shape)

# # Plot loaded data
# plt.plot(loaded_data_signal[:, 0])
# plt.show()