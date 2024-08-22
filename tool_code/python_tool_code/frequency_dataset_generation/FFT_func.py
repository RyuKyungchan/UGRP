
def FFT(data, fs=2000, single_sided=True):
    
    import numpy as np
    
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # (4000,) -> (1, 4000)
    N = len(data[0])
    
    if single_sided:
        freqs = np.fft.rfftfreq(N, d=1/fs)
        fft_result = np.fft.rfft(data)
        amplitude_spectrum = np.abs(fft_result)
        power_spectrum = amplitude_spectrum ** 2
        
        if N % 2 == 0:
            amplitude_spectrum[1:-1] *= 2
            power_spectrum[1:-1] *= 2
        else:
            amplitude_spectrum[1:] *= 2
            power_spectrum[1:-1] *= 2
    
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        fft_result = np.fft.fft(data)
        amplitude_spectrum = np.fft.fftshift(np.abs(fft_result))
        power_spectrum = amplitude_spectrum ** 2
        
    psd = power_spectrum / N
    return freqs, amplitude_spectrum, power_spectrum, psd


# power spectral density plot
def compute_psd(data, fs, n_freqs):
    import numpy as np

    freqs = np.fft.rfftfreq(len(data), 1/fs)
    psd = np.abs(np.fft.rfft(data))**2 / (fs * len(data))
    return freqs[:n_freqs], psd[:n_freqs]