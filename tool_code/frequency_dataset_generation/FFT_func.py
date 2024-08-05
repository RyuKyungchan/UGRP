def FFT(signal, fs=2000, single_sided=True):
    
    import numpy as np

    N = len(signal)
    
    if single_sided:
        freq = np.fft.rfftfreq(N, d=1/fs)
        fft_result = np.fft.rfft(signal)
        amplitude_spectrum = np.abs(fft_result)
        
        if N % 2 == 0:
            amplitude_spectrum[1:-1] *= 2
        else:
            amplitude_spectrum[1:] *= 2
    
    else:
        freq = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        fft_result = np.fft.fft(signal)
        amplitude_spectrum = np.fft.fftshift(np.abs(fft_result))
        
    power_spectrum = amplitude_spectrum ** 2
    psd = power_spectrum / N
    return freq, amplitude_spectrum, power_spectrum, psd