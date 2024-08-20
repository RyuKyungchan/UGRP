
def Data_Load_Plot(datapath):
    
    """
    데이터를 loading 하고 plot 하는 함수
    return: sig_with_artifact, sig, artifact
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    Contaminated = np.load(datapath + "contaminated_by_realistic" + ".npy")
    Clean = np.load(datapath + "clean_data" + ".npy")
    Artifact = Contaminated - Clean

    print("Contaminated_data.shape:", Contaminated.shape)
    print("Clean_data.shape:", Clean.shape)

    t = np.linspace(0, 2, num=4000) 

    # Plot All in One
    plt.figure(figsize=(20, 3))
    plt.plot(t, Artifact[0], label='Artifact Signal', color='tomato', alpha=1, linewidth=0.7)
    plt.plot(t, Contaminated[0], label='Contaminated Signal', color='orange', alpha=1, linewidth=0.7)
    plt.plot(t, Clean[0], label='Clean Signal', color='dodgerblue', alpha=1, linewidth=0.7)
    plt.xlabel('Time (seconds)');plt.ylabel('Amplitude');plt.title('Contaminated vs Clean Signal')
    plt.legend()
    plt.show()

    # Plot [Contaminated / Artifact / Clean]
    plt.figure(figsize=(20,9))
    plt.subplot(3, 1, 1)
    plt.plot(t, Contaminated[0], color='orange')
    plt.xlabel("Time (seconds)")
    plt.title('Contaminated Signal')

    plt.subplot(3, 1, 2)
    plt.plot(t, Artifact[0], color='tomato')
    plt.xlabel("Time (seconds)")
    plt.title('Artifact Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t, Clean[0], color='dodgerblue')
    plt.xlabel("Time (seconds)")
    plt.title('Clean Signal')

    plt.tight_layout()
    plt.show()

    # Plot Zoom-In [Contaminated / Artifact / Clean]
    plt.figure(figsize=(20,8))
    plt.subplot(3, 1, 1)
    plt.plot(t[:200], Contaminated[0][:200], color='orange')
    plt.xlabel("Time (seconds)")
    plt.title('Contaminated Signal')

    plt.subplot(3, 1, 2)
    plt.plot(t[:200], Artifact[0][:200], color='tomato')
    plt.xlabel("Time (seconds)")
    plt.title('Artifact Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t[:200], Clean[0][:200], color='dodgerblue')
    plt.xlabel("Time (seconds)")
    plt.title('Clean Signal')

    plt.tight_layout()
    plt.show()

    ### Frequency domain Plottig ###  
    freqs, _, _, Contaminated_psd = FFT(Contaminated, fs=2000, single_sided=True)
    _, _, _, Clean_psd = FFT(Clean, fs=2000, single_sided=True)

    print(freqs.shape)
    print(Contaminated_psd.shape)

    plt.figure(figsize=(10, 7))
    plt.plot(freqs[1:], np.log10(Contaminated_psd[0][1:]), label='Contaminated Signal', color='orange', alpha=1, linewidth=0.7)
    plt.plot(freqs[1:], np.log10(Clean_psd[0][1:]), label='Clean Signal', color='dodgerblue', alpha=1, linewidth=0.7)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Contaminated vs Clean Signal')
    plt.legend()
    plt.show()

    return Contaminated, Clean, Artifact

def Train_Loss_Plot(loss_list):

    """
    Train Loss의 진행과정을 Plot하는 함수
    parameter: loss_list
    return: None
    """

    import matplotlib.pyplot as plt

    # train loss plot
    plt.figure(figsize=(20, 3))
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title("Train Loss")
    plt.show()

    min_index, min_value = 0, 1
    for idx, val in enumerate(loss_list):
        if val < min_value:
            min_index = idx
            min_value = val
    print("Minimal Train Loss:", min_value, f"[{min_index}]\n")

def Result_Plot(Contaminated, SACed, Clean):

    """
    모델의 결과를 plot하는 함수
    parameter: [Contaminated, SACed_signal, Clean] data
    return: None
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error


    ### Time domain Plotting ###
    print("<Time Domain Error>")
    print(f"Mean Absolute Error: {mean_absolute_error(SACed, Clean)}")
    print(f"Mean Squared Error: {mean_squared_error(SACed, Clean)}")

    Contaminated_signal = Contaminated[0]
    SACed_signal = SACed[0]
    Clean_signal = Clean[0]

    t = np.linspace(0, 2, num=4000)

    ## Plot [SACed / Clean]
    plt.figure(figsize=(20,8))
    plt.subplot(2, 1, 1)
    # plt.plot(t, Contaminated_signal, label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(t, Clean_signal, label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(t, SACed_signal, label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Value')
    plt.title('CNN result')
    plt.legend()

    plt.subplot(2, 1, 2)
    # plt.plot(t[:200], Contaminated_signal[:200], label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(t[:200], Clean_signal[:200], label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(t[:200], SACed_signal[:200], label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Value')
    plt.title('zoom-in')
    plt.legend()

    plt.tight_layout()
    plt.show()

    ## Plot [Contaminated / SACed / Clean]
    plt.figure(figsize=(20,8))
    plt.subplot(2, 1, 1)
    plt.plot(t, Contaminated_signal, label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(t, Clean_signal, label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(t, SACed_signal, label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Value')
    plt.title('CNN result')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t[:200], Contaminated_signal[:200], label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(t[:200], Clean_signal[:200], label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(t[:200], SACed_signal[:200], label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Value')
    plt.title('zoom-in')
    plt.legend()

    plt.tight_layout()
    plt.show()


    ### Frequency domain Plottig ###  
    freqs, _, _, power_Contaminated = FFT(Contaminated, fs=2000, single_sided=True)
    _, _, _, power_Clean = FFT(Clean, fs=2000, single_sided=True)
    _, _, _, power_SACed = FFT(SACed, fs=2000, single_sided=True)

    # fs = 2000
    # freqs = np.fft.fftfreq(n, d=1/fs)[:n//2]

    # def fft_func(signal):
    #     n = len(signal)
    #     fs = 2000

    #     fft_signal = np.fft.fft(signal)
    #     fft_signal = np.abs(fft_signal[:n//2])
    #     fft_signal[1:] = 2*fft_signal[1:]
    #     power_signal = fft_signal**2

    #     return np.log10(power_signal)

    power_Contaminated = np.array([freqs])
    power_SACed = np.array([freqs])
    power_Clean = np.array([freqs])

    for x, y_pred, y in zip(Contaminated, SACed, Clean):
        power_Contaminated = np.vstack((power_Contaminated, fft_func(x)))
        power_SACed = np.vstack((power_SACed, fft_func(y_pred)))
        power_Clean = np.vstack((power_Clean, fft_func(y)))

    power_Contaminated = np.delete(power_Contaminated, 0, axis=0)
    power_SACed = np.delete(power_SACed, 0, axis=0)
    power_Clean = np.delete(power_Clean, 0, axis=0)

    # Frequency MAE / MSE
    print("<Frequency Domain Error>")
    print(f"Mean Absolute Error: {mean_absolute_error(power_SACed, power_Clean)}")
    print(f"Mean Squared Error: {mean_squared_error(power_SACed, power_Clean)}")

    # 결과 플로팅
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[1:600], np.log10(power_Contaminated[0][1:600]), label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], np.log10(power_Clean[0][1:600]), label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], np.log10(power_SACed[0][1:600]), label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.title('Power Spectrum of Predicted and Actual Signals')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=130, color='black', linestyle='--',label='130 Hz', linewidth=0.7)

    return None

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



