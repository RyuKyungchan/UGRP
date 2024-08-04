
def Data_Load_Plot(datapath):
    
    """
    데이터를 loading 하고 plot 하는 함수
    return: sig_with_artifact, sig, artifact
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    sig_with_artifact = np.load(datapath + "data_with_non_sine_v2_varying" + ".npy")
    sig = np.load(datapath + "data_signal" + ".npy")
    artifact = sig_with_artifact - sig

    print("Contaminated_data.shape:", sig_with_artifact.shape)
    print("Clean_data.shape:", sig.shape)

    t = np.linspace(0, 2, num=4000) 

    # Plot All in One
    plt.figure(figsize=(20, 3))
    plt.plot(t, artifact[0], label='Artifact Signal', color='tomato', alpha=1, linewidth=0.7)
    plt.plot(t, sig_with_artifact[0], label='Contaminated Signal', color='orange', alpha=1, linewidth=0.7)
    plt.plot(t, sig[0], label='Clean Signal', color='dodgerblue', alpha=1, linewidth=0.7)
    plt.xlabel('Time (seconds)');plt.ylabel('Amplitude');plt.title('Contaminated vs Clean Signal')
    plt.legend()
    plt.show()

    # Plot [Contaminated / Artifact / Clean]
    plt.figure(figsize=(20,9))
    plt.subplot(3, 1, 1)
    plt.plot(t, sig_with_artifact[0], color='orange')
    plt.xlabel("Time (seconds)")
    plt.title('Contaminated Signal')

    plt.subplot(3, 1, 2)
    plt.plot(t, artifact[0], color='darkorange')
    plt.xlabel("Time (seconds)")
    plt.title('Artifact Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t, sig[0], color='dodgerblue')
    plt.xlabel("Time (seconds)")
    plt.title('Clean Signal')

    plt.tight_layout()
    plt.show()

    # Plot Zoom-In [Contaminated / Artifact / Clean]
    plt.figure(figsize=(20,8))
    plt.subplot(3, 1, 1)
    plt.plot(t[:200], sig_with_artifact[0][:200], color='orange')
    plt.xlabel("Time (seconds)")
    plt.title('Contaminated Signal')

    plt.subplot(3, 1, 2)
    plt.plot(t[:200], artifact[0][:200], color='tomato')
    plt.xlabel("Time (seconds)")
    plt.title('Artifact Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t[:200], sig[0][:200], color='dodgerblue')
    plt.xlabel("Time (seconds)")
    plt.title('Clean Signal')

    plt.tight_layout()
    plt.show()

    return sig_with_artifact, sig, artifact

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
    n = len(SACed_signal)

    fs = 2000
    freqs = np.fft.fftfreq(n, d=1/fs)[:n//2]

    def fft_func(signal):
        n = len(signal)
        fs = 2000

        fft_signal = np.fft.fft(signal)
        fft_signal = np.abs(fft_signal[:n//2])
        fft_signal[1:] = 2*fft_signal[1:]
        power_signal = fft_signal**2

        return np.log10(power_signal)

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
    plt.plot(freqs[1:600], power_Contaminated[0][1:600], label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], power_Clean[0][1:600], label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], power_SACed[0][1:600], label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.title('Power Spectrum of Predicted and Actual Signals')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=130, color='black', linestyle='--',label='130 Hz', linewidth=0.7)

    return None




