
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

    # Plot All in One
    plt.figure(figsize=(20, 3))
    plt.plot(t, artifact[0], label='Artifact Signal', color='tomato', alpha=1, linewidth=0.7)
    plt.plot(t, sig_with_artifact[0], label='Contaminated Signal', color='orange', alpha=1, linewidth=0.7)
    plt.plot(t, sig[0], label='Clean Signal', color='dodgerblue', alpha=1, linewidth=0.7)
    plt.xlabel('Time (seconds)');plt.ylabel('Amplitude');plt.title('Contaminated vs Clean Signal')
    plt.legend()
    plt.show()

    return sig_with_artifact, sig, artifact

def Result_Plot(Contaminated_signal, Clean_signal, SACed_signal):

    """
    모델의 결과를 plot하는 함수
    parameter: Contaminated_signal, SACed_signal, Clean_signal
    return: None
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    ### Time domain Plotting ###
    print("<Time Domain Error>")
    print(f"Mean Absolute Error: {mean_absolute_error(Clean_signal, SACed_signal)}")
    print(f"Mean Squared Error: {mean_squared_error(Clean_signal, SACed_signal)}")

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

    fft_predicted = np.fft.fft(SACed_signal)
    fft_predicted = np.abs(fft_predicted[:n//2])
    fft_predicted[1:] = 2 * fft_predicted[1:]
    power_predicted = fft_predicted**2

    fft_actual = np.fft.fft(Clean_signal)
    fft_actual = np.abs(fft_actual[:n//2])
    fft_actual[1:] = 2 * fft_actual[1:]
    power_actual = fft_actual**2

    fft_nonSACed = np.fft.fft(Contaminated_signal)
    fft_nonSACed = np.abs(fft_nonSACed[:n//2])
    fft_nonSACed[1:] = 2 * fft_nonSACed[1:]
    power_nonSACed = fft_nonSACed**2

    # Frequency MAE / MSE
    print("<Frequency Domain Error>")
    print(f"Mean Absolute Error: {mean_absolute_error(np.log10(power_predicted), np.log10(power_actual))}")
    print(f"Mean Squared Error: {mean_squared_error(np.log10(power_predicted), np.log10(power_actual))}")

    # 결과 플로팅
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[1:600], np.log10(power_nonSACed)[1:600], label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], np.log10(power_actual)[1:600], label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], np.log10(power_predicted)[1:600], label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.title('Power Spectrum of Predicted and Actual Signals')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=130, color='black', linestyle='--',label='130 Hz', linewidth=0.7)

    return None




