
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

    ### Plot all in one ###

    t = np.linspace(0, 2, num=4000) 
    start_time = 1; # [sec]
    end_time = 1.5; # [sec]
    fs = 2000
    start_pts = int(start_time*fs)
    end_pts = int(end_time*fs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ### Time domain Plotting ###

    # main timeseries plot
    axes[0].plot(t[start_pts:end_pts], Contaminated[0, start_pts:end_pts], label="Contaminated", color="orange", alpha=1, linewidth=1)
    axes[0].plot(t[start_pts:end_pts], Clean[0, start_pts:end_pts], label="Clean", color='dodgerblue', alpha=1, linewidth=1)
    axes[0].legend(prop={'size': 10}, loc='lower left')
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Time Domain Plot")

    # zoom-in inset plot
    inset_axis = axes[0].inset_axes((0.12, 0.25, 0.5, 0.35))
    inset_axis.plot(t[start_pts : start_pts + 50], Contaminated[0, start_pts : start_pts + 50], color="orange")
    inset_axis.plot(t[start_pts : start_pts + 50], Clean[0, start_pts : start_pts + 50], color='dodgerblue')
    axes[0].indicate_inset_zoom(inset_axis, edgecolor="black", alpha=0.4)
    inset_axis.patch.set_alpha(0.7)

    
    ### Frequency domain Plottig ###  

    freqs, _, _, psd_Contaminated = FFT(Contaminated, fs=2000, single_sided=True)
    _, _, _, psd_Clean = FFT(Clean, fs=2000, single_sided=True)

    axes[1].semilogy(freqs[1:600], psd_Contaminated[0, 1:600], label="Contaminated", color='orange', alpha = 1, linewidth=1)
    axes[1].semilogy(freqs[1:600], psd_Clean[0, 1:600], label="Clean", color='dodgerblue', alpha = 1, linewidth=1)
    axes[1].legend(prop={'size': 10}, loc='lower left')
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Log power (dB/Hz)")
    axes[1].set_title("Frequency Domain Plot")

    fig.tight_layout()
    plt.show()


    ### Plot [Contaminated / Artifact / Clean] ###

    plt.figure(figsize=(15,7))
    plt.subplot(3, 2, 1)
    plt.plot(t[start_pts:end_pts], Contaminated[0][start_pts:end_pts], color='orange')
    plt.xlabel("Time (seconds)"); plt.title('Contaminated Signal')

    plt.subplot(3, 2, 3)
    plt.plot(t[start_pts:end_pts], Artifact[0][start_pts:end_pts], color='tomato')
    plt.xlabel("Time (seconds)"); plt.title('Artifact Signal')

    plt.subplot(3, 2, 5)
    plt.plot(t[start_pts:end_pts], Clean[0][start_pts:end_pts], color='dodgerblue')
    plt.xlabel("Time (seconds)"); plt.title('Clean Signal')

    # Zoom-In 
    plt.subplot(3, 2, 2)
    plt.plot(t[start_pts : start_pts + 200], Contaminated[0][start_pts : start_pts + 200], color='orange')
    plt.xlabel("Time (seconds)"); plt.title('Contaminated Signal (Zoom-In)')

    plt.subplot(3, 2, 4)
    plt.plot(t[start_pts : start_pts + 200], Artifact[0][start_pts : start_pts + 200], color='tomato')
    plt.xlabel("Time (seconds)"); plt.title('Artifact Signal (Zoom-In)')

    plt.subplot(3, 2, 6)
    plt.plot(t[start_pts : start_pts + 200], Clean[0][:200], color='dodgerblue')
    plt.xlabel("Time (seconds)"); plt.title('Clean Signal (Zoom-In)')

    plt.tight_layout()
    plt.show()

    ### Time domain Plotting ###
    # plt.figure(figsize=(8, 3))
    # plt.plot(t[start_pts:end_pts], Artifact[0][start_pts:end_pts], label='Artifact Signal', color='tomato', alpha=1, linewidth=0.5)
    # plt.plot(t[start_pts:end_pts], Contaminated[0][start_pts:end_pts], label='Contaminated Signal', color='orange', alpha=1, linewidth=0.5)
    # plt.plot(t[start_pts:end_pts], Clean[0][start_pts:end_pts], label='Clean Signal', color='dodgerblue', alpha=1, linewidth=0.5)
    # plt.xlabel('Time (seconds)');plt.ylabel('Amplitude');plt.title('Contaminated vs Clean Signal')
    # plt.legend(prop={'size': 10}, loc='lower left')
    # plt.show()

    ### Frequency domain Plottig ###  
    # freqs, _, _, Contaminated_psd = FFT(Contaminated, fs=2000, single_sided=True)
    # _, _, _, Clean_psd = FFT(Clean, fs=2000, single_sided=True)

    # plt.figure(figsize=(10, 7))
    # plt.plot(freqs[1:], np.log10(Contaminated_psd[0][1:]), label='Contaminated Signal', color='orange', alpha=1, linewidth=0.7)
    # plt.plot(freqs[1:], np.log10(Clean_psd[0][1:]), label='Clean Signal', color='dodgerblue', alpha=1, linewidth=0.7)
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Amplitude')
    # plt.title('Contaminated vs Clean Signal')
    # plt.legend()
    # plt.show()
    return Contaminated, Clean, Artifact



def Loss_Plot(loss_list):

    """
    Train / Test Loss의 진행과정을 Plot하는 함수
    parameter: loss_list
    return: None
    """

    import matplotlib.pyplot as plt

    # train loss plot
    plt.figure(figsize=(20, 3))
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss / Epoch")
    plt.show()

    min_index, min_value = 0, 1
    for idx, val in enumerate(loss_list):
        if val < min_value:
            min_index = idx
            min_value = val
    print("Minimal Loss:", min_value, f"[{min_index}]\n")




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
    freqs, _, _, psd_Contaminated = FFT(Contaminated, fs=2000, single_sided=True)
    _, _, _, psd_Clean = FFT(Clean, fs=2000, single_sided=True)
    _, _, _, psd_SACed = FFT(SACed, fs=2000, single_sided=True)

    # Frequency MAE / MSE
    print("<Frequency Domain Error>")
    print(f"Mean Absolute Error: {mean_absolute_error(psd_SACed, psd_Clean)}")
    print(f"Mean Squared Error: {mean_squared_error(psd_SACed, psd_Clean)}")

    # 결과 플로팅
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[1:600], np.log10(psd_Contaminated[0][1:600]), label='Contaminated Signal', color='orange', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], np.log10(psd_Clean[0][1:600]), label='Clean Signal', color='dodgerblue', alpha=0.7, linewidth=0.7)
    plt.plot(freqs[1:600], np.log10(psd_SACed[0][1:600]), label='SACed Signal', color='red', alpha=0.7, linewidth=0.7)
    plt.title('Power Spectrum of Predicted and Actual Signals')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=130, color='black', linestyle='--',label='130 Hz', linewidth=0.7)

    return None

def Result_Plot_v2(Contaminated, SACed, Clean, save_path='../../../result/', save_title='latest result'):
    
    """
    모델의 결과를 plot하고 save하는 함수
    parameter: Contaminated, SACed_signal, Clean, save_path, save_title
    return: None

    save_path: 결과를 저장할 경로 ex) '../../../result/CNN'
    save_title: 어떤 코드를 실행한 결과인지 명시. 저장되는 파일명의 앞부분에 해당됨 ex) CNN_IO_time_L_time
    
    ex) 
    save_title + '_errors.npy'
    save_title + '_fig.png'
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    ### Time domain Plotting ###

    t = np.linspace(0, 2, num=4000) 
    start_time = 1; # [sec]
    end_time = 1.5; # [sec]
    fs = 2000
    start_pts = int(start_time*fs)
    end_pts = int(end_time*fs)

    fig, axes = plt.subplots(2, 1, figsize=(5, 8))
    inset_axis = axes[0].inset_axes((0.11, 0.27, 0.5, 0.35))

    # main timeseries plot
    axes[0].plot(t[start_pts:end_pts], Contaminated[0, start_pts:end_pts], label="Contaminated", color="orange", alpha=1, linewidth=1)
    axes[0].plot(t[start_pts:end_pts], Clean[0, start_pts:end_pts], label="Clean", color='dodgerblue', alpha=1, linewidth=1)
    axes[0].plot(t[start_pts:end_pts], SACed[0, start_pts:end_pts], label="SACed", color='red', alpha=1, linewidth=1)
    axes[0].legend(prop={'size': 8}, loc='lower left')
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude (mV)"); 
    axes[0].set_xlim(t[start_pts-20], t[end_pts+20])
    axes[0].set_title("Time Domain Plot")

    # zoom-in(x1) inset plot
    inset_axis.plot(t[start_pts + 200 : start_pts + 400], Clean[0, start_pts + 200 : start_pts + 400], color='dodgerblue', linewidth=0.9)
    inset_axis.plot(t[start_pts + 200 : start_pts + 400], SACed[0, start_pts + 200 : start_pts + 400], color='red', linewidth=0.9)
    axes[0].indicate_inset_zoom(inset_axis, edgecolor="black", alpha=0.8, lw=1.2)
    inset_axis.plot(t[start_pts + 200 : start_pts + 400], Contaminated[0, start_pts + 200 : start_pts + 400], color='orange', linewidth=0.8)
    inset_axis.patch.set_alpha(1)
    inset_axis.set_xlim(t[start_pts + 200-1], t[start_pts + 400])
    min_val = min(Clean[0, start_pts + 200 : start_pts + 400].min(), SACed[0, start_pts + 200 : start_pts + 400].min())
    max_val = max(Clean[0, start_pts + 200 : start_pts + 400].max(), SACed[0, start_pts + 200 : start_pts + 400].max())
    inset_axis.set_ylim(min_val-0.2, max_val+0.2)

    ### Frequency domain Plottig ###  

    freqs, _, _, psd_Contaminated = FFT(Contaminated, fs=2000, single_sided=True)
    _, _, _, psd_Clean = FFT(Clean, fs=2000, single_sided=True)
    _, _, _, psd_SACed = FFT(SACed, fs=2000, single_sided=True)

    axes[1].semilogy(freqs[1:600], psd_Contaminated[0, 1:600], label="Contaminated", color='orange', alpha = 1, linewidth=1)
    axes[1].semilogy(freqs[1:600], psd_Clean[0, 1:600], label="Clean", color='dodgerblue', alpha = 1, linewidth=1)
    axes[1].semilogy(freqs[1:600], psd_SACed[0, 1:600], label="SACed", color='red', alpha = 1, linewidth=1)
    axes[1].legend(prop={'size': 8}, loc='lower left')
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Log power (dB/Hz)")
    axes[1].set_xlim(freqs[1]-5, freqs[600]+5)
    axes[1].set_title("Frequency Domain Plot")

    fig.tight_layout()
    
    plt.savefig(save_path + save_title + "_fig" + ".png")# figure를 저장
    plt.show()

    ### MAE / MSE ###

    errors = np.array([
        mean_absolute_error(SACed, Clean),
        mean_squared_error(SACed, Clean),
        mean_absolute_error(psd_SACed, psd_Clean),
        mean_squared_error(psd_SACed, psd_Clean)])
    
    errors = np.round(errors, 4)

    print("<Time Domain Error>")
    print(f"Mean Absolute Error: {errors[0]}")
    print(f"Mean Squared Error: {errors[1]}")

    print("<Frequency Domain Error>")
    print(f"Mean Absolute Error: {errors[2]}")
    print(f"Mean Squared Error: {errors[3]}")

    np.save(f"{save_path}{save_title + '_errors'}.npy", errors) # 결과를 numpy 배열로 저장



def Result_Plot_v3(Contaminated, SACed, Clean, save_path='../../../result/', save_title='latest result'):
    
    """
    모델의 결과를 plot하고 save하는 함수
    parameter: Contaminated, SACed_signal, Clean, save_path, save_title
    return: None

    save_path: 결과를 저장할 경로 ex) '../../../result/CNN'
    save_title: 어떤 코드를 실행한 결과인지 명시. 저장되는 파일명의 앞부분에 해당됨 ex) CNN_IO_time_L_time
    
    ex) 
    save_title + '_time_domain_errors.npy'
    save_title + '_freq_domain_errors.npy'
    save_title + '_fig.png'
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    ### Time MAE / MSE ###

    time_domain_errors = np.array([
        mean_absolute_error(SACed, Clean),
        mean_squared_error(SACed, Clean)])
    
    time_domain_errors = np.round(time_domain_errors, 4)

    np.save(f"{save_path}{save_title + '_time_domain_errors'}.npy", time_domain_errors) # 결과를 numpy 배열로 저장

    print("<Time Domain Error>")
    print(f"Mean Absolute Error: {time_domain_errors[0]}")
    print(f"Mean Squared Error: {time_domain_errors[1]}")


    ### Time domain Plotting ###

    t = np.linspace(0, 2, num=4000) 
    start_time = 1; # [sec]
    end_time = 1.5; # [sec]
    fs = 2000
    start_pts = int(start_time*fs)
    end_pts = int(end_time*fs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    inset_axis = axes[0].inset_axes((0.06, 0.4, 0.3, 0.2))

    # main timeseries plot
    axes[0].plot(t[start_pts:end_pts], Contaminated[0, start_pts:end_pts], label="Contaminated", color="orange", alpha=1, linewidth=1)
    axes[0].plot(t[start_pts:end_pts], Clean[0, start_pts:end_pts], label="Clean", color='dodgerblue', alpha=1, linewidth=1)
    axes[0].plot(t[start_pts:end_pts], SACed[0, start_pts:end_pts], label="SACed", color='red', alpha=1, linewidth=1)
    axes[0].legend(prop={'size': 10}, loc='lower left')
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude (mV)"); 
    axes[0].set_title("Time Domain Plot")

    # zoom-in(x1) inset plot
    inset_axis.plot(t[start_pts : start_pts + 100], Contaminated[0, start_pts : start_pts + 100], color="orange")
    inset_axis.plot(t[start_pts : start_pts + 100], Clean[0, start_pts : start_pts + 100], color='dodgerblue')
    inset_axis.plot(t[start_pts : start_pts + 100], SACed[0, start_pts : start_pts + 100], color='red')
    axes[0].indicate_inset_zoom(inset_axis, edgecolor="black", alpha=0.4)
    inset_axis.patch.set_alpha(0.7)

    # zoom-in(x2) inset plot
    inner_inset_axis = inset_axis.inset_axes((1.0, -1.6, 1.2, 1.2))  # 위치와 크기 조정
    inner_inset_axis.plot(t[start_pts : start_pts + 50], Clean[0, start_pts : start_pts + 50], color='dodgerblue')
    inner_inset_axis.plot(t[start_pts : start_pts + 50], SACed[0, start_pts : start_pts + 50], color='red')
    inset_axis.indicate_inset_zoom(inner_inset_axis, edgecolor="black", alpha=0.4)
    inner_inset_axis.patch.set_alpha(0.7)

    
    ### Frequency domain Plottig ###  

    freqs, _, _, psd_Contaminated = FFT(Contaminated, fs=2000, single_sided=True)
    _, _, _, psd_Clean = FFT(Clean, fs=2000, single_sided=True)
    _, _, _, psd_SACed = FFT(SACed, fs=2000, single_sided=True)

    ### Frequency MAE / MSE ###

    freq_domain_errors = np.array([
        mean_absolute_error(psd_SACed, psd_Clean),
        mean_squared_error(psd_SACed, psd_Clean)])
    
    freq_domain_errors = np.round(freq_domain_errors, 4)

    print("<Frequency Domain Error>")
    print(f"Mean Absolute Error: {freq_domain_errors[0]}")
    print(f"Mean Squared Error: {freq_domain_errors[1]}")

    np.save(f"{save_path}{save_title + '_freq_domain_errors'}.npy", freq_domain_errors) # 결과를 numpy 배열로 저장


    axes[1].semilogy(freqs[1:600], psd_Contaminated[0, 1:600], label="Contaminated", color='orange', alpha = 1, linewidth=1)
    axes[1].semilogy(freqs[1:600], psd_Clean[0, 1:600], label="Clean", color='dodgerblue', alpha = 1, linewidth=1)
    axes[1].semilogy(freqs[1:600], psd_SACed[0, 1:600], label="SACed", color='red', alpha = 1, linewidth=1)
    axes[1].legend(prop={'size': 10}, loc='lower left')
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Log power (dB/Hz)")
    axes[1].set_title("Frequency Domain Plot")

    fig.tight_layout()
    
    plt.savefig(save_path + save_title + "_fig" + ".png")# figure를 저장
    plt.show()





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