def Welch(data, fs=2000, nperseg=512, same=True):
    from scipy.signal import welch
    from scipy.interpolate import interp1d
    import numpy as np

    # data.shape을 (num_data, time_points) 형태로 변형
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # ex) (4000,) -> (1, 4000)

    # Welch's method 수행
    results = [welch(data[i], fs, nperseg=nperseg) for i in range(len(data))]
    freqs, psd = map(np.array, zip(*results)) # psd.shape = (1000, 257) 같은 형태
    
    if same: # input time series와 같은 shape의 output psd를 얻고 싶을 때
        
        # 기존 frequency의 scale을 유지하면서 4000개의 points로 upsampling
        upsampled_freqs = np.array([np.linspace(freqs[0][0], freqs[0][-1], data.shape[1]) for i in range(len(data))])
        upsampled_psd = np.zeros((data.shape[0], data.shape[1]))
        
        # 선형 보간법을 사용하여 psd를 upsampling
        for i in range(psd.shape[0]):
            interp_func = interp1d(freqs[0], psd[i], kind='linear')
            upsampled_psd[i] = interp_func(upsampled_freqs[0])
        
        return upsampled_freqs, upsampled_psd
    
    return freqs, psd