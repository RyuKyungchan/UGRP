def Welch(data, fs=2000, nperseg=512, same=True):
    from scipy.signal import welch
    from scipy.interpolate import interp1d
    import numpy as np

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # (4000,) -> (1, 4000)
        
    results = [welch(data[i], fs, nperseg=nperseg) for i in range(len(data))]
    freqs, psd = map(np.array, zip(*results)) # psd.shape = (1000, 257) 같은 형태
    
    if same:
        # 기존 주파수 범위를 유지하면서 4000개의 포인트로 확장
        new_freqs = np.array([np.linspace(freqs[0][0], freqs[0][-1], data.shape[1]) for i in range(len(data))])
        upsampled_psd = np.zeros((data.shape[0], data.shape[1]))
        
        for i in range(psd.shape[0]):
            # 선형 보간법을 사용하여 upsampling
            interp_func = interp1d(freqs[0], psd[i], kind='linear')
            upsampled_psd[i] = interp_func(new_freqs[0])
        
        return new_freqs, upsampled_psd
    
    return freqs, psd