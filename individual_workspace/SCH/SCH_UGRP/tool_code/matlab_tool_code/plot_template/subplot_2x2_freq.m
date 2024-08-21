% info
fs = 1000;
t = 0:1/fs:1-1/fs;

non_Filtered = 1 * sin(2*pi*5*t);
Filtered = lowpass(non_Filtered, 10, fs)+ 0.5 * sin(2*pi*100*t);

%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%   FFT   %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(non_Filtered);
freqs = (0:n/2)*(fs/n);

fft_non_Filtered = abs(fft(non_Filtered));
fft_non_Filtered = fft_non_Filtered(1:n/2+1);
fft_non_Filtered(2:end) = 2*fft_non_Filtered(2:end);
power1 = fft_non_Filtered.^2;
fft_Filtered = abs(fft(Filtered));
fft_Filtered = fft_Filtered(1:n/2+1);
fft_Filtered(2:end) = 2*fft_Filtered(2:end);
power2 = fft_Filtered.^2;

%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%   Visualization   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure;
fig.Position(1:4) = [300 300 1150 800];
figtitle = 'Graph Title';

% Time domain subplot
subplot(2,2,1) % time domain
plot(t, non_Filtered, 'Color', '#4DBEEE', 'DisplayName', 'Non-Filtered'); hold on
plot(t, Filtered, 'r', 'DisplayName', 'Filtered'); hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 15)
title('Time Domain')
xlim([0 1]); ylim([-2 2])

% Time domain - close-up
subplot(2,2,2)
plot(t, non_Filtered, 'Color', '#4DBEEE', 'DisplayName', 'Non-Filtered'); hold on
plot(t, Filtered, 'r', 'DisplayName', 'Filtered'); hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 15)
title('Time Domain (Close-up)')
xlim([0 0.2]); ylim([-2 2]) % close-up

% Frequency domain
subplot(2,2,3) % frequency domain
plot(freqs, log10(power1), 'Color', '#4DBEEE', 'DisplayName', 'Non-Filtered'); hold on
plot(freqs, log10(power2), 'r', 'DisplayName', 'Filtered'); hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 15)
title('Frequency Domain')
xlim([0 600]); ylim([-35 20])

% Frequency domain - close-up
subplot(2,2,4)
plot(freqs, log10(power1), 'Color', '#4DBEEE', 'DisplayName', 'Non-Filtered'); hold on
plot(freqs, log10(power2), 'r', 'DisplayName', 'Filtered'); hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 15)
title('Frequency Domain (Close-up)')
xlim([0 120]); ylim([-35 20]) % close-up

% Figure 제목 설정
sgtitle(figtitle, 'FontSize', 20);
