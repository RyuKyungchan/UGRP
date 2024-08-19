%% info - time domain
time = 0:1/fs:((size(before_data, 1))-1)/fs;

%% info - frequency domain
t_win = 1; % [sec]
n_win = t_win*fs;

% [pxx_before, freqs_before] = pwelch(before_data, hamming(n_win), [], n_win, fs);
[pxx_during, freqs_during] = pwelch(during_data, hamming(n_win), [], n_win, fs);
[pxx_filtered, freqs] = pwelch(filtered_data, hamming(n_win), [], n_win, fs);

%% %%%%%%   Visualization of data with Power spectral density  %%%%%%

fig = figure;

hold on
% plot(freqs, log10(pxx_before),'g', 'DisplayName','Before Stim.')                                              % raw data before stimulation on target channel
plot(freqs, log10(pxx_during),'r','DisplayName','During Stim.')                                                 % raw data during stimulation on target channel
plot(freqs, log10(pxx_parrmed),'b','DisplayName','PARRMed')                                                     % PARRMed data during stimulation on target channel
legend('-dynamiclegend')
xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]')
set(gca, 'FontSize', 15)

xlim([0 13000])
print(fig, '경로 이름', '-dpng') % Saving figure by as a PNG file