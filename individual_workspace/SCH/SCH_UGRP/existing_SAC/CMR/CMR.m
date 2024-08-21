%%
clc
clear
close all

before = load('C:\Users\stell\Desktop\SCH\PARRM\data\PD_rest(noStim)_ch12-13.mat'); % before stimulation
during = load('C:\Users\stell\Desktop\SCH\PARRM\data\PD_OpenLoop_ch12-13.mat'); % during stimulation

%% info
num_ch = size(before.raw, 1);
stim_ch = 11;
target_ch = 12;

before = before.raw- mean(before.raw, 2); % preprocessing - DC offset removed data
during = during.raw - mean(during.raw, 2);

fs = 24414; % Sampling rate of simulated data
stimRate = 130; 
n = size(before, 2);
time = 0:1/fs:(n-1)/fs;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   CAR   %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CARed = during - median(during, 1); % referencing - 채널 방향으로 median을 구함

%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Visualization   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

before_data = before(target_ch, :);
during_data = during(target_ch, :);
CARed_data = CARed(target_ch, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization - time domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_time = 10; % [sec]
end_time = 30; % [sec]
start_pts = start_time*fs+1;
end_pts = end_time*fs;
t = time(start_pts:end_pts);

non_Filtered = during_data(start_pts:end_pts); % control data(before CAR)
Filtered = CARed_data(start_pts:end_pts);

%%
fig1 = figure;
fig1.Position(3:4) = [500 330];

plot(t, non_Filtered, 'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(t, Filtered,'r','DisplayName','CARed');hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 10)

ylim([-0.1 0.04])

fig1_title = 'CARed data - time domain';
sgtitle(fig1_title, 'FontSize', 11);
fig1;

%% time domain (closed up)
fig2 = figure;
fig2.Position(3:4) = [500 330];

plot(t, non_Filtered, 'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(t, Filtered,'r','DisplayName','CARed');hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 10)

xlim([22.4 22.5]);
ylim([-0.001 0.002]);

fig2_title = 'CARed data (closed up) - time domain';
sgtitle(fig2_title, 'FontSize', 11);
fig2;

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization - FFT
%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(before_data);
frequencies = (0:n/2)*fs/n;



fft_during = fft(during_data); % control data(before CAR)
fft_during = fft_during(1:n/2+1);
amplitude_during = abs(fft_during);
amplitude_during(2:end) = 2*amplitude_during(2:end);
power_during = (amplitude_during.^2)/n; 

fft_CARed = fft(CARed_data); 
fft_CARed = fft_CARed(1:n/2+1);
amplitude_CARed = abs(fft_CARed);
amplitude_CARed(2:end) = 2*amplitude_CARed(2:end);
power_CARed = (amplitude_CARed.^2)/n; 

%%
fig3 = figure;
fig3.Position(3:4) = [500 330];

plot(frequencies, log10(power_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(frequencies, log10(power_CARed),'r','DisplayName','CARed');hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)
yticks(-8:2:10);

xlim([0 13000])

fig3_title = 'CARed data - FFT';
sgtitle(fig3_title, 'FontSize', 11);
fig3;

%% FFT (closed up)
fig4 = figure;
fig4.Position(3:4) = [500 330];

plot(frequencies, log10(power_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(frequencies, log10(power_CARed),'r','DisplayName','CARed');hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)
yticks(-8:2:10);

xlim([0 1200])

fig4_title = 'CARed data (closed up) - FFT';
sgtitle(fig4_title, 'FontSize', 11);
fig4;

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization - pwelch
%%%%%%%%%%%%%%%%%%%%%%%%%
t_win = 1; % [sec]
n_win = t_win*fs;

[pxx_during, freqs_during] = pwelch(during_data, hamming(n_win), [], n_win, fs); % control data(before CAR)
[pxx_car, freqs] = pwelch(CARed_data, hamming(n_win), [], n_win, fs); % CARed data
%%
fig5 = figure;
fig5.Position(3:4) = [500 330];

plot(freqs, log10(pxx_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(freqs, log10(pxx_car),'r','DisplayName','CARed');hold off
legend('-dynamiclegend');  xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)

xlim([0 13000]); ylim([-15 -5])

fig5_title = 'CARed data - pwelch';
sgtitle(fig5_title, 'FontSize', 11);
fig5;

%% pwelch (closed up)
fig6 = figure;
fig6.Position(3:4) = [500 330];

plot(freqs, log10(pxx_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(freqs, log10(pxx_car),'r','DisplayName','CARed');hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)

xlim([0 1200]); ylim([-15 -5])

fig6_title = 'CARed data (closed up) - pwelch';
sgtitle(fig6_title, 'FontSize', 11);
fig6;