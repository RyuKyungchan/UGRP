clc
clear
close all

%% Loading Data
before = load('C:\Users\stell\Desktop\SCH\PARRM\data\PD_rest(noStim)_ch12-13.mat'); % before stimulation
during = load('C:\Users\stell\Desktop\SCH\PARRM\data\PD_OpenLoop_ch12-13.mat'); % during stimulation

%% info
num_ch = size(before.raw, 1);
stim_ch = 11;
target_ch = 12;

before = before.raw- mean(before.raw, 2);
before_data = before(target_ch, :)'; % data before stimulation on target channel

during_preproc = during.raw - mean(during.raw, 2); % data during stimulation with DC removed
during_data = during_preproc(target_ch, :)';

fs = 24414; % Sampling rate of simulated data
stimRate = 130; 
time = 0:1/fs:((size(before_data, 1))-1)/fs;

%% PARRM
addpath ('C:\Users\stell\Desktop\SCH\PARRM\PARRM-master\')

%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   Grid Search   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Grid Search - Finding True Period T
guessPeriod=fs*1/stimRate; % Starting point for period grid search                              % delta: 187.8000 samples
% gpu_raw = gpuArray(during_data);
Period = FindPeriodLFP(during_data,[1,length(during_data)-1],guessPeriod);                              % T: 188.0000 samples

%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   PARRM    %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setting appropriate parameters
sepChirp = fs; % Separation between subequent chirps in samples

winTime = 0.5;                                                                                  % Width of the window in sample space for PARRM filter [s]
skipTime = winTime / 10;                                                                        % Width of the time to ignore in each window in sample space [s]
perDist = 0.5;                                                                                  % Window in period space for which samples will be averaged [samples]

%% Filtering using PARRM filter

PARRM_Filtered = Filtering(during_data, fs, Period, winTime, skipTime, perDist); % time domain

%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Visualization   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Visualization of PARRMed data with designed parameters(winTime, skipTime, perDist)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization - time domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_time = 10; % [sec]
end_time = 30; % [sec]
start_pts = start_time*fs+1;
end_pts = end_time*fs;
t = time(start_pts:end_pts);

non_Filtered = during_data(start_pts:end_pts); % control data(before PARRM)
Filtered = PARRM_Filtered(start_pts:end_pts);

%%
fig1 = figure;
fig1.Position(3:4) = [500 330];

plot(t, non_Filtered,'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(t, Filtered,'r','DisplayName','PARRMed');hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 10)

% ylim([-0.1 0.05])

xlim([22.4 22.5]) % closed up  %     xlim([19.11 19.21]) % closed up
ylim([-0.8*10^(-3) 1.8*10^(-3)]) % closed up

fig1_title = ['PARRMed data - time domain'];
sgtitle(fig1_title, 'FontSize', 11);
fig1;

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization - FFT
%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(before_data);
frequencies = (0:n/2)*fs/n;

fft_during = fft(during_data); % control data(before PARRM)
fft_during = fft_during(1:n/2+1);
amplitude_during = abs(fft_during);
amplitude_during(2:end) = 2*amplitude_during(2:end);
power_during = (amplitude_during.^2)/n; 

fft_PARRMed = fft(PARRM_Filtered); 
fft_PARRMed = fft_PARRMed(1:n/2+1);
amplitude_PARRMed = abs(fft_PARRMed);
amplitude_PARRMed(2:end) = 2*amplitude_PARRMed(2:end);
power_PARRMed = (amplitude_PARRMed.^2)/n; 

%%
fig2 = figure;
fig2.Position(3:4) = [500 330];

plot(frequencies, log10(power_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(frequencies, log10(power_PARRMed),'r','DisplayName','PARRMed');hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)

% xlim([0 13000])

xlim([0 1200]) % closed up
ylim([-8 10]) % closed up

fig2_title = ['PARRMed data - FFT'];
sgtitle(fig2_title, 'FontSize', 11);
fig2;

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization - pwelch
%%%%%%%%%%%%%%%%%%%%%%%%%
t_win = 1; % [sec]
n_win = t_win*fs;

[pxx_during, freqs] = pwelch((during_data), hamming(n_win), [], n_win, fs); % control data(before PARRM)
[pxx_parrm, freqs] = pwelch(PARRM_Filtered, hamming(n_win), [], n_win, fs); % PARRMed data
%%
fig3 = figure;
fig3.Position(3:4) = [500 330];

plot(freqs, log10(pxx_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
plot(freqs, log10(pxx_parrm),'r','DisplayName','PARRMed');hold off
legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)

% xlim([0 13000])
% ylim([-15 -5])

xlim([0 1200]) % closed up
ylim([-15 -5]) % closed up

fig3_title = ['PARRMed data - pwelch'];
sgtitle(fig3_title, 'FontSize', 11);
fig3;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Definition of Visualization function    %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function PARRM_Filtered = Filtering(stim_data, fs, Period, winTime, skipTime, perDist) % perDist = 2 * D_period [samples]
    winSize = fs*winTime;                                                                       % N_bins [samples]
    skipSize = floor(fs*skipTime);                                                              % N_skip [samples]
    winDir = "both";
    PARRM = PeriodicFilter(Period,winSize,perDist,skipSize,winDir); % T: 188.0000, N_bins: 122070, N_skip: 12207, D_period: 0.5 [samples]
    a = filter2(PARRM.', stim_data, 'same');
    c = (1-filter2(PARRM.', ones(size(stim_data)), 'same'));
    PARRM_Filtered = (a - stim_data)./c + stim_data;
end