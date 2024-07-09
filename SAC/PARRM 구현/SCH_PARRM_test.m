clc
clear
close all

%% Loading Data
before = load('C:\Users\ryuda\OneDrive - dgist.ac.kr\바탕 화면\-\DGIST\UGRP\data\PD_rest(noStim)_ch12-13.mat'); % before stimulation
during = load('C:\Users\ryuda\OneDrive - dgist.ac.kr\바탕 화면\-\DGIST\UGRP\data\PD_OpenLoop_ch12-13.mat'); % during stimulation

%% info
num_ch = size(before.raw, 1);
stim_ch = 11;
target_ch = 12;

fs = 24414; % Sampling rate of simulated data
stimRate = 130; 

during_preproc = during.raw - mean(during.raw, 2); % data during stimulation with DC removed
stim_data = during_preproc(target_ch, :)';

%% control data(before stimualtion)
ctrl = before.raw(target_ch, :)'; % data before stimulation on target channel

%% PARRM
addpath ('C:\Users\ryuda\OneDrive - dgist.ac.kr\바탕 화면\-\DGIST\UGRP\PARRM-master\')

%% Grid Search - Finding True Period T
guessPeriod=fs*1/stimRate; % Starting point for period grid search                              % delta: 187.8000 samples
gpu_raw = gpuArray(stim_data);
Period = FindPeriodLFP(gpu_raw,[1,length(gpu_raw)-1],guessPeriod);                              % T: 188.0000 samples

%% Setting appropriate parameters
sepChirp = fs; % Separation between subequent chirps in samples

winTime = 5;                                                                                    % win time: 5s
winSize=fs*winTime; % Width of the window in sample space for PARRM filter                      % N_bins: 122070 samples

skipTime = winTime / 10;                                                                        % skip time: 0.5s
skipSize=floor(fs*skipTime); % Number of samples to ignore in each window in sample space       % N_skip: 12207 samples

perDist = 1; % Window in period space for which samples will be averaged                      % D_period: 0.5 samples

winDir = "both"; % Filter using samples from the past and future
num_Dir = numel(winDir);

%% Filtering using PARRM filter
PARRM = PeriodicFilter(Period,winSize,perDist,skipSize,winDir);                                 % T: 188.0000, N_bins: 122070, N_skip: 12207, D_period: 0.5 [samples]
a = filter2(PARRM.', stim_data, 'same');
c = (1-filter2(PARRM.', ones(size(stim_data)), 'same'));

% Filter using the linear filter and remove edge effects
PARMM_Filtered = (a - stim_data)./c + stim_data;

%% Visualization of PARRM filter
[f, x] = PeriodicFilter(Period,winSize,perDist,skipSize,winDir);

s = -winSize:winSize;
nT = 0:Period:winSize;
nT = [-nT, nT];
% figure
% hold on
% stem(s, ones(numel(s), 1), 'k')
% stem(s, f, 'g')
% scatter(nT, zeros(numel(nT), 1), 'r*')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Visualization of PARRMed data   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% info - time domain
% time = 0:1/fs:((size(ctrl, 1))-1)/fs;
% start_time = 10; % [sec]
% end_time = 30; % [sec]
% start_pts = start_time*fs+1;
% end_pts = end_time*fs;
% t = time(start_pts:end_pts);
% 
% %% info - frequency domain
% t_win = 1; % [sec]
% n_win = t_win*fs;

% %% %%%%%%   Visualization of PARRMed data with Time-voltage series  %%%%%%
% 
% % Slicing the timescale of data in time domain figure
% non_Filtered = stim_data(start_pts:end_pts);
% Filtered = PARRM_Filtered(start_pts:end_pts);
% 
% %%
% fig1 = figure;
% hold on
% % plot(time(start_pts:end_pts), ctrl(start_pts:end_pts),'g', 'DisplayName','Before Stim.');                     % raw data before stimulation on target channel
% plot(t, non_Filtered,'r','DisplayName','During Stim.');                                                         % raw data during stimulation on target channel
% plot(t, Filtered,'b','DisplayName','PARRMed');                                                                  % PARRMed data during stimulation on target channel
% legend('-dynamiclegend')
% xlabel('Time [sec]'); ylabel('Voltage [V]')
% set(gca, 'FontSize', 15)
% hold off
% 
% % print(fig1, 'C:\Users\stell\Desktop\SCH\학부연\2학년 겨울학기 인턴\인턴 최종 발표\Time_domain', '-dpng') % Saving figure by as a PNG file
% 
% % PARRM으로 artifact가 잘 제거된 구간을 캡쳐
% xlim([22 22.1])
% ylim([-0.8*10^(-3) 1.8*10^(-3)])
% print(fig1, 'C:\Users\stell\Desktop\SCH\학부연\2학년 겨울학기 인턴\인턴 최종 발표\Time_domain_closedup_successpart', '-dpng')
% 
% % PARRM으로 artifact가 잘 제거되지 않는 구간을 캡쳐
% xlim([22.4 22.5])
% ylim([-0.8*10^(-3) 1.8*10^(-3)])
% print(fig1, 'C:\Users\stell\Desktop\SCH\학부연\2학년 겨울학기 인턴\인턴 최종 발표\Time_domain_closedup_failpart', '-dpng')
% 
% %% %%%%%%   Visualization of PARRMed data with Power spectral density  %%%%%%
% 
% % [pxx_before, freqs_before] = pwelch(ctrl, hamming(n_win), [], n_win, fs);
% [pxx_during, freqs_during] = pwelch(stim_data, hamming(n_win), [], n_win, fs);
% [pxx_parrmed, freqs] = pwelch(PARRMM_Filtered, hamming(n_win), [], n_win, fs);
% 
% %%
% fig2 = figure;
% 
% hold on
% % plot(freqs, log10(pxx_before),'g', 'DisplayName','Before Stim.')                                              % raw data before stimulation on target channel
% plot(freqs, log10(pxx_during),'r','DisplayName','During Stim.')                                                 % raw data during stimulation on target channel
% plot(freqs, log10(pxx_parrmed),'b','DisplayName','PARRMed')                                                     % PARRMed data during stimulation on target channel
% legend('-dynamiclegend')
% xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]')
% set(gca, 'FontSize', 15)
% 
% xlim([0 13000])
% print(fig2, 'C:\Users\stell\Desktop\SCH\학부연\2학년 겨울학기 인턴\인턴 최종 발표\Frequency_domain_closedup', '-dpng') % Saving figure by as a PNG file


%% %%%%%%   Finding appropriate parameters  %%%%%%
clc
close all
path_savefig = 'C:\Users\stell\Desktop\SCH\학부연\2학년 겨울학기 인턴\인턴 최종 발표\세로버전\';

%% info - time domain
time = 0:1/fs:((size(ctrl, 1))-1)/fs;
start_time = 10; % [sec]
end_time = 30; % [sec]
start_pts = start_time*fs+1;
end_pts = end_time*fs;
t = time(start_pts:end_pts);

%% info - frequency domain
t_win = 1; % [sec]
n_win = t_win*fs;

%% control data(before PARRM)
non_Filtered = stim_data(start_pts:end_pts); % scaled time domain
[pxx_during, freqs] = pwelch((stim_data), hamming(n_win), [], n_win, fs); % frequency domain

%% Visualization of optimized PARRMed data with the best parameters(winTime, skipTime, perDist)

opt_winTime = 0.5; %10, 5, 1, 0.5, [0.4], 0.3, [0.2] <0.1>, [0.05],[0.04],[0.38],[0.375],[0.374989752],[0.037498975150420426477595>, <0.374989751504204264775932>, <0.3745>, <0.374>, <0.37>, <0.35>, <0.03>, <0.01>
opt_skipTime = opt_winTime / 10;
opt_perDist = 0.5;

PARRM_Filtered = Filtering(stim_data, fs, Period, opt_winTime, opt_skipTime, opt_perDist); % time domain

fig_optimized = figure('Position', [10 440 500 800]);
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered, t_win, fs, pxx_during, ['Using the best parameters' newline '( winTime = ', num2str(opt_winTime), ' [sec], skipTime = ', num2str(opt_skipTime), ' [sec], perDist = ', num2str(opt_perDist), ' [pts] )'])
print(fig_optimized, [path_savefig,'optimized with the best parameters (artifact removal이 잘 수행되지 않은 구간)'], '-dpng')






%% %%%%%%   Adjustments of parameters  %%%%%%

%% Adjustment of winTime

small_winTime = 0.04;
large_winTime = 100;
modified_small_skipTime = small_winTime / 10;
modified_large_skipTime = large_winTime / 10;

PARRM_Filtered1 = Filtering(stim_data, fs, Period, small_winTime, modified_small_skipTime, opt_perDist);     % too small winTime를 사용한 PARRM fitering 수행

fig_small_winTime = figure('Position', [1010 1040 500 800]); % figure
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered1, t_win, fs, pxx_during, ['Using too small winTime ' newline '( winTime = ', num2str(small_winTime), ' [sec] )']) % subplot 4개
print(fig_small_winTime, [path_savefig,'too small winTime (closed-up)'], '-dpng')

PARRM_Filtered2 = Filtering(stim_data, fs, Period, large_winTime, modified_large_skipTime, opt_perDist);     % too large winTime를 사용한 PARRM fitering 수행

fig_large_winTime = figure('Position', [1010 740 500 800]); % figure
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered2, t_win, fs, pxx_during, ['Using too large winTime ' newline '( winTime = ', num2str(large_winTime), ' [sec] )'])
print(fig_large_winTime, [path_savefig,'too large winTime (closed-up)'], '-dpng')

%% Adjustment of skipTime

small_skipTime = opt_winTime / 10^30;
large_skipTime = opt_winTime / 1.1;

PARRM_Filtered1 = Filtering(stim_data, fs, Period, opt_winTime, small_skipTime, opt_perDist);     % too small skipTime를 사용한 PARRM fitering

fig_small_skipTime = figure('Position', [10 440 500 800]); % figure
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered1, t_win, fs, pxx_during, ['Using too small skipTime ' newline '( skipTime = ', num2str(small_skipTime), ' [sec] )']) % subplot 4개
print(fig_small_skipTime, [path_savefig,'too small skipTime (closed-up)'], '-dpng')

PARRM_Filtered2 = Filtering(stim_data, fs, Period, opt_winTime, large_skipTime, opt_perDist);     % too large skipTime를 사용한 PARRM fitering

fig_large_skipTime = figure('Position', [10 40 500 800]); % figure
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered2, t_win, fs, pxx_during, ['Using too large skipTime ' newline '( skipTime = ', num2str(large_skipTime), ' [sec] )'])
print(fig_large_skipTime, [path_savefig,'too large skipTime (closed-up)'], '-dpng')

%% Adjustment of perDist

small_perDist = 0.1;
large_perDist = 10;

PARRM_Filtered1 = Filtering(stim_data, fs, Period, opt_winTime, opt_skipTime, small_perDist);     % too small perDist를 사용한 PARRM fitering

fig_small_perDist = figure('Position', [1010 440 500 800]); % figure
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered1, t_win, fs, pxx_during, ['Using too small perDist ' newline '( perDist = ', num2str(small_perDist), ' [pts] )']) % subplot 4개
print(fig_small_perDist, [path_savefig,'too small perDist (closed-up)'], '-dpng')

PARRM_Filtered2 = Filtering(stim_data, fs, Period, opt_winTime, opt_skipTime, large_perDist);     % too large perDist를 사용한 PARRM fitering

fig_large_perDist = figure('Position', [1010 40 500 800]); % figure
VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered2, t_win, fs, pxx_during, ['Using too large perDist ' newline '( perDist = ', num2str(large_perDist), ' [pts] )'])
print(fig_large_perDist, [path_savefig,'too large perDist (closed-up)'], '-dpng')

%% 이유를 생각해보기**

%% 인턴 최종 발표 자료 만들기
% 중간 발표까지의 내용 요약
% PARRM 알고리즘 설명
% 적용 결과
% 한계점
% 계획




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Definition of Visualization function  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PARRM_Filtered = Filtering(stim_data, fs, Period, winTime, skipTime, perDist)
    winSize = fs*winTime;
    skipSize = floor(fs*skipTime);     
    winDir = "both";
    PARRM = PeriodicFilter(Period,winSize,perDist,skipSize,winDir);
    a = filter2(PARRM.', stim_data, 'same');
    c = (1-filter2(PARRM.', ones(size(stim_data)), 'same'));
    PARRM_Filtered = (a - stim_data)./c + stim_data;
end


function VisualizeFig(t, start_pts, end_pts, non_Filtered, PARRM_Filtered, t_win, fs, pxx_during, titlestr)

    % time domain
    Filtered = PARRM_Filtered(start_pts:end_pts);

    % frequency domain
    n_win = fs*t_win;
    [pxx_parrm, freqs] = pwelch(PARRM_Filtered, hamming(n_win), [], n_win, fs); % too small parameter

    % too small parameter

    subplot(2,1,1) % time domain
    plot(t, non_Filtered,'Color','#4DBEEE','DisplayName','During Stim.');hold on
    plot(t, Filtered,'r','DisplayName','PARRMed');hold off
    legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 10)
    
%     ylim([-0.1 0.05])

%     xlim([22.4 22.5]) % closed up
    xlim([19.11 19.21]) % closed up
    ylim([-0.8*10^(-3) 1.8*10^(-3)]) % closed up
    
    subplot(2,1,2) % frequency domain
    plot(freqs, log10(pxx_during),'Color','#4DBEEE','DisplayName','During Stim.');hold on
    plot(freqs, log10(pxx_parrm),'r','DisplayName','PARRMed');hold off
    legend('-dynamiclegend'); xlabel('Frequency [Hz]'); ylabel('Power [log_{10}(V^2)]'); set(gca, 'FontSize', 10)

%     xlim([0 13000])
%     ylim([-15 -5])

    xlim([0 1200]) % closed up
    ylim([-15 -5]) % closed up

    sgtitle(titlestr, 'FontSize', 11);

end
