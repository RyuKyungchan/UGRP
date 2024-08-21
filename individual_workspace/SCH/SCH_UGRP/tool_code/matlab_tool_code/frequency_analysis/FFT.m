clc
clear
close all

%% Loading Data
before = load('C:\Users\stell\Desktop\SCH\UGRP\data\PD_rest(noStim)_ch12-13.mat'); % before stimulation
during = load('C:\Users\stell\Desktop\SCH\UGRP\data\PD_OpenLoop_ch12-13.mat'); % during stimulation

%% info
num_ch = size(before.raw, 1);
stim_ch = 11;
target_ch = 12;

data = during.raw(target_ch,:);

fs = 24414;
Ts = 1/24414;
n = length(data);
t = 0:Ts:60;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   FFT  %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FFT
f_data = fft(data); % FFT 계산

% fftshift를 통해 주파수 스펙트럼을 중앙으로 시프트
f_data_double_side = fftshift(f_data);
frequencies_double_side = (-n/2:n/2-1)*(fs/n);

% fftshift된 주파수 스펙트럼을 다시 단일 측정 스펙트럼으로 변환
f_data_single_side = f_data_double_side(n/2:end); % f>=0인 주파수 성분만 선택(dc 포함)
frequencies_single_side = (0:n/2)*(fs/n); % 주파수 벡터 생성

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Amplitude / Power  %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 진폭 계산
amplitude_single_side = abs(f_data_single_side);
amplitude_single_side(2:end) = 2*amplitude_single_side(2:end); % 단일 측정 스펙트럼

% Power 계산
power_single_side = amplitude_single_side.^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Plotting     %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Amplitude Plotting

% figure;
% plot(frequencies_single_side, log10(amplitude_single_side));
% title('Single-Sided Frequency Spectrum');
% xlabel('Frequency [Hz]');
% ylabel('Voltage [V]');
% grid on;

%% Power Plotting

figure;
plot(frequencies_single_side, log10(power_single_side));
title('Single-Sided Frequency Spectrum');
xlabel('Frequency [Hz]');
ylabel('Power [log_{10}(V^2)]');
grid on;