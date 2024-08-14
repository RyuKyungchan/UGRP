%%
clc
clear
close all

before=load("data\PD_rest(noStim)_ch12-13.mat");
during=load("data\PD_OpenLoop_ch12-13.mat");

before_data = before.raw;
during_data = during.raw;
mean_data = mean(during_data);

filtered_data = zeros(16, 7324200);
for i=1:16 
   filtered_data(i, :) = during_data(i, :) - mean(during_data); 
end

%% 푸리에 변환 수행
fs = 24414;  % 샘플링 주파수 (예: 1000 Hz, 데이터에 따라 조절)
N = size(filtered_data, 2);  % 데이터 포인트 수
frequencies = linspace(-fs/2, fs/2, N);  % 주파수 범위 설정

fft_result = fftshift(fft(filtered_data, [], 2));  % 2차원 푸리에 변환

% 양수 부분의 주파수와 푸리에 변환 결과 가져오기
positive_frequencies = frequencies(N/2+1:end);
positive_fft_result = 2*fft_result(:, N/2+1:end);


%% Short-Time Fourier Transform (STFT) 수행
fs = 24414;  % 샘플링 주파수 (예: 1000 Hz, 데이터에 따라 조절)
window_length = fs; % 윈도우 길이 설정
overlap_length = window_length / 2; % 오버랩 길이 설정

% STFT plot
% subplot(1, 2, 2)
spectrogram(filtered_data(1, :), window_length, overlap_length, [], fs, 'yaxis');
title('Filtered Data - STFT', FontSize=20);
xlabel('시간 (분)', FontSize=20)
ylabel('주파수 (kHz)', FontSize=20)
%set(gca, 'YScale', 'log');
colorbar;



% [sx, fx, tx] = spectrogram(filtered_data(1, :), window_length, overlap_length, [], fs);
% 
% % 스펙트로그램을 로그 스케일로 변환
% log_sx = 10*log10(abs(sx));
% 
% % 시각화
% figure;
% waterplot(sx, fx, tx);
% 
% function waterplot(s,f,t)
%     % Waterfall plot of spectrogram
%     waterfall(t, f, s)
%     set(gca, 'YScale', 'log'); % y축 로그 스케일로 설정
%     view(0, 90);
%     xlabel('시간 (s)', FontSize=16);
%     ylabel('주파수 (Hz)', FontSize=16);
%     zlabel('파워 (dB)', FontSize=16);
%     title('Filtered Data - Spectrogram (Log Scale)', FontSize=20);
% end