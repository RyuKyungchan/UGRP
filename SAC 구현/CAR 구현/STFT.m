%%
clc
clear
close all

before=load("data\PD_rest(noStim)_ch12-13.mat");
during=load("data\PD_OpenLoop_ch12-13.mat");

before_data = before.raw;
during_data = during.raw;
mean_data = mean(during_data);

CAR_filtered_data = zeros(16, 7324200);
for i=1:16 
   CAR_filtered_data(i, :) = during_data(i, :) - mean(during_data); 
end

CMR_filtered_data = zeros(16, 7324200);
for i=1:16 
   CMR_filtered_data(i, :) = during_data(i, :) - median(during_data); 
end

%% 푸리에 변환 수행
fs = 24414;  % 샘플링 주파수 (예: 1000 Hz, 데이터에 따라 조절)
N = size(CAR_filtered_data, 2);  % 데이터 포인트 수
frequencies = linspace(-fs/2, fs/2, N);  % 주파수 범위 설정

raw_fft_result = fftshift(fft(before_data, [], 2));
CAR_fft_result = fftshift(fft(CAR_filtered_data, [], 2));  % 2차원 푸리에 변환
CMR_fft_result = fftshift(fft(CMR_filtered_data, [], 2));

% 양수 부분의 주파수와 푸리에 변환 결과 가져오기
positive_frequencies = frequencies(N/2+1:end);
positive_raw_fft_result = 2*raw_fft_result(:, N/2+1:end);
positive_CAR_fft_result = 2*CAR_fft_result(:, N/2+1:end);
positive_CMR_fft_result = 2*CMR_fft_result(:, N/2+1:end);



%% Short-Time Fourier Transform (STFT) 수행
fs = 24414;  % 샘플링 주파수 (예: 1000 Hz, 데이터에 따라 조절)
window_length = fs; % 윈도우 길이 설정
overlap_length = window_length / 2; % 오버랩 길이 설정


%%
figure (1);

spectrogram(CAR_filtered_data(1, :), window_length, overlap_length, [], fs, 'yaxis');
title('CAR Filtered Data - STFT', FontSize=16);
xlabel('Time (minute)', FontSize=16)
ylabel('Frequency (kHz)', FontSize=16)
%set(gca, 'YScale', 'log');
colorbar;

figure (2);

spectrogram(CMR_filtered_data(1, :), window_length, overlap_length, [], fs, 'yaxis');
title('CMR Filtered Data - STFT', FontSize=16);
xlabel('Time (minute)', FontSize=16)
ylabel('Frequency (kHz)', FontSize=16)
%set(gca, 'YScale', 'log');
colorbar;

figure (3);

spectrogram(during_data(1, :), window_length, overlap_length, [], fs, 'yaxis');
title('Raw Data - STFT', FontSize=16);
xlabel('Time (minute)', FontSize=16)
ylabel('Frequency (kHz)', FontSize=16)
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