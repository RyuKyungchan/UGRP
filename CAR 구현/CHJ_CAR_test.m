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


%%

% 푸리에 변환 수행
fs = 24414;  % 샘플링 주파수 (예: 1000 Hz, 데이터에 따라 조절)
N = size(filtered_data, 2);  % 데이터 포인트 수
frequencies = linspace(-fs/2, fs/2, N);  % 주파수 범위 설정

fft_result = fftshift(fft(filtered_data, [], 2));  % 2차원 푸리에 변환

% 양수 부분의 주파수와 푸리에 변환 결과 가져오기
positive_frequencies = frequencies(N/2+1:end);
positive_fft_result = 2*fft_result(:, N/2+1:end);

%%
% 푸리에 스펙트럼 플로팅
figure;

% subplot(1, 2, 1)
% plot(before_data(1, :), 'r');
% hold on
% plot(during_data(1, :), 'g');
% plot(filtered_data(1, :), 'b');
% hold off
% 
% title('CAR 결과');
% xlabel('시간');
% ylabel('신호 세기');


% subplot(1, 2, 2)
plot(positive_frequencies, abs(positive_fft_result(1, :)), 'b');
xlim([0 fs/2])
title('Filtered Data - 푸리에 스펙트럼 (Single-Sided)', FontSize=16);
xlabel('주파수 (Hz)', FontSize=16);
ylabel('스펙트럼 세기',FontSize=16);
set(gca, 'YScale', 'log');