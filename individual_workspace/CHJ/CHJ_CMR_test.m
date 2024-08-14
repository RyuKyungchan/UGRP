clc
clear
close all

before=load("PD_rest(noStim)_ch12-13.mat");
during=load("PD_OpenLoop_ch12-13.mat");

before_data = before.raw;
during_data = during.raw;
median_data = median(during_data);

filtered_data = zeros(16, 7324200);
for i=1:16 
   filtered_data(i, :) = during_data(i, :) - median(during_data); 
end


plot(before_data(1, :), 'r');
hold on
plot(during_data(1, :), 'g');
plot(filtered_data(1, :), 'b');
hold off

title('CMR 필터링 적용 결과');
xlabel('시간');
ylabel('신호 세기');
