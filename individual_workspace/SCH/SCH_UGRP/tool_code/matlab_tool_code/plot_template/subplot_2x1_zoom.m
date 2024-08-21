fig = figure('Position', [500 500 500 800]); % figure 세로
figtitle = '그래프 제목'

subplot(2,1,1)
plot(t, non_Filtered,'Color','#4DBEEE','DisplayName','name1');hold on
plot(t, Filtered,'r','DisplayName','name2');hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 10)

xlim([-5 5])
ylim([-5 5])

subplot(2,1,2) % closed up
plot(t, non_Filtered,'Color','#4DBEEE','DisplayName','name1');hold on
plot(t, Filtered),'r','DisplayName','name2');hold off
legend('-dynamiclegend'); xlabel('Time [sec]'); ylabel('Voltage [V]'); set(gca, 'FontSize', 10)

xlim([-1 1]) % closed up
ylim([-1 1]) % closed up

sgtitle(figtitle, 'FontSize', 11);