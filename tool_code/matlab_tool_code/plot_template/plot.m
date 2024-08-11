path_savefig = '그래프 저장 경로 이름'

fig = figure;
fig.Position(3:4) = [500 330];
figtitle = '그래프 제목'

hold on
plot(t, plotA,'g', 'DisplayName','name1'); 
plot(t, plotB,'r','DisplayName','name2');
plot(t, plotC,'b','DisplayName','name3'); 
legend('-dynamiclegend')

xlabel('Time [sec]'); ylabel('Voltage [V]')
set(gca, 'FontSize', 15)
hold off

xlim([-5 5])
ylim([-5 5])

% xlim([-1 1]) % closed up
% ylim([-1 1]) % closed up

sgtitle(figtitle, 'FontSize', 11);

print(fig, path_savefig, '-dpng')
% print(fig, [path_savefig,'파일 이름'], '-dpng') % 파일 이름 설정