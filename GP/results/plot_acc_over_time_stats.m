x = linspace(0, 100, 18);

d = load('acc_over_time_stats.mat');
d = d.res;

d(1,:) = 100 - 100*d(1, :)/297;
d(3,:) = 100 - 100*d(3, :)/297;
d(2,:) = 100*d(2, :)/297;
d(4,:) = 100*d(4, :)/297;
y1_top = d(1,:) + d(2, :);
y1_bot = d(1,:) - d(2, :);
y2_top = d(3,:) + d(4, :);
y2_bot = d(3,:) - d(4, :);

fill([x, flip(x)], [y1_top, fliplr(y1_bot)], 'b', 'LineStyle', 'none')
hold on;
fill([x, flip(x)], [y2_top, fliplr(y2_bot)], 'r', 'LineStyle', 'none')
plot(x, y1_top, 'b', x, y1_bot, 'b', x, d(1, :), 'b--')  
plot(x, y2_top, 'r', x, y2_bot, 'r', x, d(3, :), 'r--')  

ylim([0, 100])
alpha(.25)