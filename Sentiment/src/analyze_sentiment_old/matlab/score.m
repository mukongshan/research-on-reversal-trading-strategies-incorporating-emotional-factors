% 假设你的文件名仍为 'data.xlsx'

% 读取表格数据
filename = 'D:\All_of_mine\大学\比赛\da_chuang\src\data_2\zssh000300\分词得分（影响力加权）.xlsx';
data = readtable(filename);

% 检查列名（可选）
% disp(data.Properties.VariableNames);

% 提取日期和得分列（请根据你的实际列名确认）
dates = datetime(data.date, 'InputFormat', 'yyyy-MM-dd HH:mm');

% 提取分数
scores = data.final_score;

% 绘制散点图（实心圆点，无连线）
figure;
scatter(dates, scores, 20, 'filled');  % 20 是点的大小

% 图形美化
xlabel('日期');
ylabel('Final Score');
title('Final Score 随时间变化的散点图');
grid on;
