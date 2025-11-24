% 读取 Excel 文件
filename = 'D:\All_of_mine\大学\比赛\da_chuang\src\data_2\zssh000300\按天整合_舆情总分.xlsx';
data = readtable(filename);

% 转换日期格式
dates = datetime(data.date, 'InputFormat', 'yyyy-MM-dd');

% 提取得分
scores = data.final_score;

% 将日期转换为数字以用于拟合
x = datenum(dates);  % 日期转为数值（便于拟合）
y = scores;

% 绘制散点图
figure;
scatter(dates, scores, 20, 'filled');
hold on;

% 进行平滑拟合（使用低ess smoothing，平滑度 0.2 可调）
yy = smooth(x, y, 0.05, 'loess');  % 或 'rloess' 更稳健

% 绘制拟合曲线
plot(dates, yy, 'r-', 'LineWidth', 2);

% 图形美化
xlabel('日期');
ylabel('Final Score');
legend('舆情总量', '拟合曲线');
grid off;
