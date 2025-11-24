%% 舆情数据处理
filename1 = 'D:\All_of_mine\大学\比赛\da_chuang\src\data_2\zssh000300\按天整合_舆情总分.xlsx';
data1 = readtable(filename1);
dates1 = datetime(data1.date, 'InputFormat', 'yyyy-MM-dd');
scores = data1.final_score;
x1 = datenum(dates1);
yy1 = smooth(x1, scores, 0.05, 'loess');  % 平滑拟合舆情得分

%% 指数数据处理
filename2 = 'D:\All_of_mine\大学\比赛\da_chuang\res\data\index_data.xlsx';
data2 = readtable(filename2);
dates2 = datetime(data2.date, 'InputFormat', 'yyyy/MM/dd');
close_price = data2.close;
x2 = datenum(dates2);
yy2 = smooth(x2, close_price, 0.05, 'loess');  % 平滑拟合指数收盘价

%% 绘图（双纵轴）
figure;
yyaxis left
plot(dates1, yy1, 'b-', 'LineWidth', 2);
ylabel('舆情得分');
ylim([-0.6, 0.6]);  % 固定刻度范围
yticks(-1:0.2:1);

yyaxis right
plot(dates2, yy2, 'r-', 'LineWidth', 2);
ylabel('指数收盘价');

xlabel('日期');
legend('舆情得分', '指数收盘价', 'Location', 'best');
grid off;
