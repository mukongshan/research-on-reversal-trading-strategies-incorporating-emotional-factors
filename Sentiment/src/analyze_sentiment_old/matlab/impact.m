% 分布图绘制脚本
% 假设文件名为 'data.xlsx'，并且包含一列名为 '阅读量'

% 读取 Excel 文件
filename = 'D:\All_of_mine\大学\项目和比赛\da_chuang\src\data_2\zssh000300\stock_info.xlsx';  % 修改为你自己的文件名
data = readtable(filename);

% 查看列名（可选）
% disp(data.Properties.VariableNames);

% 提取“阅读量”列数据（确保列名正确）
read_counts = data.comments;

% 定义每 1000 为一组的区间
binEdges = 0:1:max(read_counts)+1;

% 统计每个区间的数量
[counts, edges] = histcounts(read_counts, binEdges);

% 计算每个区间的中心点，用于横轴
binCenters = edges(1:end-1) + diff(edges)/2;

% 绘制分布图
figure;
bar(binCenters, counts, 'hist');

% 图形美化
xlabel('评论量');
ylabel('帖子数量');
title('评论量分布图');
grid on;
