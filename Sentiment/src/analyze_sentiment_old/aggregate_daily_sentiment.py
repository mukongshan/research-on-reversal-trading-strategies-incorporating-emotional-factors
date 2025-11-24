import pandas as pd
import numpy as np

# === 修改为你的输入路径 ===
input_path = r'D:\All_of_mine\大学\比赛\da_chuang\src\data_2\zssh000300\分词得分（影响力加权）.xlsx'
output_path = r'D:\All_of_mine\大学\比赛\da_chuang\src\data_2\zssh000300\按天整合_舆情总分.xlsx'

# 读取数据
df = pd.read_excel(input_path)

# 1. 时间转换为字符串格式的日期
df['日期'] = pd.to_datetime(df['时间']).dt.strftime('%Y-%m-%d')

# 2. 取每个标题的“最终分数”中绝对值最大者（保留正负号）
# 自定义函数：返回绝对值最大行的原始得分
def max_abs_score(group):
    return group.loc[group['最终分数'].abs().idxmax(), '最终分数']

# 按日期 + 标题 分组，取每组中绝对值最大的“最终分数”
title_scores = df.groupby(['日期', '标题']).apply(max_abs_score).reset_index(name='标题得分')

# 3. 每天的“舆情总分” = 所有标题得分之和
daily_sum = title_scores.groupby('日期')['标题得分'].sum().reset_index(name='舆情总分')

# 4. 最大绝对值归一化
max_abs = daily_sum['舆情总分'].abs().max()
daily_sum['归一化舆情总分'] = daily_sum['舆情总分'] / max_abs if max_abs != 0 else 0.0

# 5. 保存
daily_sum.to_excel(output_path, index=False)
print(f"✅ 舆情分析结果已保存至：{output_path}")