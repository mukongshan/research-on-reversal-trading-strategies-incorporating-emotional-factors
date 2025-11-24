import pandas as pd
import numpy as np

# === 修改这里：你的原始 Excel 文件路径 ===
input_path = r'D:\All_of_mine\大学\项目和比赛\da_chuang\src\data_2\zssh000300\股吧评论分词结果_带分数.xlsx'  # 例如：'情绪数据2025-04-23.xlsx'
output_path = r'D:\All_of_mine\大学\项目和比赛\da_chuang\src\data_2\zssh000300\分词得分（影响力加权）.xlsx'

# 读取 Excel（默认读取第一个 sheet）
df = pd.read_excel(input_path)

# === 计算过程 ===

# 1. 阅读量因子 = ln(1 + 阅读量)^2
df['阅读量因子'] = np.log1p(df['阅读量']) ** 2

# 2. 评论因子 = ln(1 + 评论数)^2
df['评论因子'] = np.log1p(df['评论数']) ** 2

# 3. 影响力权重 = 0.6 * 阅读量因子 + 评论因子
df['影响力权重'] = 0.6 * df['阅读量因子'] + df['评论因子']

# 4. 加权分数 = 基础分数 * 影响力权重
df['加权分数'] = df['基础分数'] * df['影响力权重']

# 5. 最终得分（加权平均，归一化）
# 加权平均归一化（前面那种）
avg_score = df['加权分数'].sum() / df['影响力权重'].sum()
df['加权平均归一化得分'] = avg_score

# 最大值归一化（你说的这个）
max_abs = df['加权分数'].abs().max()
df['最大绝对值归一化得分'] = df['加权分数'] / max_abs if max_abs != 0 else 0.0


# === 保存结果 ===
df.to_excel(output_path, index=False)
print(f"✅ 已保存至：{output_path}")
