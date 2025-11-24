import pandas as pd
import os

# 读取文件（支持 xlsx / csv）
file_path = r"../../mid_result/hs300_data/stocks_split_titles_data/part_1.csv"  # 替换为你的文件路径
if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
elif file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("只支持 .xlsx 或 .csv 文件")

# 随机抽取指定行数（不足则取全部）
words_num = 200   # 想要抽取的行数
sample_size = min(words_num, len(df))
df_sample = df.sample(n=sample_size, random_state=42)  # 设置 random_state 保证可复现

# 生成输出文件名（K 为单位，自动格式化）
k_size = sample_size
output_path = f"random_{k_size}_rows.csv"

# 保存为 CSV
df_sample.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ 已随机抽取 {sample_size} 行，并保存到 {output_path}")
