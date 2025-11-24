# -*- coding: utf-8 -*-
"""
功能：
1. 遍历股票子目录，找到 daily_fitted_sentiment.csv；
2. 将文件统一复制到指定输出目录；
3. 可选：保留股票代码信息到文件名。
"""

import os
import shutil

# ========== 配置 ==========
INPUT_ROOT = "../../mid_result/hs300_data/stocks_separate_data"  # 每个子目录是股票代码
INPUT_FILENAME = "daily_fitted_sentiment.csv"
OUTPUT_ROOT = "../../mid_result/hs300_data/single_stock_fitted_sentiments"  # 输出目录

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ========== 遍历子目录 ==========
for stock_dir in os.listdir(INPUT_ROOT):
    full_dir = os.path.join(INPUT_ROOT, stock_dir)
    if not os.path.isdir(full_dir):
        continue

    input_file = os.path.join(full_dir, INPUT_FILENAME)
    if not os.path.exists(input_file):
        print(f"⚠️ 文件不存在: {input_file}")
        continue

    # 输出文件名加上股票代码
    output_file = os.path.join(OUTPUT_ROOT, f"{stock_dir}_{INPUT_FILENAME}")

    shutil.copy2(input_file, output_file)
    print(f"✅ 已复制: {input_file} → {output_file}")

print("\n🎯 所有文件已提取完成！")
