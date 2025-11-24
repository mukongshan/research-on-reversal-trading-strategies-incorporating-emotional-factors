# -*- coding: utf-8 -*-
"""
批量计算加权情绪贡献
输入: 含有 ["基础分数", "阅读量", "评论"] 列的 CSV
目录结构:
    主目录/
        股票代码1/
            titles_with_score.csv
        股票代码2/
            titles_with_score.csv
输出: 每个子目录下新增 titles_with_weighted_score.csv
"""

import pandas as pd
import numpy as np
import os

# ========== 配置 ==========
INPUT_ROOT = "../../../../mid_result/hs300_data/stocks_separate_data"
INPUT_FILENAME = "titles_with_score.csv"          # 子目录中要处理的文件名
OUTPUT_FILENAME = "titles_with_weighted_score.csv" # 子目录中输出的文件名
ALPHA = 0.5
BETA = 0.5

# ========== 函数 ==========
def compute_weighted_emotion(input_path, output_path, alpha=ALPHA, beta=BETA):
    df = pd.read_csv(input_path)
    df["权重"] = alpha * np.log1p(df["阅读量"])**2 + beta * np.log1p(df["评论"])**2
    df["加权情绪贡献"] = df["基础分数"] * df["权重"]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已处理: {input_path} → {output_path}")
    return df

# ========== 主程序 ==========
if __name__ == "__main__":
    for stock_dir in os.listdir(INPUT_ROOT):
        full_dir = os.path.join(INPUT_ROOT, stock_dir)
        if not os.path.isdir(full_dir):
            continue  # 跳过非目录

        input_path = os.path.join(full_dir, INPUT_FILENAME)
        output_path = os.path.join(full_dir, OUTPUT_FILENAME)

        if os.path.exists(input_path):
            compute_weighted_emotion(input_path, output_path)
        else:
            print(f"⚠️ 文件不存在: {input_path}")

    print("\n🎯 所有子目录已处理完成！")
