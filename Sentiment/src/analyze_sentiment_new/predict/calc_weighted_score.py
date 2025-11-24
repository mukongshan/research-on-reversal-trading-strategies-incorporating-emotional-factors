# -*- coding: utf-8 -*-
"""
根据公式计算加权情绪贡献
输入: 含有 ["基础分数", "阅读量", "评论"] 列的 CSV
输出: 新增 [ "权重", "加权情绪贡献"] 列
"""

import pandas as pd
import numpy as np

def compute_weighted_emotion(input_path, output_path, alpha=0.5, beta=0.5):
    # 读取数据
    df = pd.read_csv(input_path)

    # 计算综合权重
    df["权重"] = alpha * np.log1p(df["阅读量"])**2 + beta * np.log1p(df["评论"])**2

    # 计算加权情绪贡献
    df["加权情绪贡献"] = df["基础分数"] * df["权重"]

    # 保存结果
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到 {output_path}")

    return df


if __name__ == "__main__":
    compute_weighted_emotion(
        input_path="../../../mid_result/hs300_data/hs300_stocks_forum/titles_with_score.csv",   # 输入文件
        output_path="../../../mid_result/hs300_data/hs300_stocks_forum/titles_with_weighted_score.csv",            # 输出文件
        alpha=0.5,  # 可调整
        beta=0.5    # 可调整
    )
