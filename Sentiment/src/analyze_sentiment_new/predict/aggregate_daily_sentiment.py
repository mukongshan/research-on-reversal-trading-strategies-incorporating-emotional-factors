# -*- coding: utf-8 -*-
"""
功能：
1. 读取帖子级数据；
2. 按日聚合计算“指数1_强度*热度”；
3. 对指数1进行归一化（范围[-1,1]，抗异常值）；
4. 批量输出日度情绪指标文件。
"""

import pandas as pd
import numpy as np

# ========== 📂 输入输出路径列表 ==========
input_paths = [
    "../../../mid_result/hs300_data/hs300_stock_forum/titles_with_weighted_score.csv",
    "../../../mid_result/hs300_data/single_stock/titles_with_weighted_score.csv"
]

output_paths = [
    "../../../mid_result/hs300_data/hs300_stock_forum/daily_sentiment_index.csv",
    "../../../mid_result/hs300_data/single_stock/daily_sentiment_index.csv"
]

# ========== 🧠 封装归一化函数 ==========
def normalize_series(series, method="tanh_zscore"):
    """
    对序列进行归一化（默认输出范围约为 [-1, 1]）。
    参数:
        series: pd.Series
        method: str，可选
            - "tanh_zscore": 先zscore标准化，再tanh压缩，抗异常值（推荐）
            - "robust": 按分位数裁剪后线性映射到[-1,1]
            - "minmax": 简单min-max映射到[-1,1]
        clip_percentile: robust模式下的分位数裁剪比例
    """
    s = series.copy().astype(float)

    if method == "tanh_zscore":
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0:
            return pd.Series(0, index=s.index)
        z = (s - mu) / sigma
        return np.tanh(0.5 * z)

    elif method == "robust":
        clip_percentile = 0.05
        q_low, q_high = s.quantile(clip_percentile), s.quantile(1 - clip_percentile)
        s_clipped = np.clip(s, q_low, q_high)
        return 2 * (s_clipped - q_low) / (q_high - q_low) - 1

    elif method == "minmax":
        min_val, max_val = s.min(), s.max()
        if max_val == min_val:
            return pd.Series(0, index=s.index)
        return 2 * (s - min_val) / (max_val - min_val) - 1

    else:
        raise ValueError(f"未知归一化方法: {method}")





# ========== 🚀 遍历处理 ==========
for file_path, output_path in zip(input_paths, output_paths):
    print(f"\n📘 正在处理文件：{file_path}")

    # ---------- 1. 读取数据 ----------
    df = pd.read_csv(file_path)
    df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
    df = df.dropna(subset=["时间"])

    # ---------- 2. 按日聚合 ----------
    daily = (
        df.groupby(df["时间"].dt.date)
        .agg(日度加权情绪总和=("加权情绪贡献", "sum"),
             帖子数=("标题", "count"))
        .reset_index()
        .rename(columns={"时间": "日期"})
    )

    # ---------- 3. 计算情绪指标 ----------
    daily["情绪均值"] = daily["日度加权情绪总和"] / daily["帖子数"]
    daily["指数1_强度*热度"] = daily["情绪均值"] * np.log1p(daily["帖子数"])

    # ---------- 4. 调用归一化函数 ----------
    daily["归一化情绪因子"] = normalize_series(daily["指数1_强度*热度"], method="tanh_zscore")

    # ---------- 5. 保存结果 ----------
    daily.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已完成：{output_path}")

print("\n🎯 所有文件已处理完成！")
