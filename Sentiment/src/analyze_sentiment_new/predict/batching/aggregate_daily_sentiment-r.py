# -*- coding: utf-8 -*-
"""
批量计算日度情绪指数
目录结构:
    主目录/
        股票代码1/
            titles_with_weighted_score.csv
        股票代码2/
            titles_with_weighted_score.csv
输出: 每个子目录下新增 daily_sentiment_index.csv
"""

import pandas as pd
import numpy as np
import os

# ========== 配置 ==========
INPUT_ROOT = "../../../../mid_result/hs300_data/stocks_separate_data"
INPUT_FILENAME = "titles_with_weighted_score.csv"
OUTPUT_FILENAME = "daily_sentiment_index.csv"
NORMALIZE_METHOD = "tanh_zscore"  # 可选: "tanh_zscore", "robust", "minmax"

# ========== 归一化函数 ==========
def normalize_series(series, method=NORMALIZE_METHOD):
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

# ========== 主程序 ==========
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2025-02-28")

for stock_dir in os.listdir(INPUT_ROOT):
    full_dir = os.path.join(INPUT_ROOT, stock_dir)
    if not os.path.isdir(full_dir):
        continue

    input_path = os.path.join(full_dir, INPUT_FILENAME)
    output_path = os.path.join(full_dir, OUTPUT_FILENAME)

    if not os.path.exists(input_path):
        print(f"⚠️ 文件不存在: {input_path}")
        continue

    print(f"\n📘 正在处理: {input_path}")
    df = pd.read_csv(input_path)
    df["纠正后时间"] = pd.to_datetime(df["纠正后时间"], errors="coerce")
    df = df.dropna(subset=["纠正后时间"])

    # ===== 截断时间 =====
    df = df[(df["纠正后时间"] >= start_date) & (df["纠正后时间"] <= end_date)].reset_index(drop=True)
    if df.empty:
        print(f"⚠️ {stock_dir} 数据在指定日期范围为空，跳过")
        continue

    # 按日聚合
    daily = (
        df.groupby(df["纠正后时间"].dt.date)
        .agg(日度加权情绪总和=("加权情绪贡献", "sum"),
             帖子数=("标题", "count"))
        .reset_index()
        .rename(columns={"纠正后时间": "日期"})
    )

    # 计算指数1
    daily["情绪均值"] = daily["日度加权情绪总和"] / daily["帖子数"]
    daily["指数1_强度*热度"] = daily["情绪均值"] * np.log1p(daily["帖子数"])

    # 归一化
    daily["归一化情绪因子"] = normalize_series(daily["指数1_强度*热度"], method=NORMALIZE_METHOD)

    # 保存结果
    daily.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存: {output_path}")

print("\n🎯 所有子目录已处理完成！")

