# -*- coding: utf-8 -*-
"""
功能：
1. 批量遍历股票子目录读取 daily_sentiment_index.csv；
2. 对列“归一化情绪因子”使用 Savitzky–Golay 滤波；
3. 生成新列“情绪拟合值”；
4. 保存到同一子目录下 daily_fitted_sentiment.csv。
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

# ========== 📂 目录配置 ==========
INPUT_ROOT = "../../../../mid_result/hs300_data/stocks_separate_data"
INPUT_FILENAME = "daily_sentiment_index.csv"
OUTPUT_FILENAME = "daily_fitted_sentiment.csv"

# ========== 🔧 滤波参数 ==========
TARGET_COLUMN = "归一化情绪因子"
WINDOW_LENGTH = 15  # 必须为奇数
POLYORDER = 2       # 建议 2 或 3

# ========== 🧠 SG 平滑函数 ==========
def smooth_with_savgol(df, col_name, window_length=15, polyorder=3):
    y = df[col_name].astype(float).values
    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        print(f"⚠️ 数据较短，自动调整窗口长度为 {window_length}")
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y_smooth

# ========== 🚀 主程序 ==========
# 可选：截断时间
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
    df = pd.read_csv(input_path, parse_dates=['日期'])

    # 截断时间范围
    df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)].reset_index(drop=True)
    if df.empty:
        print(f"⚠️ {stock_dir} 数据在指定日期范围为空，跳过")
        continue

    if TARGET_COLUMN not in df.columns:
        print(f"❌ 找不到列：{TARGET_COLUMN}")
        continue

    # SG 滤波
    df["情绪拟合值"] = smooth_with_savgol(df, TARGET_COLUMN, WINDOW_LENGTH, POLYORDER)

    # 保存结果
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存: {output_path}")

print("\n🎯 所有子目录均已处理完成！")
