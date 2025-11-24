# -*- coding: utf-8 -*-
"""
功能：
1. 批量读取多个 CSV 文件；
2. 对列“归一化情绪因子”使用 Savitzky–Golay 滤波；
3. 生成新列“情绪拟合值”；
4. 将结果保存为新的 CSV 文件。
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

# ========== 📂 输入输出路径列表 ==========
input_paths = [
    "../../../mid_result/hs300_data/hs300_stock_forum/daily_sentiment_index.csv",
    "../../../mid_result/hs300_data/single_stock/daily_sentiment_index.csv"
]

output_paths = [
    "../../../mid_result/hs300_data/hs300_stock_forum/daily_fitted_sentiment.csv",
    "../../../mid_result/hs300_data/single_stock/daily_fitted_sentiment.csv"
]

# ========== 🔧 滤波参数 ==========
TARGET_COLUMN = "归一化情绪因子"
WINDOW_LENGTH = 51   # 必须为奇数
POLYORDER = 2        # 建议 2 或 3


# ========== 🧠 平滑函数 ==========
def smooth_with_savgol(df, col_name, window_length=15, polyorder=3):
    """
    使用 Savitzky–Golay 滤波器进行平滑拟合
    """
    y = df[col_name].astype(float).values

    # 若数据长度小于窗口，则自动调小窗口
    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        print(f"⚠️ 数据较短，自动调整窗口长度为 {window_length}")

    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y_smooth


# ========== 🚀 主程序 ==========
def main():
    for in_path, out_path in zip(input_paths, output_paths):
        print(f"\n📘 正在处理文件：{in_path}")

        if not os.path.exists(in_path):
            print(f"❌ 找不到输入文件：{in_path}")
            continue

        df = pd.read_csv(in_path)

        if TARGET_COLUMN not in df.columns:
            print(f"❌ 找不到列：{TARGET_COLUMN}")
            print("当前列名：", list(df.columns))
            continue

        # 应用 SG 滤波器
        df["情绪拟合值"] = smooth_with_savgol(df, TARGET_COLUMN, WINDOW_LENGTH, POLYORDER)

        # 保存结果
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 拟合完成！结果已保存到：{out_path}")

    print("\n🎯 所有文件均已处理完成！")


if __name__ == "__main__":
    main()
