# -*- coding: utf-8 -*-
"""
功能：
1. 批量读取情绪拟合文件与股票文件；
2. 对齐日期后绘制“情绪拟合值 vs 股票收盘价”；
3. LOWESS 平滑；
4. 每组可单独设置时间范围；
5. 自动保存对比图。
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import os

# ========== 1️⃣ 字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ========== 2️⃣ 输入输出路径列表 ==========
sentiment_paths = [
    "../../mid_result/hs300_data/hs300_stock_forum/daily_fitted_sentiment.csv",
    "../../mid_result/hs300_data/single_stock/daily_fitted_sentiment.csv"
]

stock_paths = [
    "../../res/stocks_index_data.csv",
    "../../res/stocks_index_data.csv"
]

output_imgs = [
    "沪深300股吧近五年情绪指标与股票收盘价对比.png",
    "沪深300成分股近一年情绪指标与股票收盘价对比.png"
]

# ========== 3️⃣ 每个文件的时间范围（与上面对应） ==========
time_ranges = [
    ("2020-01-01", "2025-02-28"),  # 情绪1: 沪深300
    ("2024-01-01", "2025-02-28")   # 情绪2: 个股
]

# ========== 4️⃣ 绘图参数 ==========
sentiment_cols = ["情绪拟合值"]  # 要绘制的情绪列
frac_smooth = 0.08               # LOWESS 平滑程度

# ========== 5️⃣ 主逻辑 ==========
for sent_path, stock_path, out_img, (start_str, end_str) in zip(sentiment_paths, stock_paths, output_imgs, time_ranges):
    print(f"\n📘 正在处理：{sent_path}")

    # ---- 读取数据 ----
    if not os.path.exists(sent_path) or not os.path.exists(stock_path):
        print(f"❌ 文件缺失：{sent_path} 或 {stock_path}")
        continue

    df_sent = pd.read_csv(sent_path, parse_dates=['日期'])
    df_stock = pd.read_csv(stock_path, parse_dates=['date'])

    # ---- 日期对齐 ----
    df = pd.merge(df_sent, df_stock, left_on='日期', right_on='date', how='inner')

    # ---- 筛选不同时间段 ----
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)].reset_index(drop=True)
    if df.empty:
        print(f"⚠️ 数据为空：{sent_path}")
        continue

    # ---- 绘图 ----
    fig, ax1 = plt.subplots(figsize=(14, 6))
    colors = ['blue', 'orange']

    for col, color in zip(sentiment_cols, colors):
        smoothed = lowess(df[col], np.arange(len(df)), frac=frac_smooth, return_sorted=True)[:, 1]
        label = "综合情绪指标" if col == "指数1_强度*热度" else f"{col}"
        ax1.plot(df['日期'], smoothed, color=color, linewidth=1.8, label=label)

    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('情绪指标值', fontsize=12, color='blue')
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='blue')

    import matplotlib.dates as mdates
    # 设置 x 轴刻度：每 3 个月一个，比如 2024-01, 2024-04, 2024-07...
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 关键：每3个月一个刻度
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式：2024-04

    # ---- 绘制股票收盘价（右轴） ----
    ax2 = ax1.twinx()
    smoothed_close = lowess(df['close'], np.arange(len(df)), frac=frac_smooth, return_sorted=True)[:, 1]
    ax2.plot(df['日期'], smoothed_close, color='red', linewidth=2, label='股票收盘价')
    ax2.set_ylabel('收盘价', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # ---- 图例 ----
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)

    # ---- 标题 ----
    plt.title(f"{os.path.basename(out_img).split('.')[0]}", fontsize=15)
    fig.autofmt_xdate()
    plt.tight_layout()

    # ---- 保存图片 ----
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"✅ 图像已保存：{out_img}")

print("\n🎯 所有情绪-股价对比图绘制完成！")
