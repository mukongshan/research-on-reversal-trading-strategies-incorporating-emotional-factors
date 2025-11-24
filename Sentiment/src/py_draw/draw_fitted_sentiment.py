# -*- coding: utf-8 -*-
"""
功能：
1. 读取“情绪拟合值”文件 与 股票行情文件；
2. 自动按日期对齐；
3. 绘制双轴图：左轴为情绪拟合值，右轴为股票收盘价；
4. 可视化情绪与行情的联动趋势。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== 🔧 参数设置 ==========
sentiment_path = "../../mid_result/hs300_data/single_stock/daily_fitted_sentiment.csv"  # 含“情绪拟合值”的文件
stock_path = "../../res/stocks_index_data.csv"                     # 股票行情文件
output_path = "情绪拟合_股票对比.png"    # 输出图片路径

# 列名配置
date_col_sent = "日期"
sentiment_col = "情绪拟合值"
date_col_stock = "date"
close_col = "close"

# ========== 🧠 数据读取 ==========
df_sent = pd.read_csv(sentiment_path, parse_dates=[date_col_sent])
df_stock = pd.read_csv(stock_path, parse_dates=[date_col_stock])

# ==== 日期对齐 ====
df = pd.merge(df_sent, df_stock, left_on=date_col_sent, right_on=date_col_stock, how='inner')

# ==== 只取有效数据 ====
df = df[[date_col_sent, sentiment_col, close_col]].dropna().reset_index(drop=True)

# ==== 时间范围筛选（可选） ====
# ==== 筛选 2023年7月及之后的数据 ====
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2025-02-28")
df = df[df['日期'] >= start_date].reset_index(drop=True)
df = df[df['日期'] <= end_date].reset_index(drop=True)

# ========== 🎨 绘图 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(14,6))

# --- 左轴：情绪拟合值 ---
ax1.plot(df[date_col_sent], df[sentiment_col], color='blue', linewidth=2, label='情绪拟合值')
ax1.set_xlabel("日期", fontsize=12)
ax1.set_ylabel("情绪拟合值", color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(alpha=0.3)

# --- 右轴：股票收盘价 ---
ax2 = ax1.twinx()
ax2.plot(df[date_col_sent], df[close_col], color='red', linewidth=1.8, label='股票收盘价')
ax2.set_ylabel("收盘价", color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

# --- 标题与图例 ---
fig.autofmt_xdate()
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title("情绪拟合值 与 股票走势 对比图", fontsize=16)
plt.tight_layout()

# 保存图片
plt.savefig(output_path, dpi=300)
plt.close()
print(f"✅ 图像已保存：{output_path}")
