import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np

# ========== 1. 字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==== 读取综合情绪数据 ====
sentiment_path = "../../mid_result/hs300_data/single_stock/daily_composite_sentiment.csv"
df_sent = pd.read_csv(sentiment_path, parse_dates=['日期'])

# ==== 读取股票数据 ====
stock_path = "../../res/stocks_index_data.csv"
df_stock = pd.read_csv(stock_path, parse_dates=['date'])

# ==== 日期对齐 ====
df = pd.merge(df_sent, df_stock, left_on='日期', right_on='date', how='inner')

# ==== 筛选 2023年7月及之后的数据 ====
start_date = pd.to_datetime("2023-07-01")
df = df[df['日期'] >= start_date].reset_index(drop=True)

# ==== 计算 LOWESS 平滑趋势 ====
frac_smooth = 0.08  # 平滑系数，可调
smoothed_sentiment = lowess(df['综合情绪得分'], np.arange(len(df)), frac=frac_smooth, return_sorted=True)[:,1]
smoothed_close = lowess(df['close'], np.arange(len(df)), frac=frac_smooth, return_sorted=True)[:,1]

# ==== 绘图 ====
fig, ax1 = plt.subplots(figsize=(14,6))

# --- 左轴：综合情绪趋势 ---
ax1.plot(df['日期'], smoothed_sentiment, color='red', linewidth=2, label='情绪趋势曲线')
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('综合情绪得分', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# --- 右轴：平滑股票收盘价 ---
ax2 = ax1.twinx()
ax2.plot(df['日期'], smoothed_close, color='green', linewidth=1.5, label='股票收盘价（平滑）')
ax2.set_ylabel('收盘价', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')

# --- 图表美化 ---
fig.autofmt_xdate()
ax1.grid(alpha=0.3)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('2023年7月之后综合情绪趋势与股票收盘价对比（平滑）', fontsize=16)
plt.tight_layout()

# 保存图片
plt.savefig("综合情绪得分趋势与股票收盘价对比.png", dpi=300)
plt.close()
