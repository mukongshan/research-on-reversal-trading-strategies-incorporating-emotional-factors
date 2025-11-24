import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import font_manager
import numpy as np


frac = 0.02

# ==== 读取日度综合情绪结果 ====
file_path = "../../mid_result/hs300_data/single_stock/daily_composite_sentiment.csv"  # 替换为你的路径
df = pd.read_csv(file_path, parse_dates=['日期'])

# ==== 筛选 2023年7月及之后的数据 ====
start_date = pd.to_datetime("2023-07-01")
df = df[df['日期'] >= start_date].reset_index(drop=True)

# ==== 设置中文显示 ====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# ==== 画散点图 ====
plt.figure(figsize=(14,6))
plt.scatter(df['日期'], df['综合情绪得分'], s=20, alpha=0.3, color='blue', label='每日综合情绪得分')

# ==== LOWESS 拟合平滑曲线 ====
# frac 控制平滑程度，数据多可以设置小一点
smoothed = lowess(df['综合情绪得分'], np.arange(len(df)), frac=frac, return_sorted=True)
plt.plot(df['日期'], smoothed[:,1], color='red', linewidth=2, label='情绪趋势曲线')

# ==== 图表美化 ====
plt.title('每日综合情绪得分趋势', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('综合情绪得分', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存图片（不展示）
plt.savefig("composite_sentiment.png", dpi=300)
plt.close()
