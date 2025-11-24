# save_daily_index.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== 读取原始文件 ====
file_path = "../../mid_result/hs300_data/single_stock/daily_composite_sentiment.csv"  # 换成你的源数据路径
df = pd.read_csv(file_path)

# ========== 1. 字体设置 ==========
# 尝试设置中文字体（根据你的系统修改）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 转换日期
df["日期"] = pd.to_datetime(df["日期"])

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(df["日期"], df["指数1_均值logN"], label="指数1: 均值×logN")
plt.plot(df["日期"], df["指数2_Zscore"], label="指数2: Z-score总和")
plt.plot(df["日期"], df["指数3_强度"], label="指数3: 强度(均值)")
plt.plot(df["日期"], df["指数3_热度"], label="指数3: 热度(log帖子数)", linestyle="--")

plt.xlabel("日期")
plt.ylabel("情绪指数")
plt.title("三种情绪指数对比")
plt.legend()
plt.tight_layout()

# 保存图片（不展示）
plt.savefig("sentiment_index_compare.png", dpi=300)
plt.close()

print("✅ 图像已保存为 sentiment_index_compare.png")