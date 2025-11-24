import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# 读取结果数据
INPUT_FILE = "../../../mid_result/model_compare/multitask_sentiment_title_predictions.csv"  # 输入文件路径
OUTPUT_FILE = "../../../mid_result/model_compare/multitask_model_results.csv"  # 输出文件路径

# 读取CSV文件
df = pd.read_csv(INPUT_FILE)

# 检查数据中是否存在 '分数' 和 '基础分数' 列
if "分数" not in df.columns or "基础分数" not in df.columns:
    raise ValueError(f"❌ 未找到列 '分数' 或 '基础分数'，请检查文件列名。当前列为: {df.columns.tolist()}")

# 获取真实分数和预测分数
y_true = df["分数"].values  # 真实分数
y_pred = df["基础分数"].values  # 预测分数

# 计算 RMSE, MAE, R2 和 Pearson 相关系数
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
r2 = r2_score(y_true, y_pred)  # R-squared
corr, _ = pearsonr(y_true, y_pred)  # Pearson Correlation

# 打印计算结果
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
print(f"Pearson Correlation: {corr}")

# 保存结果到新的CSV文件
evaluation_results = {
    "RMSE↓": [rmse],
    "MAE↓": [mae],
    "R2↑": [r2],
    "Pearson↑": [corr]
}

# 创建一个新的DataFrame
evaluation_df = pd.DataFrame(evaluation_results)

# 检查输出路径是否存在，如果没有则创建
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# 保存评估结果
evaluation_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 评估结果已保存至: {OUTPUT_FILE}")
