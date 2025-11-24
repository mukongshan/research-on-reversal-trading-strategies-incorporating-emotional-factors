import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 从 CSV 文件读取数据（请替换为你的实际文件路径）
# 假设你的 csv 文件名为：model_metrics.csv
csv_file = '../../mid_result/model_compare/multi_or_not_index_compare.csv'  # ← 请修改为你的实际文件路径
df = pd.read_csv(csv_file)


# 2. 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 3. 模型和指标
models = df['模型'].tolist()  # 如 ['sentence_only', 'multitask']
model_names = models  # 直接使用
metrics = ['RMSE', 'MAE', 'R2', 'Pearson']  # 指标顺序要与 CSV 列顺序一致

# 4. 构造正确的数据结构：每个模型对应所有指标的值
# 我们要得到：values[0] = [model1的RMSE, MAE, R2, Pearson], values[1] = [model2的...]
values = []
for _, row in df.iterrows():
    model_values = [
        row['RMSE'],
        row['MAE'],
        row['R2'],
        row['Pearson']
    ]
    values.append(model_values)

num_models = len(models)
num_metrics = len(metrics)

# 5. 绘图
x = np.arange(num_metrics)  # 指标位置：0:RMSE, 1:MAE, 2:R2, 3:Pearson
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 6. 遍历每个模型，画出它在所有指标上的柱子
for i in range(num_models):
    ax.bar(x + i * width, values[i], width, label=model_names[i])

# 7. 图表元素
ax.set_xlabel('指标', fontsize=12)
ax.set_ylabel('指标值', fontsize=12)
ax.set_title('不同模型在各指标上的表现对比', fontsize=14)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)

# 8. 显示
plt.tight_layout()
plt.savefig('多任务模型_按指标对比柱状图.png')