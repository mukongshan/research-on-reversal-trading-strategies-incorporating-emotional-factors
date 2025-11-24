import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 训练数据路径
train_data_path = "D:\\All_of_mine\\大学\\项目和比赛\\da_chuang\\res\\训练集.xlsx"

print('开始加载模型')
# 加载 BERT 预训练模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")


# BERT 特征提取函数
def get_bert_embedding(text):
    if pd.isna(text):  # 避免 NaN 传入 BERT
        text = ""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=10)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()  # 取 [CLS] 位置的向量


print('开始读取训练数据')
# 读取 Excel 数据
train_data = pd.read_excel(train_data_path, engine="openpyxl")

# 确保列名无空格或 BOM
train_data.columns = train_data.columns.str.strip()

# 确保 "词语" 列是字符串
train_data["词语"] = train_data["词语"].astype(str)

# 确保 "分数" 列是数值型
train_data["分数"] = pd.to_numeric(train_data["分数"], errors="coerce")

# 删除可能的 NaN 数据行
train_data.dropna(subset=["词语", "分数"], inplace=True)

print('开始提取特征')
# 提取特征
X = np.array(train_data["词语"].apply(get_bert_embedding).tolist())
y = train_data["分数"].values

print('开始训练模型')
# 训练回归模型
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)

# 计算训练集误差
y_pred = regressor.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"训练集均方误差: {mse}")

# 保存模型
model_save_path = "random_forest_model.pkl"
joblib.dump(regressor, model_save_path)
print(f"模型已保存到 {model_save_path}")
