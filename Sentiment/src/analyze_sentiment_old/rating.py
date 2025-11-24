import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 读取 test_data.xlsx 数据，用来训练模型
train_data = pd.read_excel("D:\\All_of_mine\\大学\\项目和比赛\\da_chuang\\res\\汉语情感词极值表_1.xlsx")  # 假设有 "词语" 和 "分数"

print('开始加载模型')
# 加载 BERT 预训练模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")



# BERT 特征提取函数
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=10)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()  # 取 [CLS] 位置的向量

print('开始提取特征')
# 为训练数据提取特征
X = train_data["词语"].apply(get_bert_embedding).tolist()
X = np.array(X)  # 转换为 NumPy 数组
y = train_data["分数"].values

print('开始训练')

# 训练回归模型
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)

# 预测
y_pred = regressor.predict(X)

# 计算训练集误差
mse = mean_squared_error(y, y_pred)
print(f"均方误差: {mse}")



print('开始预测')
# 读取 comments_processed.xlsx 数据
comments_data = pd.read_excel("D:\\All_of_mine\\大学\\项目和比赛\\da_chuang\\src\\data")


# 为 "分词结果" 计算分数
def calculate_score_for_terms(term):
    if isinstance(term, str):  # 仅对字符串类型进行处理
        terms = term.split()  # 将 "分词结果" 按空格分割成多个词语
        scores = []

        # 对每个词语进行评分
        for t in terms:
            emb = get_bert_embedding(t)  # 获取每个词语的 BERT 向量
            score = regressor.predict([emb])  # 预测分数
            scores.append(score[0])

        # 计算词语的平均分数
        if scores:
            return np.mean(scores)
    return np.nan  # 如果输入不是字符串或没有有效的词语，返回 NaN


# 应用函数为每条 "分词结果" 计算平均分数
comments_data["预测分数"] = comments_data["分词结果"].apply(calculate_score_for_terms)

# 存储预测结果
comments_data.to_excel(r"guba_comments_with_scores.xlsx", index=False)
print("预测结果已保存到 comments_with_scores.xlsx")
