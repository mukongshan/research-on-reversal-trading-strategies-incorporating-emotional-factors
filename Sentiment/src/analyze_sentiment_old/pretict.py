import pandas as pd
import torch
import joblib
import numpy as np
import os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm  # 进度条

# 训练好的模型路径
model_path = "random_forest_model.pkl"
data_dir = r"D:\All_of_mine\大学\项目和比赛\da_chuang\src\data_2\zssh000300"

print('🔹 开始加载模型...')
# 加载 BERT 预训练模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")

# 加载训练好的回归模型
regressor = joblib.load(model_path)

# 🔹 BERT 特征提取（使用缓存加速）
bert_cache = {}

def get_bert_embedding(text):
    """ 获取 BERT 词向量（支持缓存） """
    if text in bert_cache:
        return bert_cache[text]  # 直接返回缓存

    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=10)
    with torch.no_grad():
        output = bert_model(**tokens)

    emb = output.last_hidden_state[:, 0, :].squeeze().numpy()  # 取 [CLS] 位置的向量
    bert_cache[text] = emb  # 存入缓存
    return emb


def calculate_score_for_terms(term):
    """ 计算单个分词结果的预测分数 """
    if isinstance(term, str):  # 仅对字符串处理
        terms = term.split()  # 按空格拆分
        scores = []

        for t in terms:
            emb = get_bert_embedding(t)  # 获取 BERT 向量
            score = regressor.predict([emb])  # 预测分数
            scores.append(score[0])

        return np.mean(scores) if scores else np.nan  # 计算平均分
    return np.nan  # 非字符串返回 NaN


# 🔹 遍历 `data_dir` 下所有子文件夹
all_files = []
for root, _, files in os.walk(data_dir):
    for file in files:
        if file == "extracted_words.xlsx":
            all_files.append(os.path.join(root, file))

if not all_files:
    print("❌ 未找到任何 `extracted_words.xlsx` 文件")
    exit()

# 🔹 处理所有找到的文件
for file in all_files:
    print(f"🔍 正在处理: {file}")
    comments_data = pd.read_excel(file)

    # 确保 "分词结果" 列存在
    if "分词" not in comments_data.columns:
        print(f"⚠️ 警告: {file} 缺少 '分词结果' 列，已跳过")
        continue

    # 计算评分（显示进度条）
    comments_data["预测分数"] = [calculate_score_for_terms(text) for text in tqdm(comments_data["分词"])]

    # 🔹 保存回原文件夹
    output_path = os.path.join(os.path.dirname(file), "股吧评论分词结果_带分数.xlsx")
    comments_data.to_excel(output_path, index=False)
    print(f"✅ 结果已保存到: {output_path}")
