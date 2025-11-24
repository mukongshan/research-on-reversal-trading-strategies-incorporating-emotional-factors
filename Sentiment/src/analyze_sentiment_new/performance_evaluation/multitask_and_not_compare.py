# -*- coding: utf-8 -*-
"""
单文件预测脚本：
1. 加载已训练好的 encoder 和 LightGBM
2. 读取单个 CSV 文件
3. 抽取“标题” → embedding → LightGBM 预测
4. 在原文件中新增“基础分数”列并保存
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from lightgbm import Booster
from tqdm import tqdm

# =============== 配置 ===============
MODEL_NAME = "shibing624/text2vec-base-chinese-sentence"
ENCODER_PATH = "../../../res/training_model/multitask_encoder.pt"
LGBM_PATH = "../../../res/training_model/multitask_lightgbm.txt"

SENTENCE_ONLY_MODEL_PATH = "shibing624/text2vec-base-chinese-sentence"  # 仅句子模型路径
SENTENCE_ONLY_LGBM_PATH = "../../../res/training_model/sentence_only_lightgbm.txt"

INPUT_FILE = "../../../mid_result/training_data/merged_2w_scored_titles_LLM.csv"  # 🔹 指定要预测的文件
OUTPUT_FILE = "titles_with_score.csv"  # 🔹 输出路径

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== 数据集 ===============
class TitleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=32):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["text"] = self.texts[idx]
        return item

# =============== 模型结构 ===============
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head_title = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, task_type="title"):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.head_title(pooled)

    def get_embedding(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        return pooled.detach()

# =============== 加载模型 ===============
print(f"🔹 当前预测设备: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 加载多任务模型（MultiTaskModel）和LightGBM模型
encoder = MultiTaskModel(MODEL_NAME).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()
lgbm_model = Booster(model_file=LGBM_PATH)

# 加载仅句子模型和对应的LightGBM
sentence_encoder = AutoModel.from_pretrained(SENTENCE_ONLY_MODEL_PATH).to(device)
sentence_encoder.eval()
sentence_lgbm_model = Booster(model_file=SENTENCE_ONLY_LGBM_PATH)

# =============== 工具函数 ===============
def get_embeddings(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_vecs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding embeddings"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            vecs = model.get_embedding(input_ids, mask)
            all_vecs.append(vecs.cpu().numpy())
    return np.vstack(all_vecs)

def predict_file(file_path, save_path):
    print(f"📂 正在处理文件: {file_path}")
    df = pd.read_csv(file_path)

    if "标题" not in df.columns:
        raise ValueError(f"❌ 未找到列 '标题'，请检查文件列名。当前列为: {df.columns.tolist()}")

    texts = df["标题"].astype(str).tolist()
    dataset = TitleDataset(texts, tokenizer, max_len=32)

    # 获取多任务模型（MultiTaskModel）的嵌入向量并预测
    X_multitask = get_embeddings(encoder, dataset, batch_size=BATCH_SIZE)
    y_pred_multitask = lgbm_model.predict(X_multitask)

    # 获取仅句子模型的嵌入向量并预测
    X_sentence_only = get_embeddings(sentence_encoder, dataset, batch_size=BATCH_SIZE)
    y_pred_sentence_only = sentence_lgbm_model.predict(X_sentence_only)

    # 将两个模型的预测结果添加到原始 DataFrame 中
    df["MultiTask_基础分数"] = y_pred_multitask
    df["SentenceOnly_基础分数"] = y_pred_sentence_only

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 预测完成，结果已保存至: {save_path}")

# =============== 主程序 ===============
if __name__ == "__main__":
    predict_file(INPUT_FILE, OUTPUT_FILE)
