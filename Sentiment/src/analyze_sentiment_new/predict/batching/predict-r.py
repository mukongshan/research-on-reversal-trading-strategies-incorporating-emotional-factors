# -*- coding: utf-8 -*-
"""
批量预测脚本：
1. 加载已训练好的 encoder 和 LightGBM
2. 遍历指定文件夹下的 69 个 CSV
3. 抽取标题 → embedding → LightGBM 预测
4. 在原文件中加一列“预测分数”，保存到输出目录
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
ENCODER_PATH = "../../../mid_result/training_model/multitask_encoder.pt"
LGBM_PATH = "../../../mid_result/training_model/multitask_lightgbm.txt"

INPUT_DIR = "../../../mid_result/hs300_data/unsocred_split_titles_data"  # 69个csv文件目录
OUTPUT_DIR = "../../../mid_result/hs300_data/680W_scored_titles"
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# =============== 模型 ===============
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head_word = nn.Linear(hidden_size, 1)
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

encoder = MultiTaskModel(MODEL_NAME).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()

lgbm_model = Booster(model_file=LGBM_PATH)

# =============== 工具函数 ===============
def get_embeddings(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_vecs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            vecs = model.get_embedding(input_ids, mask)
            all_vecs.append(vecs.cpu().numpy())
    return np.vstack(all_vecs)

def predict_file(file_path, save_path):
    df = pd.read_csv(file_path)
    texts = df["标题"].astype(str).tolist()

    dataset = TitleDataset(texts, tokenizer, max_len=32)
    X = get_embeddings(encoder, dataset, batch_size=BATCH_SIZE)
    y_pred = lgbm_model.predict(X)

    df["基础分数"] = y_pred
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已完成 {file_path} → {save_path}")

# =============== 主程序 ===============
def main():
    print(f"🔹 当前训练设备: {device}")
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    files.sort()  # 保证顺序一致

    for f in tqdm(files, desc="处理所有CSV文件"):
        file_path = os.path.join(INPUT_DIR, f)
        save_path = os.path.join(OUTPUT_DIR, f.replace(".csv", "_scored.csv"))
        predict_file(file_path, save_path)

if __name__ == "__main__":
    main()
