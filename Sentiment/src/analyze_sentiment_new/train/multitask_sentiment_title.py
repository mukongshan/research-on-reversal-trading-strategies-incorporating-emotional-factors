# -*- coding: utf-8 -*-
"""
路径B：多任务学习 Encoder → LightGBM 回归
训练时：词语 + 标题 (多任务，支持Loss权重控制)
预测时：提取标题 embedding → LightGBM 回归
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============== 配置 ===============
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 3

WORD_DATA_PATH = "scored_words.csv"     # 词语数据：列 ["文本","分数"]
TITLE_DATA_PATH = "scored_titles.csv"   # 标题数据：列 ["文本","分数"]

SAVE_ENCODER_PATH = "../../../mid_result/training_model/multitask_encoder.pt"
SAVE_LIGHTGBM_PATH = "../../../mid_result/multitask_lightgbm.txt"
SAVE_PRED_PATH = "multitask_sentiment_title_predictions.csv"

# =============== 数据集 ===============
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32, task_type="title"):
        self.texts = df["文本"].tolist()
        self.labels = df["分数"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_type = task_type

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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        item["task_type"] = self.task_type
        item["text"] = self.texts[idx]
        return item

# =============== 模型 ===============
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head_word = nn.Linear(hidden_size, 1)   # 词语回归
        self.head_title = nn.Linear(hidden_size, 1)  # 标题回归

    def forward(self, input_ids, attention_mask, task_type):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        if task_type == "word":
            return self.head_word(pooled)
        else:
            return self.head_title(pooled)

    def get_embedding(self, input_ids, attention_mask):
        """抽取共享encoder的embedding"""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        return pooled.detach()

# =============== 训练 & 提取特征 ===============
def train_epoch(model, loaders, optimizer, loss_fn, alpha=1.0, beta=1.0):
    """
    多任务训练，支持loss权重
    alpha: 词语任务loss权重
    beta:  标题任务loss权重
    """
    model.train()
    total_loss = 0
    for task, loader in loaders.items():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, mask, task_type=task).squeeze()

            loss = loss_fn(preds, labels)
            # 根据任务类型调整权重
            if task == "word":
                loss = alpha * loss
            else:
                loss = beta * loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss

def extract_embeddings(model, dataset, batch_size=64):
    """从训练好的 encoder 提取标题 embedding"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_vecs, labels, texts = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            vecs = model.get_embedding(input_ids, mask)
            all_vecs.append(vecs.cpu().numpy())
            labels.extend(batch["labels"].numpy())
            texts.extend(batch["text"])
    return np.vstack(all_vecs), np.array(labels), texts

# =============== LightGBM 回归 ===============
def pearson_corr(a, b):
    return np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan

def run_lightgbm(X, y, texts, save_path=None):
    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, texts, test_size=0.2, random_state=SEED
    )

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    model.booster_.save_model(SAVE_LIGHTGBM_PATH)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    corr = pearson_corr(y_test, y_pred)

    # 保存预测结果
    if save_path:
        df = pd.DataFrame({
            "文本": text_test,
            "真实分数": y_test,
            "预测分数": y_pred
        })
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存至: {save_path}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": corr}

# =============== 主程序 ===============
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 加载数据
    word_df = pd.read_csv(WORD_DATA_PATH)
    title_df = pd.read_csv(TITLE_DATA_PATH)
    word_dataset = TextDataset(word_df, tokenizer, max_len=8, task_type="word")
    title_dataset = TextDataset(title_df, tokenizer, max_len=32, task_type="title")

    word_loader = DataLoader(word_dataset, batch_size=BATCH_SIZE, shuffle=True)
    title_loader = DataLoader(title_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loaders = {"word": word_loader, "title": title_loader}

    # ====== Step1: 多任务训练 Encoder ======
    model = MultiTaskModel(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== 多任务训练 Encoder ===")
    for epoch in range(EPOCHS):
        # 🔹 强调标题任务：beta=3.0
        loss = train_epoch(model, loaders, optimizer, loss_fn, alpha=1.0, beta=3.0)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}")

    torch.save(model.state_dict(), SAVE_ENCODER_PATH)
    print(f"共享Encoder已保存: {SAVE_ENCODER_PATH}")

    # ====== Step2: 提取标题 Embeddings ======
    X, y, texts = extract_embeddings(model, title_dataset)

    # ====== Step3: LightGBM 回归评估 ======
    results = run_lightgbm(X, y, texts, save_path=SAVE_PRED_PATH)

    print("\n=== 评估结果 (LightGBM on Title Embeddings) ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()


# [原始文本数据：词语 & 标题]
#        ↓
# [Tokenize + 构造成Dataset]
#        ↓
# [MultiTaskModel：共享BERT Encoder + 两个任务头]
#        ↓
# [多任务训练：同时优化词语和标题的回归任务，可调节权重]
#        ↓
# [保存共享Encoder（不含任务头）]
#        ↓
# [提取标题的Embedding向量（用Encoder的pooled output）]
#        ↓
# [用这些Embedding作为特征，训练LightGBM回归模型]
#        ↓
# [评估模型效果并保存预测结果]

#