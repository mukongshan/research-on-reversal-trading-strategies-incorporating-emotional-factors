# -*- coding: utf-8 -*-
"""
Phase 1: Baseline
功能：
- 输入：词语 -> BERT CLS 向量 (768维)
- 模型：LinearRegression / Ridge / Lasso
- 划分：80% 训练集 / 20% 测试集
- 输出：预测结果 CSV + 指标对比表
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================= 全局设置 =================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据加载 =================
def load_data(data_path):
    """读取 Excel 数据，返回词语和分数"""
    df = pd.read_excel(data_path, engine="openpyxl")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["词语", "分数"]).copy()
    df["词语"] = df["词语"].astype(str)
    df["分数"] = pd.to_numeric(df["分数"], errors="coerce")
    df = df.dropna(subset=["分数"])
    return df["词语"].tolist(), df["分数"].to_numpy()

# ================= BERT 向量化 =================
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    bert = BertModel.from_pretrained("bert-base-chinese").to(device)
    bert.eval()
    return tokenizer, bert

@torch.no_grad()
def get_embeddings(texts, tokenizer, bert, batch_size=64, max_len=10):
    """将词语转为 BERT CLS 向量"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = bert(**enc)
        cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_vecs.append(cls)
    return np.vstack(all_vecs)

# ================= 模型训练与评估 =================
def pearson_corr(a, b):
    return np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练并评估三种 baseline 模型"""
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
        "Lasso": Lasso(alpha=0.001, random_state=SEED, max_iter=10000)
    }

    results = []
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 兼容老版本 sklearn：RMSE 手动开根号
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        corr = pearson_corr(y_test, y_pred)

        results.append({
            "模型": name,
            "RMSE↓": rmse,
            "MAE↓": mae,
            "R2↑": r2,
            "Pearson↑": corr
        })
        preds[name] = y_pred

    return pd.DataFrame(results), preds


# ================= 主入口 =================
def main():
    DATA_PATH = r"../../../mid_result/training_data/训练集.xlsx"
    SAVE_PRED_PATH = "baseline_predictions.csv"
    SAVE_SUMMARY_PATH = "baseline_results.csv"

    # 1. 加载数据
    texts, y = load_data(DATA_PATH)

    # 2. BERT 向量化
    tokenizer, bert = load_bert_model()
    X = get_embeddings(texts, tokenizer, bert, batch_size=64, max_len=10)

    # 3. 划分训练/测试集
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, texts, test_size=0.2, random_state=SEED
    )

    # 4. 训练与评估
    summary, preds = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 5. 保存结果
    pred_df = pd.DataFrame({"词语": texts_test, "真实分数": y_test})
    for name, y_pred in preds.items():
        pred_df[f"{name}_pred"] = y_pred
    pred_df.to_csv(SAVE_PRED_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SAVE_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    # 6. 打印结果
    print("\n=== Baseline 结果 ===")
    print(summary.to_string(index=False))
    print(f"\n预测明细保存至: {os.path.abspath(SAVE_PRED_PATH)}")
    print(f"指标对比保存至: {os.path.abspath(SAVE_SUMMARY_PATH)}")

# ================= 运行 =================
if __name__ == "__main__":
    main()
