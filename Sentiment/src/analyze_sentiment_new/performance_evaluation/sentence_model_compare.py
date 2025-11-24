# -*- coding: utf-8 -*-
"""
多模型比较 + LightGBM 回归
遇到无法拉取的模型时自动跳过，不中断程序
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# =============== 全局设置 ===============
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== 输入 / 输出 / 模型选择 ===============
DATA_PATH = "../../../mid_result/training_data/merged_15K_scored_titles.csv"
SAVE_SUMMARY_PATH = "model_comparison_results.csv"

MODEL_LIST = [
    "shibing624/text2vec-base-chinese", # 0.4844
    "shibing624/text2vec-base-chinese-sentence", # 0.4933
    "shibing624/text2vec-base-chinese-paraphrase", # 0.4505
    "hfl/chinese-roberta-wwm-ext", # 0.4395
    "hfl/chinese-macbert-base", # 0.3679
    "nghuyong/ernie-3.0-base-zh", # 0.1793
    "bert-base-chinese" # 0.3639
]

# =============== 数据加载 ===============
def load_data(data_path):
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(data_path, engine="openpyxl")

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["标题", "分数"]).copy()
    df["标题"] = df["标题"].astype(str)
    df["分数"] = pd.to_numeric(df["分数"], errors="coerce")
    df = df.dropna(subset=["分数"])
    return df["标题"].tolist(), df["分数"].to_numpy()

# =============== 向量化 ===============
def get_embeddings(texts, model_name, batch_size=64, max_len=32):
    try:
        print(f"\n⚙️ 正在加载模型 {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型 {model_name} 加载失败: {e}")
        return None

    print(f"✅ 模型 {model_name} 加载完成，开始生成向量 ...")
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)

            # 兼容 text2vec（sentence embedding）和普通 BERT
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                cls = out.pooler_output.detach().cpu().numpy()
            else:
                cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()

            all_vecs.append(cls)
    return np.vstack(all_vecs)

# =============== 模型训练与评估 ===============
def pearson_corr(a, b):
    return np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan

def train_and_evaluate(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    corr = pearson_corr(y_test, y_pred)

    return {
        "模型": model_name,
        "RMSE↓": rmse,
        "MAE↓": mae,
        "R2↑": r2,
        "Pearson↑": corr
    }

# =============== 主入口 ===============
def main():
    texts, y = load_data(DATA_PATH)
    all_results = []

    for model_name in MODEL_LIST:
        X = get_embeddings(texts, model_name)
        if X is None:  # 跳过失败的模型
            continue
        try:
            results = train_and_evaluate(X, y, model_name)
            all_results.append(results)
            print(f"✅ {model_name} 评估完成: R2={results['R2↑']:.4f}")
        except Exception as e:
            print(f"❌ {model_name} 训练/评估失败: {e}")
            continue

    # 保存所有结果
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(SAVE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
        print("\n=== 所有模型对比结果 ===")
        print(df_results.to_string(index=False))
        print(f"\n💾 结果已保存至: {os.path.abspath(SAVE_SUMMARY_PATH)}")
    else:
        print("⚠️ 没有成功运行的模型")

if __name__ == "__main__":
    main()
