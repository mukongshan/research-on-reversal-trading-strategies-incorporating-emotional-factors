"""
extract_qwen_embeddings.py

功能：
1. 读取 CSV (词语 + 分数)
2. 使用本地 Qwen2-7B-Instruct 模型生成向量 (CLS embedding)
3. 保存为 CSV，供 LightGBM 使用
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================
# 配置
# =========================
INPUT_FILE = r"../../../mid_result/training_data/scored_16K_LLM.csv"   # 输入 CSV
OUTPUT_FILE = r"test_scored_1600_QwenEmbeddings.csv"   # 输出 CSV
MODEL_PATH = r"Qwen/Qwen2-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16   # 根据显存调整
MAX_LEN = 128     # 截断长度


# =========================
# Step 1: 加载数据
# =========================
def load_data(input_file):
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    texts = df["词语"].astype(str).tolist()
    scores = df["分数"].values
    return df, texts, scores


# =========================
# Step 2: 加载模型
# =========================
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(DEVICE)
    model.eval()
    return tokenizer, model


# =========================
# Step 3: 生成向量
# =========================
def generate_embeddings(texts, tokenizer, model, batch_size=16, max_len=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            # 取 CLS token 表示作为句向量
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.append(cls_emb)

    embeddings = np.vstack(embeddings)  # shape = (N, hidden_dim)
    return embeddings


# =========================
# Step 4: 保存结果
# =========================
def save_embeddings(df, scores, embeddings, output_file):
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols)
    df_out = pd.concat([df[["词语"]], pd.Series(scores, name="分数"), df_emb], axis=1)
    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存向量化结果到 {output_file}")


# =========================
# 主函数
# =========================
def main():
    # Step 1: 读取数据
    df, texts, scores = load_data(INPUT_FILE)

    # Step 2: 加载模型
    tokenizer, model = load_model(MODEL_PATH)

    # Step 3: 生成向量
    embeddings = generate_embeddings(texts, tokenizer, model,
                                     batch_size=BATCH_SIZE, max_len=MAX_LEN)

    # Step 4: 保存
    save_embeddings(df, scores, embeddings, OUTPUT_FILE)


if __name__ == "__main__":
    main()
