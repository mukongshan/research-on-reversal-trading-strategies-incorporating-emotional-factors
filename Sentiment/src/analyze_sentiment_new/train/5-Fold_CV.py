import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 你之前的模型类要导入
# from your_module import MultiTaskModel, TextDataset, extract_embeddings, pearson_corr

# =============== 配置 ===============
MODEL_NAME = "shibing624/text2vec-base-chinese-sentence"
SAVE_ENCODER_PATH = "/content/drive/MyDrive/Reversal_Trading_Strategy/model/training_modelmultitask_encoder.pt"
TITLE_DATA_PATH = "/content/drive/MyDrive/Reversal_Trading_Strategy/data/merged_2w_scored_titles_multi.csv"
BATCH_SIZE = 64
SEED = 42
FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# =============== 函数 ===============

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

def pearson_corr(a, b):
    return np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan

def run_kfold_lightgbm(X, y, texts, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics = []

    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

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

        metrics.append([rmse, mae, r2, corr])
        print(f"[Fold {fold}] RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Pearson={corr:.4f}")
        fold += 1

    metrics = np.array(metrics)
    print("\n=== 平均结果 (KFold CV) ===")
    print(f"RMSE={metrics[:,0].mean():.4f}, MAE={metrics[:,1].mean():.4f}, "
          f"R2={metrics[:,2].mean():.4f}, Pearson={metrics[:,3].mean():.4f}")
    return metrics

# =============== 主程序 ===============
def main():
    print(f"🔹 当前训练设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    title_df = pd.read_csv(TITLE_DATA_PATH)
    title_dataset = TextDataset(title_df, tokenizer, max_len=32, task_type="title")

    # 加载训练好的encoder
    model = MultiTaskModel(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(SAVE_ENCODER_PATH, map_location=device))
    model.eval()

    # 提取标题embedding
    X, y, texts = extract_embeddings(model, title_dataset, batch_size=BATCH_SIZE)

    # 交叉验证
    run_kfold_lightgbm(X, y, np.array(texts), n_splits=FOLDS)

if __name__ == "__main__":
    main()
