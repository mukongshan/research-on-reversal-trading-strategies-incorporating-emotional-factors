"""
train_multiclass_tfidf_lightgbm.py

功能：
1. 从 Excel 中读取 "词语" + "分数"
2. 用 TF-IDF 向量化文本
3. 将连续标签 [-1,1] 分桶成 9 类 / 5 类
4. 用 LightGBM 训练多分类
5. 保存预测结果到 Excel
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# =========================
# 配置
# =========================
INPUT_FILE = r"../../../mid_result/training_data/scored_16K_LLM.xlsx"
TEXT_COL = "词语"
LABEL_COL = "分数"
N_CLASS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000   # TF-IDF 最大特征数，可调整

# =========================
# 标签分桶函数
# =========================
def bucketize_labels(y, n_class=9):
    # 确保 y 在 [-1,1] 之间
    y = np.clip(y, -1, 1)

    if n_class == 9:
        bins = [-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
    elif n_class == 5:
        bins = [-1, -0.6, -0.2, 0.2, 0.6, 1.0]
    else:
        raise ValueError("只支持 n_class=9 或 n_class=5")

    # digitize 的结果范围 [1, len(bins)]
    labels = np.digitize(y, bins, right=False) - 1

    # 最后 clip 一次，确保不会越界
    labels = np.clip(labels, 0, len(bins) - 2)
    return labels


# =========================
# 主函数
# =========================
def main():
    # 1. 读取数据
    df = pd.read_excel(INPUT_FILE)
    texts = df[TEXT_COL].astype(str).tolist()
    y_cont = df[LABEL_COL].values

    # 2. 分桶
    y_cls = bucketize_labels(y_cont, N_CLASS)

    # 3. 向量化文本 (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(texts)

    # 4. 划分训练/测试集 (保留索引)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_cls, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls
    )

    # 5. LightGBM 数据集
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    # 6. LightGBM 参数
    params = {
        "objective": "multiclass",
        "num_class": N_CLASS,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    # 7. 训练
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=500,
    )

    # 8. 预测
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 9. 指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 10. 保存预测结果到 Excel
    results = pd.DataFrame({
        "词语": df.loc[idx_test, TEXT_COL].values,
        "y_true": y_test,
        "y_pred": y_pred
    })
    results.to_excel("classification_results_tfidf.xlsx", index=False)

    # 11. 保存模型
    model.save_model("lightgbm_multiclass_tfidf.txt")

if __name__ == "__main__":
    main()
