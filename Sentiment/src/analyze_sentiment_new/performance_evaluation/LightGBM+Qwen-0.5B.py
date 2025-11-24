"""
train_lightgbm_with_embeddings.py

功能：
1. 加载 embedding 数据 (CSV)
2. 可选 PCA 降维
3. 使用 LightGBM 训练回归模型
4. 输出 R² / RMSE / MAE
5. 保存预测结果
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================
# 配置
# =========================
INPUT_FILE = "../../../mid_result/training_data/test_scored_16K_QwenEmbeddings.csv"  # 输入 embedding CSV
OUTPUT_FILE = "../train/lightgbm_predictions.csv"  # 输出预测结果 CSV
USE_PCA = True
PCA_DIM = 300
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =========================
# Step 1: 加载数据
# =========================
def load_data(input_file):
    df = pd.read_csv(input_file)
    X = df[[col for col in df.columns if col.startswith("emb_")]]
    y = df["分数"]
    print(f"📥 数据加载完成: {X.shape[0]} 行, {X.shape[1]} 维")
    return df, X, y


# =========================
# Step 2: 降维 (可选)
# =========================
def reduce_dimensionality(X, use_pca=True, dim=200, random_state=42):
    if use_pca:
        pca = PCA(n_components=dim, random_state=random_state)
        X_reduced = pca.fit_transform(X)
        print(f"📉 已降维: {X.shape[1]} -> {X_reduced.shape[1]}")
        return X_reduced, pca
    else:
        print("➡️ 跳过 PCA 降维")
        return X, None


# =========================
# Step 3: 划分数据集
# =========================
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =========================
# Step 4: 训练 LightGBM
# =========================
def train_lightgbm(X_train, y_train, random_state=42):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    print("⏳ 正在训练 LightGBM...")
    model.fit(X_train, y_train)
    print("✅ 训练完成")
    return model


# =========================
# Step 5: 评估模型
# =========================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # 兼容旧版
    mae = mean_absolute_error(y_test, y_pred)

    print("\n📊 模型表现：")
    print(f"R²     = {r2:.4f}")
    print(f"RMSE   = {rmse:.4f}")
    print(f"MAE    = {mae:.4f}")
    return y_pred, r2, rmse, mae



# =========================
# Step 6: 保存预测结果
# =========================
def save_results(y_test, y_pred, output_file):
    df_out = pd.DataFrame({
        "真实分数": y_test,
        "预测分数": y_pred
    })
    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"💾 已保存预测结果到 {output_file}")


# =========================
# 主流程
# =========================
def main():
    # Step 1: 加载数据
    df, X, y = load_data(INPUT_FILE)

    # Step 2: 降维
    X, pca_model = reduce_dimensionality(X, use_pca=USE_PCA, dim=PCA_DIM, random_state=RANDOM_STATE)

    # Step 3: 划分数据集
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Step 4: 训练模型
    model = train_lightgbm(X_train, y_train, random_state=RANDOM_STATE)

    # Step 5: 评估模型
    y_pred, r2, rmse, mae = evaluate_model(model, X_test, y_test)

    # Step 6: 保存结果
    save_results(y_test, y_pred, OUTPUT_FILE)


if __name__ == "__main__":
    main()
