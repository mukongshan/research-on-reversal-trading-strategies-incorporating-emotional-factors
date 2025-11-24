import os
import pandas as pd

# ===== 配置 =====
SCORED_FILE = "../../../mid_result/hs300_data/merged_all_scored_titles.csv"   # 680W 大文件，含 标题 + 基础分数
ROOT_DIR = "mid_result/hs300_data/stocks_words_data"  # 子目录的根目录
OUTPUT_NAME = "titles_with_score.csv"  # 输出文件名（避免覆盖原始）

# ===== 1. 读大文件，生成映射 =====
print("加载大文件中...")
big_df = pd.read_csv(SCORED_FILE, usecols=["标题", "基础分数"])
score_map = dict(zip(big_df["标题"], big_df["基础分数"]))
print(f"大文件加载完成，共 {len(score_map)} 条映射")

# ===== 2. 遍历子目录，处理小文件 =====
for code in os.listdir(ROOT_DIR):
    subdir = os.path.join(ROOT_DIR, code)
    if not os.path.isdir(subdir):
        continue

    file_path = os.path.join(subdir, "titles_time_fixed.csv")
    if not os.path.exists(file_path):
        continue

    print(f"处理股票 {code} ...")
    try:
        df = pd.read_csv(file_path)
        if "标题" not in df.columns:
            print(f"⚠️ 文件 {file_path} 缺少 '标题' 列，跳过")
            continue

        # 查基础分数
        df["基础分数"] = df["标题"].map(score_map)

        # 输出（存在每个子目录下）
        out_path = os.path.join(subdir, OUTPUT_NAME)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ 已输出 {out_path}")

    except Exception as e:
        print(f"❌ 处理 {file_path} 出错: {e}")

print("全部处理完成！")
