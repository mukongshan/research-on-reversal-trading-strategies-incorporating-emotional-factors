import os
import pandas as pd

# ===== 配置 =====
ROOT_DIR = "../../mid_result/hs300_data/stocks_words_data"
INPUT_NAME = "titles_with_score.csv"   # 每个子目录的目标文件
OUTPUT_FILE = "all_scored_titles_with_fixed_time.csv"

# ===== 1. 遍历并读取 =====
all_dfs = []

for code in os.listdir(ROOT_DIR):
    subdir = os.path.join(ROOT_DIR, code)
    file_path = os.path.join(subdir, INPUT_NAME)

    if not os.path.isfile(file_path):
        continue

    try:
        df = pd.read_csv(file_path)
        df["股票代码"] = code  # 加股票代码列
        all_dfs.append(df)
        print(f"读取完成: {file_path}, 行数 {len(df)}")
    except Exception as e:
        print(f"❌ 读取 {file_path} 出错: {e}")

# ===== 2. 合并 =====
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ 合并完成，总行数 {len(merged_df)}，已保存到 {OUTPUT_FILE}")
else:
    print("⚠️ 没有找到任何可合并的文件")
