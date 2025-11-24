# -*- coding: utf-8 -*-
"""
功能：
1. 读取 CSV 文件；
2. 修改指定列名；
3. 保存结果（可选择覆盖原文件或另存为新文件）。
"""

import pandas as pd
import os

# ======= ⚙️ 参数设置 =======
INPUT_FILE = "../../mid_result/hs300_data/single_stock/titles_with_weighted_score.csv"   # 输入文件路径
OUTPUT_FILE = "../../mid_result/hs300_data/single_stock/titles_with_weighted_score.csv"  # 输出文件路径（若想覆盖原文件，可与INPUT_FILE相同）

# 旧列名与新列名的映射关系
RENAME_MAP = {
    "纠正后时间": "时间"
}

# ======= 🚀 主逻辑 =======
def rename_csv_columns(input_path, output_path, rename_map):
    # 1. 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 找不到文件：{input_path}")
        return

    # 2. 读取CSV
    df = pd.read_csv(input_path)

    # 3. 执行重命名
    df = df.rename(columns=rename_map)

    # 4. 保存结果
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 列名修改完成！保存至：{output_path}")

# ======= 🧠 执行 =======
if __name__ == "__main__":
    rename_csv_columns(INPUT_FILE, OUTPUT_FILE, RENAME_MAP)
