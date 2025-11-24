import os
import pandas as pd

def merge_csv_files(input_dir, output_file):
    """
    合并文件夹下的所有 CSV 文件
    :param input_dir: 输入文件夹
    :param output_file: 输出 CSV 文件路径
    """
    all_dfs = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            print(f"📥 正在读取: {file_path}")
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            all_dfs.append(df)

    if not all_dfs:
        print("⚠️ 没有找到 CSV 文件")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ 合并完成，保存到 {output_file}, 总行数 {len(merged_df)}")


# 使用示例
if __name__ == "__main__":
    input_dir = r"../../mid_result/hs300_data/680W_scored_titles"    # 替换为目标文件夹
    output_file = r"../../mid_result/hs300_data/merged_all_scored_titles.csv"
    merge_csv_files(input_dir, output_file)
