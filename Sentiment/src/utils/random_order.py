import pandas as pd

def shuffle_excel(input_file, output_file):
    # 读取 Excel
    df = pd.read_csv(input_file)

    # 打乱顺序（frac=1 表示打乱所有行，random_state 可固定随机种子保证可复现）
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存为新的 Excel
    df_shuffled.to_csv(output_file, index=False)
    print(f"✅ 已打乱并保存为 {output_file}")

if __name__ == "__main__":
    input_path = r"../../mid_result/training_data/merged_2w_scored_titles.csv"
    output_path = r"../../mid_result/training_data/merged_15K_scored_titles.csv"

    shuffle_excel(input_path, output_path)
