import os
import pandas as pd

def split_csv(input_file, output_dir, chunk_size=2000):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 用 pandas 分块读取
    reader = pd.read_csv(input_file, chunksize=chunk_size)

    for i, chunk in enumerate(reader, start=1):
        output_file = os.path.join(output_dir, f"part_{i}.csv")
        chunk.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存 {output_file}, 行数 {len(chunk)}")

    print("🎉 拆分完成！")

if __name__ == "__main__":
    INPUT_FILE = r"../../src/analyze_sentiment_new/train/random_20000_rows.csv"
    OUTPUT_DIR = r"../../mid_result/2w_titles_slice"

    split_csv(INPUT_FILE, OUTPUT_DIR, chunk_size=400)
