import pandas as pd

def xlsx_to_csv(input_file, output_file):
    # 读取 Excel
    df = pd.read_excel(input_file)

    # 保存为 CSV（utf-8-sig 避免中文乱码）
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ 已转换: {input_file} → {output_file}")

if __name__ == "__main__":
    input_path = r"../../res/hs300_data/hs300股吧.xlsx"
    output_path = r"../../res/hs300_data/hs300_forum.csv"

    xlsx_to_csv(input_path, output_path)
