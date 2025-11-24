import os
import pandas as pd


def convert_xlsx_to_csv(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~"):  # 忽略临时文件
                xlsx_path = os.path.join(subdir, file)
                csv_path = os.path.join(subdir, "commends.csv")

                try:
                    df = pd.read_excel(xlsx_path)
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    print(f"转换成功: {xlsx_path} -> {csv_path}")
                except Exception as e:
                    print(f"转换失败: {xlsx_path}, 错误: {e}")


if __name__ == "__main__":
    root_directory = r"D:\\All_of_mine\\大学\\项目和比赛\\da_chuang\\res"
    convert_xlsx_to_csv(root_directory)
