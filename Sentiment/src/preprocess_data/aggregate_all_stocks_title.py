import os
import pandas as pd

def merge_stocks_data(root_dir, output_file):
    all_data = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "股吧评论分词结果.xlsx":
                file_path = os.path.join(dirpath, file)
                try:
                    df = pd.read_excel(file_path, usecols=["标题", "阅读量", "评论", "最后更新时间"])
                    all_data.append(df)
                    print(f"读取成功: {file_path}, 行数: {len(df)}")
                except Exception as e:
                    print(f"读取失败: {file_path}, 错误: {e}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✅ 数据整合完成，共 {len(merged_df)} 行，已保存到 {output_file}")
    else:
        print("⚠ 没有找到任何可用的数据文件。")

if __name__ == "__main__":
    root_directory = r"D:\All_of_mine\大学\比赛\引入情绪的反转交易策略\mid_result\stocks_words_data"
    output_path = r"stocks_all_titles_data.csv"
    merge_stocks_data(root_directory, output_path)
