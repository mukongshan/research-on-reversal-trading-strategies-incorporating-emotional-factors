import csv

# 替换为你的CSV文件路径
file_path = '../../mid_result/hs300_data/single_stock/all_scored_titles_with_cut_time.csv'

row_count = 0

with open(file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        row_count += 1

print(f"CSV 文件共有 {row_count} 行")