import csv

def remove_multiple_csv_columns_by_name(input_file, output_file, columns_to_remove):
    """
    根据列名列表，从 CSV 文件中删除多个指定列

    参数：
        input_file (str): 输入的 CSV 文件路径
        output_file (str): 输出的 CSV 文件路径
        columns_to_remove (list of str): 要删除的列名列表，如 ['年龄', '性别']
    """
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)  # 读取表头

        # 检查要删除的列是否都存在于表头中
        missing_columns = [col for col in columns_to_remove if col not in headers]
        if missing_columns:
            print(f"❌ 错误：以下列名在 CSV 文件中不存在：{missing_columns}")
            print(f"    当前 CSV 表头为：{headers}")
            return

        # 找出所有要删除的列的索引
        columns_indexes_to_remove = [headers.index(col) for col in columns_to_remove]

        # 构建新的表头：只保留不在删除列表中的列
        new_headers = [
            header for idx, header in enumerate(headers)
            if idx not in columns_indexes_to_remove
        ]

        # 写入到输出文件
        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(new_headers)  # 写入新表头

            # 遍历每一行数据，删除指定索引的列
            for row in reader:
                new_row = [
                    value for idx, value in enumerate(row)
                    if idx not in columns_indexes_to_remove
                ]
                writer.writerow(new_row)

    print(f"✅ 已成功删除列 {columns_to_remove}，结果已保存到：{output_file}")

# ======================
# 使用
# ======================

# 输入和输出文件路径
input_csv = 'all_scored_titles_with_fixed_time.csv'      # 替换为你的输入 CSV 文件路径
output_csv = 'all_scored_titles_with_fixed_time_2.csv'    # 输出的新文件路径，也可以设置为 input_csv 来覆盖原文件

# 指定要删除的多个列名（必须与表头完全一致，包括中文和空格）
columns_to_remove = ['最后更新时间', '股票代码']  # 你可以自由添加或删除列名，比如再加上 '电话', '地址' 等

# 调用函数删除多个列
remove_multiple_csv_columns_by_name(input_csv, output_csv, columns_to_remove)