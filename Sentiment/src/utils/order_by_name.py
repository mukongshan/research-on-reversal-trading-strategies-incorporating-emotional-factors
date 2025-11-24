import csv

def sort_csv_by_column(input_file, output_file, sort_column_name, ascending=True):
    """
    按照指定的列名对 CSV 文件进行排序，并保存结果

    参数：
        input_file (str): 输入的 CSV 文件路径
        output_file (str): 输出的 CSV 文件路径
        sort_column_name (str): 用于排序的列名（表头名称，如 "年龄"、"姓名"）
        ascending (bool): 是否升序排序，默认为 True；False 表示降序
    """
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)  # 读取表头

        # 检查排序列名是否存在
        if sort_column_name not in headers:
            print(f"❌ 错误：找不到列名 '{sort_column_name}'。当前表头有：{headers}")
            return

        # 找到排序列的索引
        sort_column_index = headers.index(sort_column_name)

        # 读取所有数据行（包含表头之外的所有行）
        rows = list(reader)

        # 定义排序 key 函数：根据 sort_column_index 列的值排序
        def get_key(row):
            value = row[sort_column_index]
            # 尝试转换为数字排序（如果可能），否则按字符串排序
            try:
                return float(value)  # 支持数字，如年龄、分数等
            except ValueError:
                return value         # 否则按字符串排序（如姓名、地址等）

        # 排序：升序或降序
        reverse_sort = not ascending
        sorted_rows = sorted(rows, key=get_key, reverse=reverse_sort)

        # 写入到输出文件
        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)  # 写入表头
            writer.writerows(sorted_rows)  # 写入排序后的数据行

    order = "升序" if ascending else "降序"
    print(f"✅ 已按照列 '{sort_column_name}' ({order}) 排序，结果已保存到：{output_file}")

# ======================
#  使用
# ======================

# 输入和输出文件路径
input_csv = 'all_scored_titles_with_clear_time.csv'     # 原始 CSV 文件
output_csv = 'all_titles_ordered_by_clear_time.csv'  # 排序后的输出文件（可设为 input_csv 以覆盖，但不推荐一开始就覆盖）

# 指定用于排序的列名，比如 "年龄"、"姓名"、"分数" 等
sort_by_column = '纠正后时间'  # 你可以改成 '姓名' 或其他列

# 是否升序排序？True 为升序（从小到大），False 为降序（从大到小）
sort_ascending = True    # 默认升序

# 调用排序函数
sort_csv_by_column(input_csv, output_csv, sort_by_column, sort_ascending)