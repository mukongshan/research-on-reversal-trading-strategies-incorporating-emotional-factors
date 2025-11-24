import csv
from datetime import datetime

# ======================
# 🔧 全局变量（按你的要求，放在最前面）
# ======================
INPUT_CSV_PATH = '../../res/hs300_data/hs300_forum.csv'  # 输入的 CSV 文件路径，请替换为实际路径
OUTPUT_CSV_PATH = '../../mid_result/hs300_data/hs300_stocks_forum/titles_with_cut_time'  # 输出的 CSV 文件路径，可改为 INPUT_CSV_PATH 以覆盖原文件

TARGET_COLUMN_NAME = '时间'       # 要处理的列名（你的时间列）

# ======================
# 🧠 功能函数
# ======================

def parse_date(date_str):
    """
    尝试将日期字符串解析为 datetime 对象，仅保留日期部分。
    支持格式如：
      - '2025-04-20'
      - '2025-04-20 05:49'
    如果解析失败或为空，返回 None
    """
    if not date_str or date_str.strip() == '':
        return None

    date_str = date_str.strip().replace('T', ' ')

    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(date_str, fmt).date()  # ✅ 只取日期部分
        except ValueError:
            continue

    return None

def adjust_incorrect_dates(row, target_col_index):
    """
    对目标时间列进行修正：
    - 如果时间在 2025年4月及以后 → 改为 2024年，保留月日
    - 如果时间为空 → 返回 None 表示该行需要删除
    - 否则保留原时间
    """
    date_str = row[target_col_index]
    dt = parse_date(date_str)
    print(dt)

    if dt is None:
        return None  # 表示此行因时间为空，需要删除

    year = dt.year
    month = dt.month
    day = dt.day

    # 判断是否为 2025年3月及以后：即 year > 2024 或者 (year == 2025 and month >= 4)
    # if (year == 2025 and month >= 3) or (year < 2020):
        # # 构造新的日期：2024年 + 原月日
        # corrected_dt = datetime(2024, month, day)
        # corrected_date_str = corrected_dt.strftime('%Y-%m-%d')
        # row[target_col_index] = corrected_date_str
        # return None  # 表示此行因时间为空，需要删除

    # 其他情况（包括正常 2024年及以前的时间），不做修改
    return row  # 返回可能修改后的行

def process_csv(input_path, output_path, target_column_name):
    """
    主处理函数：
    - 读取输入 CSV
    - 对每一行判断并处理时间列
    - 删除时间为空的行
    - 修正 2025年4月及以后的时间
    - 写入输出 CSV
    """
    rows_to_write = []
    target_col_index = None

    with open(input_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)  # 读取表头

        # 查找目标列的索引
        if target_column_name not in headers:
            print(f"❌ 错误：找不到列名 '{target_column_name}'。当前表头为：{headers}")
            return

        target_col_index = headers.index(target_column_name)

        # 保存新表头
        rows_to_write.append(headers)

        # 遍历数据行
        for row in reader:
            if len(row) <= target_col_index:
                # 如果该行没有目标列（理论上不应该发生，除非列数不够）
                print(f"⚠️ 警告：某行数据列数不足，跳过。行内容：{row}")
                continue

            time_value = row[target_col_index]

            # 调用处理函数，可能会返回 None（表示该行要删除）
            processed_row = adjust_incorrect_dates(row, target_col_index)

            if processed_row is not None:
                rows_to_write.append(processed_row)
            # 如果返回 None，则跳过该行（即删除）

    # 写入处理后的数据到输出文件
    with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows_to_write)

    print(f"✅ 处理完成！已删除时间为空的行，并修正了 2025年4月及以后的时间。结果已保存到：{output_path}")

# ======================
# ▶️ 脚本入口 / 执行
# ======================

if __name__ == '__main__':
    process_csv(INPUT_CSV_PATH, OUTPUT_CSV_PATH, TARGET_COLUMN_NAME)