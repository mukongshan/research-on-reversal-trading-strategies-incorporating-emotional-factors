# -*- coding: utf-8 -*-
"""
在每个股票子目录中修复时间（倒序数据）。
判定规则：
- 前 30 条至少 20 条是 1 月
- 后 30 条至少 25 条是 12 月
"""

import os
import pandas as pd


def restore_years_in_df(df, start_year=2025, end_year=2020,
                        pre_window=50, post_window=50,
                        pre_threshold=35, post_threshold=45,
                        lock_span=80):
    """修复 DataFrame 中的 '最后更新时间'，生成 '纠正后时间' 列"""
    date_str = df["最后更新时间"].astype(str).str.split().str[0]
    months = date_str.str.split("-").str[0].astype(int).tolist()
    days = date_str.str.split("-").str[1].astype(int).tolist()

    year = start_year
    years = []
    lock_counter = 0
    n = len(months)

    for i in range(n):
        if i > 0:
            if lock_counter > 0:
                lock_counter -= 1
            else:
                # ===== 倒序数据的跨年趋势 =====
                if months[i - 1] == 1 and months[i] == 12:
                    pre_months = months[max(0, i - pre_window):i]
                    post_months = months[i:min(n, i + post_window)]

                    if pre_months.count(1) >= pre_threshold and post_months.count(12) >= post_threshold:
                        year -= 1
                        if year < end_year:
                            year = start_year
                        lock_counter = lock_span  # 锁定 N 行

        years.append(year)

    df["纠正后时间"] = pd.to_datetime(
        [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)],
        format="%Y-%m-%d",
        errors="coerce"
    )

    return df


def process_all_stocks(root_dir="mid_result/hs300_data/stocks_words_data/"):
    """遍历所有子目录，修复时间并保存 CSV"""
    for stock_code in os.listdir(root_dir):
        stock_dir = os.path.join(root_dir, stock_code)
        if not os.path.isdir(stock_dir):
            continue

        file_path = os.path.join(stock_dir, "股吧评论分词结果.xlsx")
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_excel(file_path)
            df = df[["标题", "阅读量", "评论", "最后更新时间"]].copy()

            df_fixed = restore_years_in_df(df)

            save_path = os.path.join(stock_dir, "titles_time_fixed.csv")
            df_fixed.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"✅ 已处理 {stock_code}, 输出: {save_path}")
        except Exception as e:
            print(f"❌ 处理 {stock_code} 失败: {e}")


if __name__ == "__main__":
    process_all_stocks("../../mid_result/hs300_data/stocks_words_data")
