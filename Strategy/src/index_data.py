#爬取沪深300指数(2020年1月1日-2025年4月20日)日线数据
import akshare as ak

COLUMN_MAPPING = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover_rate"
}

def get_ak_index(symbol, start, end):
    df = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end)
    return df

df = get_ak_index('000300', '20200101', '20250420')
df = df.rename(columns=COLUMN_MAPPING)
keep_columns = ["date", "open", "high", "low", "close", "volume"]
index_data = df[keep_columns]
index_data.to_csv('index_data.csv', index=False)
