#爬取沪深300成分股(2020年1月1日-2025年4月20日)日线数据
import akshare as ak
import pandas as pd
import os
from tqdm import tqdm

# 列名中英对照表 (根据akshare最新接口字段更新)
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

# 获取沪深300成分股
def get_hs300_components():
    df = ak.index_stock_cons_sina(symbol="000300")
    df["code"] = df["symbol"].str[2:]  # 移除交易所前缀(sh/sz)
    return df[["code", "name"]]

# 获取单只股票历史数据
def get_stock_data(code, start_date="20200101", end_date="20250420"):
    try:
        # 获取原始数据
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            adjust="hfq",  # 使用后复权
            start_date=start_date,
            end_date=end_date
        )
        
        # 添加股票代码列
        df["code"] = code
        
        # 列名转换和筛选
        df = df.rename(columns=COLUMN_MAPPING)
        keep_columns = ["date", "open", "high", "low", "close", "volume", "code"]
        df = df[keep_columns]
        
        # 转换日期格式
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    except Exception as e:
        print(f"\n[Error] 获取 {code} 数据失败: {str(e)}")
        return None

# 主程序
if __name__ == "__main__":
    # 创建数据目录
    os.makedirs("stock_data", exist_ok=True)
    
    # 获取成分股列表
    components = get_hs300_components()
    print(f"成功获取 {len(components)} 只沪深300成分股")
    
    # 遍历获取数据
    success_count = 0
    for code, name in tqdm(components[["code", "name"]].values, 
                         desc="下载进度"):
        # 检查文件是否已存在
        if os.path.exists(f"stock_data/{code}.csv"):
            continue
            
        # 获取数据
        df = get_stock_data(code)
        
        if df is not None and not df.empty:
            # 保存为CSV
            df.to_csv(f"stock_data/{code}.csv", index=False)
            success_count += 1
    
    print(f"\n数据获取完成！成功获取 {success_count}/{len(components)} 只股票数据")
    print(f"数据保存路径: {os.path.abspath('stock_data')}")
