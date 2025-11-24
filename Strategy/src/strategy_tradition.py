import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  
from matplotlib.ticker import FuncFormatter
# ----------------------
# 1.数据预处理
# ----------------------
index_data = pd.read_csv("C:/Users/huawei1/Desktop/index_data.csv", parse_dates=['date'])
index_data.set_index('date', inplace=True)

stock_folder = "C:/Users/huawei1/Desktop/stock_data"
all_files = os.listdir(stock_folder)

stock_list = []
for file in all_files:
    code = file.split('.')[0]
    df = pd.read_csv(os.path.join(stock_folder, file), parse_dates=['date'])
    df['code'] = code
    stock_list.append(df)

stock_data = pd.concat(stock_list, ignore_index=True)
stock_data.sort_values(['code', 'date'], inplace=True)
stock_data = stock_data[stock_data['volume'] > 0]

# ----------------------
# 2.回测框架
# ----------------------
def run_strategy(stock_data, lookback=5, holding_period=5, commission=0.0003, slippage=0.0002):
    """
    核心策略函数（含交易成本）
    :param lookback: 形成期（日）
    :param holding_period: 持有期（日）
    :param commission: 单边佣金率
    :param slippage: 单边滑点率
    :return: 策略累计收益序列、绩效指标
    """
    # 计算反转因子
    df = stock_data.copy()
    df['ret'] = df.groupby('code')['close'].pct_change(lookback)
    df['factor'] = df.groupby('date')['ret'].transform(lambda x: (x - x.mean())/x.std())
    
    # 生成调仓日期（每隔holding_period天调仓）
    dates = df['date'].unique()
    rebalance_dates = dates[::holding_period]
    
    # 初始化持仓和收益
    holdings = pd.Series(0, index=df['code'].unique())
    portfolio_value = [1.0]  # 初始净值1
    trade_dates = []
    
    # 遍历调仓日
    for i in tqdm(range(len(rebalance_dates)-1)):
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 获取当前因子值
        factor_data = df[df['date'] == current_date][['code', 'factor']]
        
        # 选择底部10%股票
        selected = factor_data.nsmallest(int(len(factor_data)*0.1), 'factor')['code']
        new_holdings = pd.Series(0, index=holdings.index)
        new_holdings[selected] = 1 / len(selected)  # 等权重
        
        # 计算换仓成本（卖出旧持仓 + 买入新持仓）
        turnover = (holdings - new_holdings).abs().sum()
        cost = turnover * (commission + slippage)
        
        # 计算持有期收益
        period_data = df[(df['date'] >= current_date) & (df['date'] < next_date)]
        period_ret = period_data.groupby('code')['close'].last() / period_data.groupby('code')['close'].first() - 1
        
        # 更新净值（扣除交易成本）
        current_value = portfolio_value[-1]
        portfolio_value.append(
            current_value * (1 + (holdings * period_ret).sum()) - current_value * cost
        )
        trade_dates.append(next_date)
        holdings = new_holdings
    
    # 构建收益序列
    result = pd.Series(portfolio_value, index=[df['date'].min()] + trade_dates)
    result = result.resample('D').ffill().pct_change().fillna(0)
    
    # 计算绩效指标
    # 对齐基准数据
    index_ret = index_data['close'].pct_change().dropna()
    aligned_dates = result.index.intersection(index_ret.index)
    strategy_ret = result.loc[aligned_dates]
    benchmark_ret = index_ret.loc[aligned_dates]
    # 计算复利累计收益
    cum_strategy = (1 + strategy_ret).cumprod()
    cum_benchmark = (1 + benchmark_ret).cumprod()
    excess_cum = cum_strategy - cum_benchmark  # 新增超额累计序列
    # 复利年化收益率
    total_years = len(cum_strategy) / 252
    annual_ret = (cum_strategy.iloc[-1] ** (1 / total_years)) - 1 if total_years > 0 else 0
    # 年化波动率
    annual_vol = strategy_ret.std() * np.sqrt(252)
    # 夏普比率
    sharpe = annual_ret / annual_vol
    # 最大回撤
    max_drawdown = (cum_strategy.cummax() - cum_strategy).max()
    # 超额累计收益率
    excess_cum = cum_strategy.iloc[-1] - cum_benchmark.iloc[-1]
    # 胜率（策略日收益战胜基准的比例）
    win_rate = (strategy_ret > benchmark_ret).mean()
    # 盈亏比
    gains = strategy_ret[strategy_ret > 0]
    losses = strategy_ret[strategy_ret < 0]
    profit_ratio = gains.mean() / losses.abs().mean() if len(losses) > 0 else np.inf
    
    # ========== 新增Alpha/Beta计算 ==========
    if len(strategy_ret) >= 2 and len(benchmark_ret) >= 2:
        # 计算协方差矩阵
        cov_matrix = np.cov(strategy_ret, benchmark_ret, ddof=1)  # 使用样本协方差
        covariance = cov_matrix[0, 1]
        var_benchmark = cov_matrix[1, 1]
    
        beta = covariance / var_benchmark if var_benchmark != 0 else np.nan
        alpha_daily = strategy_ret.mean() - beta * benchmark_ret.mean()
        alpha_annual = alpha_daily * 252  # 年化处理
    else:
        alpha_annual = beta = np.nan
    
    # 修改返回值包含aligned_dates
    return_data = (
        strategy_ret,   # 日收益率 pd.Series
        cum_strategy,   # 策略累计净值 pd.Series
        cum_benchmark,  # 基准累计净值 pd.Series
        aligned_dates   # 新增：对齐后的日期索引
    )
    
    # 构建指标字典
    metrics = {
           'sharpe': sharpe,
           'annual_ret': annual_ret,
           'annual_vol': annual_vol,
           'max_drawdown': max_drawdown,
           'excess_cum_return': excess_cum,
           'win_rate': win_rate,
           'profit_ratio': profit_ratio,
           'alpha': alpha_annual,  
           'beta': beta            
       }

    return return_data, metrics

# ----------------------
# 3.参数网格搜索（添加过滤条件）
# ----------------------
lookback_list = [5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 180, 240]
holding_period_list = [5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 180, 240]
# 生成参数网格时过滤无效组合
param_grid = [
    (lb, hp) 
    for lb in lookback_list 
    for hp in holding_period_list 
    if lb >= hp
]

# 存储优化结果
results = []
best_sharpe = -np.inf
best_params = None
best_cum = None
# 在参数网格搜索中保存aligned_dates
best_aligned_dates = None  # 初始化
for lb, hp in param_grid:
    # 修改接收返回值
    (daily_ret, cum_strategy, cum_benchmark, aligned_dates), metrics = run_strategy(stock_data, lb, hp)
    results.append({
        'lookback': lb,
        'holding': hp,
        'sharpe': metrics['sharpe'],
        'annual_ret': metrics['annual_ret'],
        'annual_vol': metrics['annual_vol'],
        'max_drawdown': metrics['max_drawdown'],
        'excess_cum': metrics['excess_cum_return'],
        'win_rate': metrics['win_rate'],
        'profit_ratio': metrics['profit_ratio'],
        'alpha': metrics['alpha'],
        'beta': metrics['beta']
    })

    
    # 更新最优参数时保存对齐日期
    if metrics['sharpe'] > best_sharpe:
        best_sharpe = metrics['sharpe']
        best_params = (lb, hp)
        best_aligned_dates = aligned_dates  # 保存关键变量
        best_ret = daily_ret
        best_cum = cum_strategy  # 保存最优累计收益
        best_excess = cum_strategy - cum_benchmark  # 保存最优超额收益
        best_cum_benchmark = cum_benchmark

# 转换为DataFrame
result_df = pd.DataFrame(results)

# ----------------------
# 全局绘图样式与配色（蓝色系主调 + 无网格）
# ----------------------
plt.style.use('seaborn-white')  # 无默认网格的基础样式
plt.rcParams.update({
    'font.size': 12,           # 全局字体大小
    'axes.titlesize': 14,      # 标题字体大小（保留参数一致性）
    'axes.labelsize': 12,      # 坐标轴标签字体大小
    'xtick.labelsize': 10,     # X轴刻度字体
    'ytick.labelsize': 10,     # Y轴刻度字体
    'legend.fontsize': 10,     # 图例字体
    'figure.dpi': 300,         # 输出分辨率
    'savefig.dpi': 300,        # 保存分辨率
    # 全局文字改为楷体（中文优先，西文兼容）
    'font.family': ['KaiTi', 'SimKai', 'Times New Roman'],  
    'axes.unicode_minus': False  # 解决负号显示异常
})

# 新配色方案：传统策略「蓝」vs 基准「绿」，正负超额收益保留原对比色，热力图「蓝色渐变」
COLORS = {
    '传统蓝': '#1f77b4',       # 传统策略主色（专业蓝色，高识别度）
    '基准绿': '#2ca02c',       # 沪深300基准主色（自然绿色，与蓝色强区分）
    '正超额绿': '#80cba4',     # 正超额收益填充色（延续原浅绿，保持视觉统一）
    '负超额红': '#d63031',     # 负超额收益填充色（延续原深红，强化正负对比）
    '热力图蓝': 'Blues',       # 夏普比率热力图蓝色系渐变（matplotlib内置蓝渐变）
}


# ----------------------
# 基准指标计算模块（功能不变）
# ----------------------
def calculate_benchmark_metrics(benchmark_returns):
    """计算沪深300基准绩效指标"""
    if benchmark_returns.empty:
        return {k: 0 for k in ['年化收益率', '年化波动率', '夏普比率', '累计收益率', '最大回撤']}
    
    cum_benchmark = (1 + benchmark_returns).cumprod()
    total_return = cum_benchmark.iloc[-1] - 1
    total_days = len(cum_benchmark)
    total_years = total_days / 252
    
    # 年化收益率
    annual_ret = (cum_benchmark.iloc[-1] ** (1/total_years)) - 1 if total_years > 0 else 0
    # 年化波动率
    annual_vol = benchmark_returns.std() * np.sqrt(252)
    # 夏普比率（无风险利率=0）
    sharpe = annual_ret / annual_vol if annual_vol != 0 else np.nan
    # 最大回撤
    rolling_max = cum_benchmark.cummax()
    drawdown = (rolling_max - cum_benchmark) / rolling_max
    max_drawdown = drawdown.max()
    
    return {
        '年化收益率': annual_ret,
        '年化波动率': annual_vol,
        '夏普比率': sharpe,
        '累计收益率': total_return,
        '最大回撤': max_drawdown
    }

# 获取基准收益（对齐策略日期）
benchmark_returns = index_data['close'].pct_change().dropna().loc[best_aligned_dates]
benchmark_metrics = calculate_benchmark_metrics(benchmark_returns)

# 最优策略指标
best_strategy_metrics = result_df.loc[result_df['sharpe'].idxmax()]

# 构建对比表（功能不变）
comparison_data = {
    '指标': ['年化收益率', '年化波动率', '夏普比率', '累计收益率', '最大回撤'],
    '传统反转策略': [
        best_strategy_metrics['annual_ret'],
        best_strategy_metrics['annual_vol'],
        best_strategy_metrics['sharpe'],
        best_strategy_metrics['excess_cum'] + benchmark_metrics['累计收益率'],
        best_strategy_metrics['max_drawdown']
    ],
    '沪深300基准': [
        benchmark_metrics['年化收益率'],
        benchmark_metrics['年化波动率'],
        benchmark_metrics['夏普比率'],
        benchmark_metrics['累计收益率'],
        benchmark_metrics['最大回撤']
    ]
}

# 格式化并保存对比表（功能不变）
comparison_df = pd.DataFrame(comparison_data)
comparison_df['传统反转策略'] = comparison_df['传统反转策略'].apply(
    lambda x: f"{x:.2%}" if isinstance(x, (float, int)) else x)
comparison_df['沪深300基准'] = comparison_df['沪深300基准'].apply(
    lambda x: f"{x:.2%}" if isinstance(x, (float, int)) else x)

comparison_path = "策略与基准对比.csv"
comparison_df.to_csv(comparison_path, index=False, encoding='utf_8_sig')
print(f"\n策略与基准对比文件已生成：{comparison_path}")
print(comparison_df.to_markdown(index=False))


# ----------------------
# （1）夏普比率热力图（无标题 + 蓝色渐变 + 楷体）
# ----------------------
plt.figure(figsize=(10, 8))
heatmap_data = result_df.pivot(index='lookback', columns='holding', values='sharpe')
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f",
    cmap=COLORS['热力图蓝'],  # 改用蓝色渐变
    annot_kws={"size": 9},    # 标注文字大小（楷体由全局设置生效）
    cbar_kws={'label': '夏普比率'}  # 色条标签（楷体生效）
)
plt.xlabel('持有期（天）', fontweight='bold')  # X轴标签（楷体 + 加粗）
plt.ylabel('形成期（天）', fontweight='bold')  # Y轴标签（楷体 + 加粗）
plt.savefig('sharpe_heatmap.png', bbox_inches='tight')  # 无标题、无网格


# ----------------------
# （2）最优策略收益对比（无标题 + 蓝绿主色 + 图例左置）
# ----------------------
plt.figure(figsize=(12, 6))
# 计算累计收益
strategy_cum = (1 + best_ret).cumprod() - 1
benchmark_cum = best_cum_benchmark - 1

ax1 = plt.gca()
# 绘制传统策略曲线（改蓝色）
line_strategy, = ax1.plot(
    strategy_cum, 
    label=f'传统反转策略（形成期={best_params[0]}, 持有期={best_params[1]}）', 
    color=COLORS['传统蓝'], 
    linewidth=2
)
# 绘制基准曲线（改绿色）
line_benchmark, = ax1.plot(
    benchmark_cum, 
    label='沪深300基准', 
    color=COLORS['基准绿'], 
    linewidth=2
)

ax1.set_ylabel('累计收益率', fontweight='bold')  # Y轴标签（楷体 + 加粗）
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))  # 百分比格式（楷体生效）
ax1.grid(False)  # 强制无网格线
ax1.legend(loc='upper left', frameon=True)  # 图例左置（楷体生效）
plt.savefig('strategy_comparison.png', bbox_inches='tight')  # 无标题


# ----------------------
# （3）超额收益曲线（无标题 + 蓝绿主色 + 图例左置）
# ----------------------
plt.figure(figsize=(12, 6))
# 计算超额收益
excess_return = strategy_cum - benchmark_cum

# 主Y轴：策略与基准收益（改蓝绿主色）
ax2 = plt.gca()
ax2.plot(strategy_cum, color=COLORS['传统蓝'], linewidth=2, label='传统反转策略')
ax2.plot(benchmark_cum, color=COLORS['基准绿'], linewidth=2, label='沪深300基准')
ax2.set_ylabel('累计收益率', fontweight='bold')  # 主Y轴标签（楷体 + 加粗）
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))  # 百分比格式（楷体生效）
ax2.grid(False)  # 主Y轴无网格线

# 次Y轴：超额收益填充（保留原正负色）
ax3 = ax2.twinx()
ax3.fill_between(
    excess_return.index, 0, excess_return.values, 
    where=excess_return >= 0, 
    facecolor=COLORS['正超额绿'], alpha=0.3, interpolate=True
)
ax3.fill_between(
    excess_return.index, 0, excess_return.values,
    where=excess_return < 0,
    facecolor=COLORS['负超额红'], alpha=0.3, interpolate=True
)
ax3.set_ylabel('超额收益率', fontweight='bold')  # 次Y轴标签（楷体 + 加粗）
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))  # 百分比格式（楷体生效）
ax3.grid(False)  # 次Y轴无网格线

# 标注最大超额收益（文字为楷体）
max_excess_date = excess_return.idxmax()
plt.annotate(
    f'最大超额收益: {excess_return.max():.1%}',
    xy=(max_excess_date, excess_return.max()),
    xytext=(max_excess_date, excess_return.max() + 0.1),
    arrowprops=dict(arrowstyle="->", color=COLORS['负超额红']),
    ha='center'  # 文字居中（楷体由全局设置生效）
)

# 统一图例并左置（楷体生效）
lines, labels = ax2.get_legend_handles_labels()
patches = [
    plt.Rectangle((0,0), 1, 1, fc=COLORS['正超额绿'], alpha=0.3),
    plt.Rectangle((0,0), 1, 1, fc=COLORS['负超额红'], alpha=0.3)
]
ax2.legend(lines + patches, labels + ['正超额收益', '负超额收益'], loc='upper left', frameon=True)
plt.savefig('excess_return.png', bbox_inches='tight')  # 无标题


# ----------------------
# 保存策略绩效结果（功能不变）
# ----------------------
sorted_results = result_df.sort_values('sharpe', ascending=False).reset_index(drop=True)
sorted_results['排名'] = sorted_results.index + 1

output_columns = [
    '排名', 'lookback', 'holding', 'sharpe', 
    'annual_ret', 'annual_vol', 'max_drawdown',
    'excess_cum', 'win_rate', 'profit_ratio',
    'alpha', 'beta'
]
formatted_results = sorted_results[output_columns].round(4)
output_path = "策略绩效结果.csv"
formatted_results.to_csv(output_path, index=False, encoding='utf-8-sig')

# 输出最优参数（文字为楷体，终端显示依赖系统字体支持）
print(f"最优参数组合：形成期={best_params[0]}天，持有期={best_params[1]}天")
print(result_df.sort_values('sharpe', ascending=False).head())