# 情绪作为权重调节器：综合因子 = 反转因子 × (1 + 归一化情绪得分)  
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

sentiment = pd.read_csv("C:/Users/huawei1/Desktop/sentiment_data1.csv", parse_dates=['date'])
trading_dates = index_data.index.unique()
sentiment = sentiment.set_index('date').reindex(trading_dates, method='ffill').reset_index()
stock_data = pd.merge(stock_data, sentiment, on='date', how='left')

# ----------------------
# 2.回测框架
# ----------------------
def run_strategy(stock_data, lookback=5, holding_period=5, commission=0.0003, slippage=0.0002, use_sentiment=False):
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
   
    # 情绪增强因子（新增部分）
    if use_sentiment:
        df['factor'] = df['factor'] * (1 + df['final_score'])  # 情绪调整
        df['factor'] = df.groupby('date')['factor'].transform(
            lambda x: (x - x.mean())/x.std())  # 重新标准化
    
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
    
    return_data = (
        strategy_ret,  # 日收益率 pd.Series
        cum_strategy,  # 策略累计净值 pd.Series
        cum_benchmark  # 基准累计净值 pd.Series
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
param_grid = [
    (lb, hp, use_sentiment) 
    for lb in lookback_list 
    for hp in holding_period_list 
    if lb >= hp
    for use_sentiment in [False, True]  # 同时测试纯反转和情绪增强
]

results = []
best_sharpe = -np.inf
best_params = None

for lb, hp, use_sentiment in param_grid:
    (daily_ret, cum_strategy, cum_benchmark), metrics = run_strategy(
        stock_data, lb, hp, use_sentiment=use_sentiment)
    
    results.append({
        'lookback': lb,
        'holding': hp,
        'use_sentiment': use_sentiment,  # 新增字段
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

    if metrics['sharpe'] > best_sharpe:
        best_sharpe = metrics['sharpe']
        best_params = (lb, hp, use_sentiment)

# 转换为DataFrame
result_df = pd.DataFrame(results)

# ----------------------
# 全局绘图样式与配色（专业蓝+红色对比系）
# ----------------------
plt.style.use('seaborn-white')
plt.rcParams.update({
    'font.size': 12,           # 全局字体大小
    'axes.titlesize': 14,      # 标题字体大小
    'axes.labelsize': 12,      # 坐标轴标签字体大小
    'xtick.labelsize': 10,     # X轴刻度字体
    'ytick.labelsize': 10,     # Y轴刻度字体
    'legend.fontsize': 10,     # 图例字体
    'figure.dpi': 300,         # 输出分辨率
    'savefig.dpi': 300,        # 保存分辨率
    'font.family': ['KaiTi', 'SimKai', 'sans-serif'],  # 全局楷体
    'axes.unicode_minus': False  # 解决负号显示异常
})

# 配色体系（更新超额收益为标准蓝和红）
COLORS = {
    '传统蓝主色': '#1f77b4',    # 传统策略主色（专业蓝）
    '传统蓝标准': '#1e88e5',    # 传统策略超额收益标准蓝
    '基准自然绿': '#2ca02c',    # 沪深300基准色（自然绿）
    '中性浅灰': '#f0f0f0',     # 背景/填充中性色
    '增强红主色': '#d63031',    # 情绪增强策略主色
    '增强红标准': '#e53935',    # 情绪增强超额收益标准红
}


# ----------------------
# 4. 结果可视化
# ----------------------

# ----------------------
# （1）夏普比率热力图
# ----------------------
baseline_df = result_df[result_df['use_sentiment'] == False]  # 传统反转策略
enhanced_df = result_df[result_df['use_sentiment'] == True]   # 情绪增强策略

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# 子图1：传统反转策略（蓝系热力图）
sns.heatmap(
    baseline_df.pivot(index='lookback', columns='holding', values='sharpe'), 
    ax=axes[0], cmap='Blues', annot=True, fmt=".2f",
    cbar_kws={'label': '夏普比率'},
    linewidths=.5
)
axes[0].set_xlabel('持有期（天）', fontweight='bold')
axes[0].set_ylabel('形成期（天）', fontweight='bold')
axes[0].text(0.5, -0.15, '(a) 传统反转策略的夏普比率分布', 
             transform=axes[0].transAxes, ha='center', fontsize=12, fontweight='bold')
axes[0].grid(False)

# 子图2：情绪增强策略（红系热力图）
sns.heatmap(
    enhanced_df.pivot(index='lookback', columns='holding', values='sharpe'), 
    ax=axes[1], cmap='Reds', annot=True, fmt=".2f",
    cbar_kws={'label': '夏普比率'},
    linewidths=.5
)
axes[1].set_xlabel('持有期（天）', fontweight='bold')
axes[1].text(0.5, -0.15, '(b) 情绪增强型反转策略的夏普比率分布', 
             transform=axes[1].transAxes, ha='center', fontsize=12, fontweight='bold')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('夏普比率热力图对比.png', bbox_inches='tight')


# ----------------------
# （2）最优策略收益对比
# ----------------------
best_baseline = baseline_df.sort_values('sharpe', ascending=False).iloc[0]
best_enhanced = enhanced_df.sort_values('sharpe', ascending=False).iloc[0]

(_, cum_baseline, benchmark), _ = run_strategy(
    stock_data, best_baseline['lookback'], best_baseline['holding'], use_sentiment=False)
(_, cum_enhanced, _), _ = run_strategy(
    stock_data, best_enhanced['lookback'], best_enhanced['holding'], use_sentiment=True)

plt.figure(figsize=(12, 6))
plt.plot(cum_baseline, 
         label=f'传统反转（形成期={best_baseline["lookback"]}天，持有期={best_baseline["holding"]}天）', 
         linewidth=2, color=COLORS['传统蓝主色'])
plt.plot(cum_enhanced, 
         label=f'情绪增强（形成期={best_enhanced["lookback"]}天，持有期={best_enhanced["holding"]}天）', 
         linewidth=2, color=COLORS['增强红主色'])
plt.plot(benchmark, label='沪深300指数', linewidth=1.5, color=COLORS['基准自然绿'])

plt.legend(loc='upper left', frameon=True)
plt.ylabel('累计收益率', fontweight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
plt.gca().grid(False)
plt.tight_layout()
plt.savefig('最优策略收益对比.png', bbox_inches='tight')


# ----------------------
# （3）超额收益对比（使用标准蓝和红）
# ----------------------
(_, cum_baseline, cum_benchmark_baseline), _ = run_strategy(
    stock_data, best_baseline['lookback'], best_baseline['holding'], use_sentiment=False)
(_, cum_enhanced, cum_benchmark_enhanced), _ = run_strategy(
    stock_data, best_enhanced['lookback'], best_enhanced['holding'], use_sentiment=True)

excess_baseline = cum_baseline - cum_benchmark_baseline
excess_enhanced = cum_enhanced - cum_benchmark_enhanced

plt.figure(figsize=(12, 6))
# 传统策略超额收益使用标准蓝
plt.plot(
    excess_baseline, 
    label=f'传统反转（形成期={best_baseline["lookback"]}天，持有期={best_baseline["holding"]}天）',
    color=COLORS['传统蓝标准'],  # 标准蓝色
    linewidth=2, 
    alpha=0.9  # 稍提高透明度确保清晰
)
# 情绪增强策略超额收益使用标准红
plt.plot(
    excess_enhanced, 
    label=f'情绪增强（形成期={best_enhanced["lookback"]}天，持有期={best_enhanced["holding"]}天）',
    color=COLORS['增强红标准'],  # 标准红色
    linewidth=2, 
    alpha=0.9
)

# 标注最大超额收益点
max_excess_date = excess_enhanced.idxmax()
plt.annotate(
    f'最大超额收益: {excess_enhanced.max():.1%}',
    xy=(max_excess_date, excess_enhanced.max()),
    xytext=(max_excess_date, excess_enhanced.max() + 0.1),
    arrowprops=dict(arrowstyle="->", color=COLORS['增强红标准']),  # 箭头同步使用标准红
    ha='center',
    fontsize=11
)

plt.ylabel('超额收益率（策略 - 基准）', fontweight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
plt.legend(loc='upper left', frameon=True)  # 图例保持左侧
plt.gca().grid(False)  # 无网格线
plt.tight_layout()
plt.savefig('超额收益对比.png', bbox_inches='tight', dpi=300)


# ----------------------
# 5. 保存策略表现结果
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
output_path = "策略表现结果.csv"
formatted_results.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"最优传统策略参数：形成期={best_baseline['lookback']}天，持有期={best_baseline['holding']}天")
print(f"最优情绪增强策略参数：形成期={best_enhanced['lookback']}天，持有期={best_enhanced['holding']}天")
print(result_df.sort_values('sharpe', ascending=False).head())


# ----------------------
# 6. 最优策略绩效对比
# ----------------------
best_baseline = baseline_df.sort_values('sharpe', ascending=False).iloc[0]
best_enhanced = enhanced_df.sort_values('sharpe', ascending=False).iloc[0]

metrics_compare = pd.DataFrame({
    '指标': ['夏普比率', '年化收益率', '年化波动率', '最大回撤', 
             '累计超额收益', '胜率', '盈亏比', 'Alpha', 'Beta'],
    '传统策略': [
        best_baseline['sharpe'],
        best_baseline['annual_ret'],
        best_baseline['annual_vol'],
        best_baseline['max_drawdown'],
        best_baseline['excess_cum'],
        best_baseline['win_rate'],
        best_baseline['profit_ratio'],
        best_baseline['alpha'],
        best_baseline['beta']
    ],
    '情绪增强策略': [
        best_enhanced['sharpe'],
        best_enhanced['annual_ret'],
        best_enhanced['annual_vol'],
        best_enhanced['max_drawdown'],
        best_enhanced['excess_cum'],
        best_enhanced['win_rate'],
        best_enhanced['profit_ratio'],
        best_enhanced['alpha'],
        best_enhanced['beta']
    ]
})


# ----------------------
# 绩效对比可视化
# ----------------------

# （1）关键指标对比表格
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(
    cellText=np.round(metrics_compare.set_index('指标').values, 4),
    rowLabels=metrics_compare['指标'],
    colLabels=['传统策略', '情绪增强策略'],
    cellLoc='center',
    loc='center',
    colColours=[COLORS['传统蓝标准'], COLORS['增强红标准']]  # 表格列颜色同步更新
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.tight_layout()
plt.savefig('绩效指标对比表.png', dpi=300, bbox_inches='tight')

# （2）雷达图：多维能力对比
def create_radar_chart(categories, values1, values2, title):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    values1 += values1[:1]
    values2 += values2[:1]
    angles += angles[:1]
    
    ax.plot(angles, values1, 'o-', linewidth=2, color=COLORS['传统蓝标准'], label='传统策略')
    ax.fill(angles, values1, color=COLORS['传统蓝标准'], alpha=0.25)
    
    ax.plot(angles, values2, 'o-', linewidth=2, color=COLORS['增强红标准'], label='情绪增强策略')
    ax.fill(angles, values2, color=COLORS['增强红标准'], alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_yticklabels([])
    plt.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1), frameon=True)
    return fig

radar_metrics = metrics_compare[metrics_compare['指标'].isin(
    ['夏普比率', '年化收益率', '累计超额收益', '胜率', '盈亏比', 'Alpha', 'Beta']
)].copy()

positive_metrics = ['夏普比率', '年化收益率', '累计超额收益', '胜率', '盈亏比', 'Alpha']
negative_metrics = ['Beta']

for col in ['传统策略', '情绪增强策略']:
    pos_mask = radar_metrics['指标'].isin(positive_metrics)
    radar_metrics.loc[pos_mask, col] = (
        radar_metrics.loc[pos_mask, col] - radar_metrics.loc[pos_mask, col].min()
    ) / (
        radar_metrics.loc[pos_mask, col].max() - radar_metrics.loc[pos_mask, col].min()
    )
    neg_mask = radar_metrics['指标'].isin(negative_metrics)
    if neg_mask.any():
        radar_metrics.loc[neg_mask, col] = 1 - (
            radar_metrics.loc[neg_mask, col] - radar_metrics.loc[neg_mask, col].min()
        ) / (
            radar_metrics.loc[neg_mask, col].max() - radar_metrics.loc[neg_mask, col].min()
        )

radar_fig = create_radar_chart(
    categories=radar_metrics['指标'].tolist(),
    values1=radar_metrics['传统策略'].tolist(),
    values2=radar_metrics['情绪增强策略'].tolist(),
    title='策略多维能力对比'
)
radar_fig.tight_layout()
radar_fig.savefig('策略多维能力雷达图.png', dpi=300, bbox_inches='tight')

# （3）柱状图：核心指标对比
melt_df = metrics_compare.melt(id_vars='指标', var_name='策略类型', value_name='数值')

plt.figure(figsize=(12, 6))
sns.barplot(
    x='指标', 
    y='数值', 
    hue='策略类型',
    data=melt_df[melt_df['指标'].isin(['夏普比率', '年化收益率', '累计超额收益'])],
    palette=[COLORS['传统蓝标准'], COLORS['增强红标准']],  # 柱状图颜色同步更新
    edgecolor='black',
    linewidth=0.5
)

plt.ylabel('数值', fontweight='bold')
plt.xlabel('指标', fontweight='bold')
plt.legend(title='策略类型', loc='upper left', frameon=True)
plt.xticks(rotation=15)
plt.gca().grid(False)
plt.tight_layout()
plt.savefig('核心指标柱状对比图.png', dpi=300, bbox_inches='tight')

metrics_compare.to_csv("策略绩效对比.csv", index=False, encoding='utf-8-sig')

print("""
绩效对比结果已保存：
- 夏普比率热力图对比.png ：双图热力图
- 最优策略收益对比.png    ：收益曲线对比
- 超额收益对比.png        ：超额收益曲线（标准蓝和红）
- 绩效指标对比表.png      ：指标表格
- 策略多维能力雷达图.png  ：多维能力雷达图
- 核心指标柱状对比图.png  ：核心指标柱状图
- 策略绩效对比.csv        ：详细对比数据
""")
