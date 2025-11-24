import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# 全局变量区域（控制输入输出路径）
# ==========================================
INPUT_FILE_PATH = '../../mid_result/hs300_data/hs300_stocks_forum/daily_fitted_sentiment.csv'  # 输入文件路径
OUTPUT_FILE_PATH = 'emotion_fitting_time_series_plot.png'  # 输出文件路径

# ========== 1. 字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==========================================
# 1. 加载数据
# ==========================================
def load_data(file_path):
    """
    加载CSV文件并解析日期列
    :param file_path: CSV文件路径
    :return: DataFrame
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['日期'])
        print(f"数据加载成功: {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

# ==========================================
# 2. 绘制情绪拟合值时间序列图
# ==========================================
def plot_emotion_fitting(data):
    """
    绘制情绪拟合值随时间变化的折线图
    :param data: 包含日期和情绪拟合值的DataFrame
    """
    plt.figure(figsize=(10, 6))  # 图形大小

    # 绘制情绪拟合值的折线图
    plt.plot(data['日期'], data['情绪拟合值'], marker='o', color='b', label='情绪拟合值')

    # 设置日期格式和间隔
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # 每周一个刻度
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签

    # 添加标题和标签
    plt.title('情绪拟合值随时间的变化', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('情绪拟合值', fontsize=12)

    # 显示图例
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_PATH)  # 保存为文件
    print(f"图表已保存到: {OUTPUT_FILE_PATH}")

# ==========================================
# 3. 主函数
# ==========================================
def main():
    """
    主函数，负责控制整个流程
    """
    # 加载数据
    data = load_data(INPUT_FILE_PATH)
    if data is not None:
        # 绘制图表
        plot_emotion_fitting(data)

# ==========================================
# 4. 程序入口
# ==========================================
if __name__ == '__main__':
    main()
