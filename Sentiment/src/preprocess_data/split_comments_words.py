import os
import jieba
import pandas as pd


# 1. 加载停用词表
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())  # 每行是一个停用词
    return stopwords


# 2. 加载 Excel 文件并读取评论内容
def read_excel(file_path):
    df = pd.read_excel(file_path)  # 读取 Excel 文件
    return df


# 3. 分词并去停用词
def process_text(text, stopwords):
    # 确保文本是字符串，并处理缺失值
    if isinstance(text, str):
        words = jieba.cut(text)
        # 去除停用词
        filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
        return ' '.join(filtered_words)
    else:
        return ''  # 对于非字符串类型或缺失值，返回空字符串


# 4. 保存处理后的数据
def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)  # 保存为新的 Excel 文件


# 5. 递归查找指定文件夹下所有的 merged_comments.xlsx 文件并处理
def process_all_files(base_dir, stopwords):
    # 遍历指定目录下的所有文件夹和文件
    for root, dirs, files in os.walk(base_dir):
        # 如果文件夹下有 stock_info.xlsx 文件
        if 'stock_info.xlsx' in files:
            file_path = os.path.join(root, 'stock_info.xlsx')
            print(f"处理文件: {file_path}")

            # 读取 Excel 文件
            df = read_excel(file_path)

            # 检查是否有“评论内容”列
            if '标题' in df.columns:
                # 分词并去除停用词
                df['分词结果'] = df['标题'].apply(lambda x: process_text(x, stopwords))

                # 保存处理后的文件
                output_file = os.path.join(root, '股吧评论分词结果.xlsx')
                save_to_excel(df, output_file)
                print(f"已处理并保存为: {output_file}")
            else:
                print(f"警告: 文件 {file_path} 中没有 '评论内容' 列，跳过该文件。")


# 主程序
def main():
    # 1. 加载停用词
    stopwords_file = r"D:\All_of_mine\大学\项目和比赛\da_chuang\res\停用词表.txt"
    stopwords = load_stopwords(stopwords_file)

    # 2. 指定处理的根目录
    base_dir = r"D:\All_of_mine\大学\项目和比赛\da_chuang\src\data_2"

    # 3. 处理文件夹下的所有文件
    process_all_files(base_dir, stopwords)


# 执行主程序
if __name__ == "__main__":
    main()
