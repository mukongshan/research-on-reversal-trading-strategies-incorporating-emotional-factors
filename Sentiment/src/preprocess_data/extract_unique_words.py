import pandas as pd

# 读取 Excel 文件
file_path = r"../../mid_result/hs300_sentiment_data/股吧评论分词结果.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 确保列名正确
column_name = "分词结果"

# 创建一个集合来存储去重后的词语
unique_words = set()

# 处理每一行数据，拆分单词并去重
df[column_name].dropna().apply(lambda x: unique_words.update(x.split()))

# 转换为 DataFrame
unique_words_df = pd.DataFrame(sorted(unique_words), columns=["词语"])

# 保存为 xlsx 文件
output_path = "unique_words.xlsx"
unique_words_df.to_excel(output_path, index=False, engine='openpyxl')

print(f"去重后的词语已保存到 {output_path}")
