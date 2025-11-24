import pandas as pd

# 读取原始 Excel 文件
input_path = "D:\\All_of_mine\\大学\\项目和比赛\\大创\\src\\data\\600519\\guba_comments_processed.xlsx"  # 替换为你的输入文件路径
df = pd.read_excel(input_path)

# 拆分“分词结果”列并展开
df_exploded = df["分词结果"].str.split().explode().reset_index(drop=True)

# 保存到新的 Excel 文件
output_path = "训练集.xlsx"  # 替换为你的输出文件路径
df_exploded.to_frame(name="词语").to_excel(output_path, index=False)

print(f"处理完成，结果已保存至 {output_path}")
