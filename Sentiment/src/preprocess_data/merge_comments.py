import os
import pandas as pd


# 1. 加载 Excel 文件并读取指定列
def read_comments(file_path):
    df = pd.read_excel(file_path)  # 读取 Excel 文件
    # 检查是否包含所需的列
    if all(col in df.columns for col in ['评论内容', '时间', '点赞']):
        # 提取需要的列
        return df[['评论内容', '时间', '点赞']]
    else:
        print(f"警告: 文件 {file_path} 缺少必要的列，跳过该文件。")
        return None


# 2. 合并同一文件夹中的 caifu 和 guba 评论数据
def merge_comments_in_folder(folder_path):
    # 用于存储文件夹中的评论数据
    all_comments = []

    for file_name in ['caifu_comments.xlsx', 'guba_comments.xlsx']:
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            print(f"正在处理文件: {file_path}")

            # 读取评论数据
            comments_df = read_comments(file_path)

            if comments_df is not None:
                # 添加来源列，用于标识来自哪个文件
                comments_df['来源'] = file_name.split('.')[0]  # 'caifu' 或 'guba'
                all_comments.append(comments_df)

    # 如果找到了两个文件中的数据，进行合并
    if all_comments:
        merged_comments = pd.concat(all_comments, ignore_index=True)

        # 保存合并后的数据到新的 Excel 文件
        output_file = os.path.join(folder_path, 'merged_comments.xlsx')
        merged_comments.to_excel(output_file, index=False)
        print(f"评论数据已合并并保存到: {output_file}")
    else:
        print(f"文件夹 {folder_path} 中没有有效的评论数据。")


# 3. 遍历每个文件夹并处理其中的 caifu 和 guba 文件
def process_all_folders(base_dir):
    # 遍历目标文件夹中的每个子文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            # 处理该文件夹中的 caifu 和 guba 评论数据
            merge_comments_in_folder(folder_path)


# 主程序
def main():
    # 指定处理的根目录
    base_dir = r"D:\All_of_mine\大学\项目和比赛\大创\src\data\评论"

    # 遍历每个文件夹并合并评论数据
    process_all_folders(base_dir)


# 执行主程序
if __name__ == "__main__":
    main()
