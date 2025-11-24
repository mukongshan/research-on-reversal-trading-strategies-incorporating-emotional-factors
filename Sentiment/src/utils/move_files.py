import os
import shutil

# 设置源文件夹和目标文件夹
source_root = r"D:\All_of_mine\大学\项目和比赛\da_chuang\src\data"  # 你的源文件夹
destination_root = r"D:\All_of_mine\大学\项目和比赛\da_chuang\res\src_data"  # 你的目标文件夹

# 遍历源文件夹中的所有子文件夹
for root, dirs, files in os.walk(source_root):
    if "stock_info.xlsx" in files:  # 只处理 stock_info.xlsx 文件
        # 计算相对路径（保持目录结构）
        relative_path = os.path.relpath(root, source_root)
        destination_path = os.path.join(destination_root, relative_path)

        # 确保目标路径存在
        os.makedirs(destination_path, exist_ok=True)

        # 移动文件
        source_file = os.path.join(root, "stock_info.xlsx")
        destination_file = os.path.join(destination_path, "stock_info.xlsx")

        shutil.move(source_file, destination_file)
        print(f"已移动: {source_file} -> {destination_file}")

print("所有文件已成功移动！")
