import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import InvalidArgumentException
from lxml import etree
from openpyxl import Workbook  # 引入 openpyxl 用于处理 Excel 文件

start_time = time.time()

# 设置 Edge 浏览器选项
edge_options = Options()
edge_options.add_argument("--headless")  # 无头模式
edge_options.add_argument("--disable-gpu")
edge_options.add_argument("--no-sandbox")

# 手动指定 Edge WebDriver 的路径
driver_path = r"D:\All_of_mine\大学\项目和比赛\大创\src\爬取东方财富股吧评论\.venv\Scripts\msedgedriver.exe"

# 启动 Edge 浏览器
service = Service(driver_path)
driver = webdriver.Edge(service=service, options=edge_options)

# 获取父目录下所有文件夹路径
parent_dir = r"D:\All_of_mine\大学\项目和比赛\大创\src\data"  # 修改为你存放数据的根目录
subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

# 遍历每个文件夹处理对应的 URL
for folder in subfolders:
    # 定义 Excel 文件路径和结果存储路径
    result_file = os.path.join(folder, 'caifu_comments.xlsx')

    # 创建 Excel 文件和工作表
    wb = Workbook()
    ws = wb.active
    ws.title = "Comments"
    ws.append(["title", "用户id", "评论者", "评论者IP", "时间", "评论内容", "点赞", "评论者链接"])

    # 读取当前文件夹下的 URLs
    urls_file = os.path.join(folder, 'caifu_urls.txt')  # 假设每个文件夹中都有 urls.txt
    try:
        with open(urls_file, 'r', encoding='utf-8') as file:
            referer_urls = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"文件夹 {folder} 中的 urls.txt 文件未找到，跳过此文件夹。")
        continue

    # 遍历每个 URL 并提取评论数据
    for url in referer_urls:
        try:
            # 打开目标网页
            driver.get(url)

            # 等待页面加载
            time.sleep(3)
        except InvalidArgumentException as e:
            print(f"无效的URL：{url}")
            print(f"错误信息：{e}")
        except Exception as e:
            print(f"处理URL {url} 时发生未知错误：{e}")

        # 点击“加载更多”按钮，直到按钮不存在为止
        while True:
            try:
                load_more_button = driver.find_element(By.XPATH, '//div[@class="loadMbtn bottombtn fl"]')
                load_more_button.click()
                time.sleep(3)  # 等待加载更多评论
            except:
                break

        # 获取页面HTML内容
        page_source = driver.page_source
        tree = etree.HTML(page_source)

        # 提取文章标题
        title = tree.xpath('//div[@class="grid_wrapper"]//h1[@class="article-title"]/text()')
        title = title[0].strip() if title else 'error'

        # 提取评论项
        items = tree.xpath('//div[@class="list_cont"]')
        for item in items:
            # 提取评论字段
            commenter_id = item.xpath('.//a[@class="replyer_name"]/@data-popper')
            commenter_id = commenter_id[0].strip() if commenter_id else 'error'

            commenter = item.xpath('.//a[@class="replyer_name"]/text()')
            commenter = commenter[0].strip() if commenter else 'error'

            comment_time = item.xpath('.//div[@class="publish_time fr"]/span[1]/text()')
            comment_time = comment_time[0].strip() if comment_time else 'error'

            ip = item.xpath('.//div[@class="publish_time fr"]/span[2]/text()')
            ip = ip[0].strip() if ip else 'error'

            comment_content = item.xpath('.//div[@class="short_text"]/text()')
            comment_content = comment_content[0].strip() if comment_content else 'error'

            comment_likes = item.xpath('.//div[@class="level1_btns"]//span[@class="z_num"]/text()')
            comment_likes = comment_likes[0].strip() if comment_likes else 'error'
            if comment_likes == '点赞':
                comment_likes = 0

            commenter_link = item.xpath('.//a[@class="replyer_name"]/@href')
            commenter_link = "https:" + commenter_link[0].strip() if commenter_link else 'error'

            # 将评论数据添加到 Excel 文件中
            ws.append([title, commenter_id, commenter, ip, comment_time, comment_content, comment_likes, commenter_link])

            # 提取并写入回复评论
            replies = item.xpath('.//div[@class="level2_item"]')
            for reply in replies:
                replier_id = reply.xpath('.//a[@class="replyer_name"]/@data-popper')
                replier_id = replier_id[0].strip() if replier_id else 'error'

                replier = reply.xpath('.//a[@class="replyer_name"]/text()')
                replier = replier[0].strip() if replier else 'error'

                reply_time = reply.xpath('.//div[@class="time fl"]/span[1]/text()')
                reply_time = reply_time[0].strip() if reply_time else 'error'

                reply_content = reply.xpath('.//span[@class="l2_short_text"]/text()')
                reply_content = reply_content[0].strip() if reply_content else 'error'

                reply_likes = reply.xpath('.//span[@class="z_num"]/text()')
                reply_likes = reply_likes[0].strip() if reply_likes else 'error'
                if reply_likes == '点赞':
                    reply_likes = 0

                reply_ip = reply.xpath('.//span[@class="reply_ip"]/text()')
                reply_ip = reply_ip[0].strip() if reply_ip else 'error'

                replier_link = reply.xpath('.//a[@class="replyer_name"]/@href')
                replier_link = "https:" + replier_link[0].strip() if replier_link else 'error'

                # 将回复评论数据添加到 Excel 文件中
                ws.append([title, replier_id, replier, reply_ip, reply_time, reply_content, reply_likes, replier_link])

    # 保存每个文件夹的 Excel 文件
    wb.save(result_file)
    print(f"数据已保存到 {result_file}")

# 关闭浏览器
driver.quit()

end_time = time.time()
run_time = end_time - start_time
print(f"程序运行时间: {int(run_time)}s")
