import os
import requests
import re
import time
from openpyxl import Workbook  # 引入 openpyxl 用于处理 Excel 文件

start_time = time.time()

# 定义请求URL
url = 'https://guba.eastmoney.com/api/getData'

# 定义请求载荷
payload = {
    'path': 'reply/api/Reply/ArticleNewReplyList',
    'plat': 'Web',
    'env': '2',
    'origin': '',
    'version': '2022',
    'product': 'Guba'
}

# 定义请求头
headers = {
    'Cookie': 'qgqp_b_id=9bd18148d5268b1ad604663111b25171; websitepoptg_api_time=1718433937510; mtp=1; ct=jRVvbf8C8eAUubLQdThpU-ROBuIB2bAGGJ8v7o7ovwvrakUqOcUSev704mfeCc6ps_pPNID9XTs4oeWCQCGpSKWWVAEiXdc-VO-GwrO5KoQikocxuFW8antNo5_bN-_LC00g4-PiPbSc8iSWbkMWUqBqz8fUHR6BALvemxRGi1E; ut=FobyicMgeV5FJnFT189SwCd727ut6we6RRtllPH-KJb9X6lWhAq5wCVdqitPguazjaX79vlt-3LtPZv0Gg-OIGMr_P-tE1RBZvzXAbmV4eEAFzA2jGwEI9zAYoQntfZV3s5eNwgBBL-68TjP-i_w85WeT6tOKi1sMa00VbqlVUJ0ZsOk5UDZrSrc_2Y2WhNFr_sauUxADZMoW1SoNhin2i3pEFLlwWUfbFUUsCCRFM8Y55VVNvDEgiRFTXp5i2mJenwiH2udoSbacRlT45e5v4xHuyEMJpC7AzRvpjXdrq4SAKn4ORQnmDwy5koQsM28HpCQRTLx8JuKL3jBccFSgaj7hij6jrE-; pi=1684047184343106%3Bj1684047184343106%3B%E8%82%A1%E5%8F%8B568312d77x%3Bm7AlIy3sL9qN4W0kx99BoGN9hKBUIF1Fbsq6bGgHfkQLKRAaH5ua4Qqc2oy3K65j0c5IaszLls%2BCse6hqaXztjc9g8lsMjceUEq6UPFxF9fMetST6eYIArjpeoDFM5TRODHlGVxyqUCSbM53pX0FL8Ic6f0j%2FyWfaiccAl5DgmH%2Bv61WY6pJOzytIbM7RWvacrQZDoO%2B%3B812kYt8QdSYs00plVokIB4HfxU5D8zKsLkHFOU0K2rDD82DQKrV500wt97U6vDL1U1M2HGnMnSn2WlSYe%2BuDFcvQCuDhHeHL2efdx047BFdfnoG6iD0YRJ99QRmarHVPgMQLMqtSfnTUFLlTtDWW5MMRqPn%2Fng%3D%3D; uidal=1684047184343106%e8%82%a1%e5%8f%8b568312d77x; sid=; vtpst=|; st_si=06762407280692; st_asi=delete; st_pvi=95297412281025; st_sp=2024-06-15%2014%3A45%3A37; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=2; st_psi=20240617144309573-117001354293-1897263224',
    'Host': 'guba.eastmoney.com',
    'Origin': 'https://guba.eastmoney.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

# 定义提取评论的递归函数
def extract_comments(comments):
    # 检查 comments 是否为 None 或空列表，如果是，则返回空列表
    if comments is None or not comments:
        return

    for comment in comments:
        comment_data = {
            '评论者ID': comment.get('user_id'),
            '评论者名称': comment.get('reply_user', {}).get('user_nickname'),
            '评论内容': comment.get('reply_text', ''),  # 确保即使没有评论内容也不会出错
            '点赞人数': comment.get('reply_like_count', 0),
            '发布日期': comment.get('reply_time'),
            '发布IP': comment.get('reply_ip_address', '')
        }
        yield comment_data

        # 递归提取子评论
        yield from extract_comments(comment.get('child_replys', []))


# 遍历所有股票代码对应的文件夹
base_folder = r'D:\All_of_mine\大学\项目和比赛\da_chuang\src\data'

for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)

    # 检查是否是文件夹
    if os.path.isdir(folder_path):
        # 检查文件夹内是否存在 guba_urls.txt 文件
        guba_urls_path = os.path.join(folder_path, 'guba_urls.txt')
        if os.path.exists(guba_urls_path):
            # 读取文件中的所有 URL
            with open(guba_urls_path, 'r', encoding='utf-8') as file:
                referer_urls = [line.strip() for line in file.readlines()]

            # 创建 Excel 文件和工作表
            wb = Workbook()
            ws = wb.active
            ws.title = "Comments"
            ws.append(["评论者ID", "评论者名称", "评论内容", "点赞", "时间", "发布IP"])

            cnt = 0

            # 遍历所有 URL
            for referer_url in referer_urls:
                cnt += 1
                print(cnt)
                time.sleep(0.1)
                # 从 URL 中提取股票代码和 postid
                match = re.search(r'news,(\d+),(\d+).html', referer_url)
                if match:
                    stock_code, post_id = match.groups()
                    # 更新请求头中的 Referer
                    headers['Referer'] = referer_url
                    # 更新 payload 中的股票代码和 postid
                    payload['code'] = stock_code
                    payload['param'] = f'postid={post_id}&sort=1&sorttype=1&p=1&ps=30'

                    # 发送 POST 请求
                    response = requests.post(url, data=payload, headers=headers)
                    # 解析 JSON 响应数据
                    data = response.json()

                    # 检查响应中是否有评论数据
                    if 're' in data and data['re']:
                        # 从递归函数中获取所有评论数据并写入 Excel
                        for comment_data in extract_comments(data['re']):
                            ws.append([comment_data['评论者ID'], comment_data['评论者名称'], comment_data['评论内容'],
                                       comment_data['点赞人数'], comment_data['发布日期'], comment_data['发布IP']])

            # 保存 Excel 文件
            output_file_path = os.path.join(folder_path, 'guba_comments.xlsx')
            wb.save(output_file_path)

            print(f"股票代码 {folder_name} 的评论数据已保存到 {output_file_path}")
            time.sleep(1)

end_time = time.time()
run_time = end_time - start_time
print(f"程序运行时间: {int(run_time)}s")
