import os
import requests
import re
import time
from openpyxl import Workbook  # 引入 openpyxl 用于处理 Excel 文件
from datetime import datetime

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
    if comments is None or not comments:
        return

    for comment in comments:
        comment_data = {
            '评论者ID': comment.get('user_id'),
            '评论者名称': comment.get('reply_user', {}).get('user_nickname'),
            '评论内容': comment.get('reply_text', ''),
            '点赞人数': comment.get('reply_like_count', 0),
            '发布日期': comment.get('reply_time'),
            '发布IP': comment.get('reply_ip_address', '')
        }
        yield comment_data
        yield from extract_comments(comment.get('child_replys', []))

# 定义检查时间并返回是否继续的函数
def check_post_time_and_continue(url):
    # 从 URL 中提取股票代码和 postid
    match = re.search(r'news,(\d+),(\d+).html', url)
    if match:
        stock_code, post_id = match.groups()
        # 更新请求头中的 Referer
        headers['Referer'] = url
        # 更新 payload 中的股票代码和 postid
        payload['code'] = stock_code
        payload['param'] = f'postid={post_id}&sort=1&sorttype=1&p=1&ps=30'

        # 发送 POST 请求
        # print(payload)
        response = requests.post('https://guba.eastmoney.com/api/getData', data=payload, headers=headers)

        # 解析 JSON 响应数据
        data = response.json()
        # print(data)

        # 检查响应中是否有评论数据
        if 're' in data and data['re']:
            # 获取评论发布时间
            post_time_str = data['re'][0].get('reply_time', '')
            try:
                post_time = datetime.strptime(post_time_str, "%Y-%m-%d %H:%M:%S")
                if post_time < datetime(2020, 1, 1):
                    # print(f"发现旧帖，发布时间 {post_time}，跳过 URL {url}")
                    return False  # 返回 False，表示不继续爬取
            except ValueError:
                print(f"日期格式错误: {post_time_str}，跳过 URL {url}")
        return True  # 返回 True，表示可以继续

    print(f"URL {url} 无效，无法提取股票代码和 postid")
    return True  # 如果 URL 无效，返回 True

# 示例：调用 check_post_time_and_continue 函数
# url_to_check = 'https://guba.eastmoney.com/news,000001,1519540241.html'
# should_continue = check_post_time_and_continue(url_to_check)

