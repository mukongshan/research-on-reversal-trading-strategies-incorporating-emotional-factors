import requests
import random

# 代理 IP 列表
proxy_list = [
    "http://114.231.41.30:8089",
    "http://223.215.177.179:8089",
    "http://58.214.243.91:80",
    "http://111.225.153.100:8089",
    "http://114.106.146.137:8089",
    "http://117.71.154.160:8089",
]

# 随机选择一个代理
proxy = random.choice(proxy_list)
proxies = {"http": proxy, "https": proxy}

try:
    response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=5)
    print("代理 IP 访问成功！", response.json())
except requests.exceptions.RequestException as e:
    print("代理不可用:", e)
