import os
import pandas as pd
import json
import re
from tqdm import tqdm
from openai import OpenAI

# ===== 配置 =====
API_KEY = "sk-3fe83fa822aa472b88b053fdf794c429"  # 从环境变量读取
MODEL_NAME = "qwen-plus"   # 可换 qwen2-7b-instruct / qwen2-72b-instruct
INPUT_DIR = r"../../../mid_result/2w_titles_slice"   # 输入文件夹
OUTPUT_DIR = r"../../../mid_result/2w_titles_scored"  # 输出文件夹
BATCH_SIZE = 200   # 建议 100~500，避免超 token

# 初始化 client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===== Prompt 模板 =====
PROMPT_TEMPLATE = """
你是一名金融情绪分析员，请严格按照以下规则为标题打分：
- 负值代表负面情绪，正值代表正面情绪，绝对值越大说明情绪越强烈。
- 范围：[-1, 1]  
- 如果标题涉及利空、下跌、亏损 → 负面（接近 -1）  
- 如果标题涉及利好、上涨、盈利、分红 → 正面（接近 +1）  
- 如果标题中性或无关（如公司公告、一般财务指标） → 接近 0  

⚠️ 输出要求：
- 必须输出一个合法 JSON 对象
- 不能包含 ```json 或解释说明
- 格式：{{"标题1": 分数, "标题2": 分数, ...}}

标题列表：{titles}
"""

# ===== 安全解析函数 =====
def safe_json_parse(res: str):
    if not res:
        return {}
    res = res.strip()
    match = re.search(r"\{[\s\S]*\}", res)
    if match:
        res = match.group(0)
    try:
        return json.loads(res)
    except Exception:
        print("⚠️ JSON解析失败，原始内容:", res[:200])
        return {}

# ===== 调用 Qwen API =====
def ask_model(titles):
    prompt = PROMPT_TEMPLATE.format(titles=", ".join(titles))
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个严格的金融情绪分析助手"},
            {"role": "user", "content": prompt},
        ],
        extra_body={"enable_thinking": False}
    )
    return completion.choices[0].message.content

# ===== 主流程（处理单个文件） =====
def process_file(input_file, output_file):
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    titles = df["标题"].dropna().astype(str).tolist()
    scores = {}

    for i in tqdm(range(0, len(titles), BATCH_SIZE), desc=f"处理 {os.path.basename(input_file)}"):
        batch = titles[i:i+BATCH_SIZE]
        res = ask_model(batch)
        batch_scores = safe_json_parse(res)
        scores.update(batch_scores)

    df["分数"] = df["标题"].map(scores)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ {os.path.basename(input_file)} 完成，结果已保存到 {output_file}")

# ===== 遍历文件夹 =====
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".csv"):
            input_path = os.path.join(INPUT_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, f"scored_{file}")
            process_file(input_path, output_path)

if __name__ == "__main__":
    main()
