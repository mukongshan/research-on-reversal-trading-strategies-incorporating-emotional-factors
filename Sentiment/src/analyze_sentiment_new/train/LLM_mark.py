import os
import pandas as pd
import json
import re
from tqdm import tqdm
from openai import OpenAI

# ===== 配置 =====
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取
MODEL_NAME = "qwen-plus"   # 可换 qwen2-7b-instruct / qwen2-72b-instruct
INPUT_FILE = "random_16K_rows.xlsx"
OUTPUT_FILE = "scored_16K_LLM.xlsx"
BATCH_SIZE = 400

# 初始化 client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===== Prompt 模板 =====
PROMPT_TEMPLATE = """
你是一名金融情绪分析员，请严格按照以下规则为词语打分：
- 负值代表负面情绪，正值代表正面情绪，绝对值越大说明情绪越强烈。
- 范围：[-1, 1]  
- 交易行为或走势动词 → 绝对值 = 1  
- 行情氛围相关词 → 绝对值 = 0.6~0.8  
- 其他（如股票名、财务指标、日常词汇） → 绝对值 = 0.0~0.2  

⚠️ 输出要求：
- 必须输出一个合法 JSON 对象
- 不能包含 ```json 或解释说明
- 格式：{{"词语1": 分数, "词语2": 分数, ...}}

词语列表：{words}
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
def ask_model(words):
    prompt = PROMPT_TEMPLATE.format(words=", ".join(words))
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个严格的金融情绪分析助手"},
            {"role": "user", "content": prompt},
        ],
        extra_body={"enable_thinking": False}  # 关闭思考过程
    )
    return completion.choices[0].message.content

# ===== 主流程 =====
def main():
    df = pd.read_excel(INPUT_FILE)
    words = df["词语"].dropna().astype(str).tolist()
    scores = {}

    for i in tqdm(range(0, len(words), BATCH_SIZE)):
        batch = words[i:i+BATCH_SIZE]
        print(f"\n正在处理第 {i//BATCH_SIZE+1} 批，共 {len(batch)} 个词...")
        res = ask_model(batch)
        print("模型输出:", res[:200], "..." if len(res) > 200 else "")

        batch_scores = safe_json_parse(res)
        scores.update(batch_scores)

    df["分数"] = df["词语"].map(scores)
    df.to_excel(OUTPUT_FILE, index=False)
    print("✅ 完成，结果已保存到", OUTPUT_FILE)


if __name__ == "__main__":
    main()
