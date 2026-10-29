"""
小项目：公司制度问答机器人（整合 Step 1~4 全部能力）
  - RAG：从公司制度资料里检索相关段落再回答
  - 工具调用：算年假天数、查当前时间
  - 多轮对话 + 记忆
运行前: pip install chromadb sentence-transformers
运行:   python policy_bot.py
"""
import datetime
import json
import os
import sys

# 国内访问 HuggingFace 常被墙，设置镜像以下载嵌入模型 bge-small-zh
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
from chromadb.utils import embedding_functions


def load_env_file():
    """如果脚本同目录存在 .env，就把里面的 KEY=VALUE 读进环境变量（不覆盖已存在的）。"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# 先尝试从 .env 读取 Key（系统环境变量优先级更高）
load_env_file()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("未找到 DEEPSEEK_API_KEY。")
    print("请二选一：")
    print("  A) 在项目根目录的 .env 里写一行： DEEPSEEK_API_KEY=你的key")
    print('  B) 在终端执行： $env:DEEPSEEK_API_KEY="你的key"')
    sys.exit(1)

# ---------- 1) 知识库（真实场景换成读你的 txt/md/pdf） ----------
POLICY = [
    "年假：入职满1年享5天，满3年10天，满10年15天。",
    "报销：提交发票 -> 直属领导审批 -> 财务7个工作日内打款。",
    "弹性工作：核心时段10:00-16:00，前后可弹性上下班各1小时。",
    "试用期：新员工试用期3个月，表现优异可提前转正。",
    "食堂：午餐补贴每餐8元，支持微信和饭卡支付。",
    "加班：工作日加班可换调休，周末加班按2倍工资或调休。",
]

# 中文嵌入 + 向量库
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh")
import chromadb
client = chromadb.Client()
kb = client.create_collection(name="policy", embedding_function=ef)
kb.add(documents=POLICY, ids=[f"p{i}" for i in range(len(POLICY))])

# ---------- 2) 工具 ----------
def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calc_yearly_leave(years: int) -> str:
    """根据入职年数计算年假天数。years: 入职年数（整数）"""
    if years < 1:
        return "入职不满1年，无年假。"
    if years < 3:
        return f"入职{years}年，年假5天。"
    if years < 10:
        return f"入职{years}年，年假10天。"
    return f"入职{years}年，年假15天。"

TOOLS = {"get_current_time": get_current_time, "calc_yearly_leave": calc_yearly_leave}
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_current_time", "description": "获取当前日期和时间",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calc_yearly_leave", "description": "根据入职年数计算年假天数",
        "parameters": {"type": "object",
                       "properties": {"years": {"type": "integer", "description": "入职年数"}},
                       "required": ["years"]}}},
]

history = [{"role": "system", "content":
            "你是公司制度助手，只依据检索到的资料回答；需要算年假或查时间时调用工具；资料未提及就如实说未提及。"}]

print("公司制度问答机器人已启动（exit 退出）")
print("=" * 40)

while True:
    try:
        q = input("\n你: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n再见！")
        break
    if not q:
        continue
    if q.lower() == "exit":
        print("再见！")
        break

    # 3) RAG：先检索相关资料，拼进这一轮的问题
    hits = kb.query(query_texts=[q], n_results=3)
    context = "\n".join(hits["documents"][0])
    augmented = f"【相关资料】\n{context}\n【用户问题】{q}"
    history.append({"role": "user", "content": augmented})

    # 4) 多轮 + 工具调用循环（ReAct）
    for _ in range(5):
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json={"model": "deepseek-chat", "messages": history, "tools": TOOLS_SCHEMA, "temperature": 0.3},
            timeout=60,
        )
        msg = resp.json()["choices"][0]["message"]
        if msg.get("tool_calls"):
            print("助手: [调用工具]", flush=True)
            history.append(msg)  # 原样存回，模型才知道自己要调什么
            for call in msg["tool_calls"]:
                fn = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                print(f"  -> {fn}({args})", flush=True)
                result = TOOLS[fn](**args)
                history.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
            continue  # 带着工具结果再生成
        print("助手:", msg["content"])
        history.append({"role": "assistant", "content": msg["content"]})
        break
