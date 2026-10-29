import os
import sys

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

load_env_file()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("未找到 DEEPSEEK_API_KEY。")
    print("请二选一：")
    print("  A) 在项目根目录的 .env 里写一行： DEEPSEEK_API_KEY=你的key")
    print('  B) 在终端执行： $env:DEEPSEEK_API_KEY="你的key"')
    sys.exit(1)

DOCS = [
    "公司年假政策：正式员工入职满1年享5天年假，满3年10天，满10年15天。",
    "报销流程：员工提交发票，直属领导审批后，财务在7个工作日内打款。",
    "弹性工作：核心时段为10:00-16:00，前后可弹性上下班各1小时。",
    "试用期：新员工的试用期为3个月，表现优异可提前转正。",
    "食堂：午餐补贴每人每餐8元，支持微信和饭卡支付。",
]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh")
import chromadb
client = chromadb.Client()                                                            
collection = client.create_collection(name="kb_docs", embedding_function=ef)
collection.add(documents=DOCS, ids=[f"doc{i}" for i in range(len(DOCS))])

print(f"资料库就绪，共 {collection.count()} 条。输入问题（exit 退出）")
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

    hits = collection.query(query_texts=[q], n_results=2)
    chunks = hits["documents"][0]
    print("\n[检索到的资料]")
    for c in chunks:
        print("  -", c)

    context = "\n".join(chunks)
    prompt = (
        "根据下面资料回答问题，资料中未提及就如实说'资料中未提及'。\n"
        f"【资料】\n{context}\n【问题】{q}"
    )
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是严谨的问答助手，只依据给定资料回答。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,                           
        },
        timeout=60,
    )
    print("助手:", resp.json()["choices"][0]["message"]["content"])
