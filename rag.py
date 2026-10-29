"""
Step 4: RAG 检索增强生成（让模型读懂你自己的资料）
运行前: pip install chromadb sentence-transformers
运行:   python rag.py
配置:   DEEPSEEK_API_KEY 写进同目录 .env（自动读取），或在终端设环境变量
原理: 资料切块 -> 向量化 -> 存向量库；提问时先检索相关块，
      再把"资料 + 问题"一起交给模型，让它基于资料回答（大幅减少瞎编）。
"""
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


# 先尝试从 .env 读取 Key（和 llm_test.py 一致；系统环境变量优先级更高）
load_env_file()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("未找到 DEEPSEEK_API_KEY。")
    print("请二选一：")
    print("  A) 在项目根目录的 .env 里写一行： DEEPSEEK_API_KEY=你的key")
    print('  B) 在终端执行： $env:DEEPSEEK_API_KEY="你的key"')
    sys.exit(1)

# 演示资料库；真实场景换成你自己的 txt/md/pdf 文本内容
DOCS = [
    "公司年假政策：正式员工入职满1年享5天年假，满3年10天，满10年15天。",
    "报销流程：员工提交发票，直属领导审批后，财务在7个工作日内打款。",
    "弹性工作：核心时段为10:00-16:00，前后可弹性上下班各1小时。",
    "试用期：新员工的试用期为3个月，表现优异可提前转正。",
    "食堂：午餐补贴每人每餐8元，支持微信和饭卡支付。",
]

# 1) 中文嵌入模型（本地，首次会自动下载约90MB的 bge-small-zh）
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh")
import chromadb
client = chromadb.Client()            # 内存版；要持久化用 chromadb.PersistentClient(path="db")
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

    # 2) 检索最相关的 k 块（R = Retrieval）
    hits = collection.query(query_texts=[q], n_results=2)
    chunks = hits["documents"][0]
    print("\n[检索到的资料]")
    for c in chunks:
        print("  -", c)

    # 3) 拼"资料 + 问题"，交给模型生成（G = Generation）
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
            "temperature": 0.3,        # 调低，让回答更"抠资料"、更少发挥
        },
        timeout=60,
    )
    print("助手:", resp.json()["choices"][0]["message"]["content"])
