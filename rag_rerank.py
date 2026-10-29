"""
项目打磨点 3：重排 rerank（提升 RAG 检索质量）
原理：
  向量检索擅长"召回"（拉回一批可能相关的候选），但不擅长"精准排序"。
  rerank 用交叉编码器(CrossEncoder)对「Query 与每段候选」逐一打分，重新排序取 top-n，
  让最相关的一段排在最前，减少噪声、提升命中。

演示内容：对比「纯向量检索 top10」与「向量检索 top10 + 重排 top3」，并用重排结果作答。

运行前: pip install sentence-transformers
运行:   python rag_rerank.py
注意: 首次会自动下载重排模型 BAAI/bge-reranker-base（体积较大，耐心等待，走国内镜像）
"""
import os
# 国内镜像，避免下载重排模型时被墙
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 复用 app.py 里已持久化的知识库，避免重复建库
from app import kb, API_KEY, SYSTEM_PROMPT
import requests

from sentence_transformers import CrossEncoder

# 加载中文重排模型（首次下载较大）
print("正在加载重排模型（首次需下载，请耐心）...")
reranker = CrossEncoder("BAAI/bge-reranker-base")


def vector_retrieve(query, k=10):
    """向量检索召回 top-k 候选"""
    hits = kb.query(query_texts=[query], n_results=k)
    return hits["documents"][0]


def rerank(query, docs, top_n=3):
    """用交叉编码器对候选打分并重排，取前 top_n"""
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)  # 每段与 query 的相关性分数
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return [d for d, _ in ranked[:top_n]]


print("重排演示已就绪（exit 退出）")
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

    base = vector_retrieve(q, 10)          # 向量检索 top10
    top = rerank(q, base, 3)               # 重排取 top3

    print("\n[向量检索 top10]")
    for d in base:
        print("  -", d)
    print("\n[重排后 top3]")
    for d in top:
        print("  *", d)

    # 用重排后的 top3 作为上下文去问模型
    context = "\n".join(top)
    prompt = (
        "根据下面资料回答问题，资料中未提及就如实说'资料中未提及'。\n"
        f"【资料】\n{context}\n【问题】{q}"
    )
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json={"model": "deepseek-chat",
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": prompt}],
              "temperature": 0.3},
        timeout=60,
    )
    print("助手:", resp.json()["choices"][0]["message"]["content"])
