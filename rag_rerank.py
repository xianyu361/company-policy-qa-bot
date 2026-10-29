import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from app import kb, API_KEY, SYSTEM_PROMPT
import requests

from sentence_transformers import CrossEncoder

print("正在加载重排模型（首次需下载，请耐心）...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

def vector_retrieve(query, k=10):
    """向量检索召回 top-k 候选"""
    hits = kb.query(query_texts=[query], n_results=k)
    return hits["documents"][0]

def rerank(query, docs, top_n=3):
    """用交叉编码器对候选打分并重排，取前 top_n"""
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)                    
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

    base = vector_retrieve(q, 10)                      
    top = rerank(q, base, 3)                         

    print("\n[向量检索 top10]")
    for d in base:
        print("  -", d)
    print("\n[重排后 top3]")
    for d in top:
        print("  *", d)

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
