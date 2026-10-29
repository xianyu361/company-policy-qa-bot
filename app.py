"""
Step 5: 工程化部署 —— 用 FastAPI 把「公司制度问答机器人」封装成 HTTP 服务 + 网页
把 Step 1~4 的全部能力（RAG + 工具调用 + 多轮对话）变成别人能直接用的产品。

运行前: pip install fastapi uvicorn
运行:   uvicorn app:app --reload --port 8000
网页:   浏览器打开 http://127.0.0.1:8000

你会学到:
  - Web 框架如何"包住"你的 AI 逻辑（服务化）
  - 用 HTTP 接口（/chat）对外提供能力
  - 用最简单的 HTML 页面做可演示的前端
  - 多轮对话在服务端怎么用 session 管理（而不是终端 while 循环）
"""
import datetime
import json
import os
import sys
import uuid

# 国内访问 HuggingFace 常被墙，设置镜像以下载嵌入模型 bge-small-zh
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------- 1) 读取 Key（同 .env 逻辑，和前面脚本一致） ----------------
def load_env_file():
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
    print("  A) 在项目根目录的 .env 里写一行： DEEPSEEK_API_KEY=你的key")
    print('  B) 在终端执行： $env:DEEPSEEK_API_KEY="你的key"')
    sys.exit(1)

# ---------------- 2) 知识库 ----------------
POLICY = [
    "年假：入职满1年享5天，满3年10天，满10年15天。",
    "报销：提交发票 -> 直属领导审批 -> 财务7个工作日内打款。",
    "弹性工作：核心时段10:00-16:00，前后可弹性上下班各1小时。",
    "试用期：新员工试用期3个月，表现优异可提前转正。",
    "食堂：午餐补贴每餐8元，支持微信和饭卡支付。",
    "加班：工作日加班可换调休，周末加班按2倍工资或调休。",
]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh")
import chromadb
# 持久化：向量库落盘到 ./chroma_db，重启不用重建（首次自动建，之后直接读）
client = chromadb.PersistentClient(path="./chroma_db")
kb = client.get_or_create_collection(name="policy", embedding_function=ef)
if kb.count() == 0:
    kb.add(documents=POLICY, ids=[f"p{i}" for i in range(len(POLICY))])
    print("已建立持久化知识库（首次启动）")
else:
    print(f"已加载持久化知识库，共 {kb.count()} 条")

# ---------------- 3) 工具 ----------------
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

SYSTEM_PROMPT = ("你是公司制度助手，只依据检索到的资料回答；需要算年假或查时间时调用工具；"
                 "资料未提及就如实说未提及。")

# ---------------- 4) 多轮对话：服务端用 session 管理历史 ----------------
# 终端版用 while 循环 + 全局 history；服务版是多人并发，必须按 session 分开存。
sessions: dict[str, list] = {}

def get_history(session_id: str) -> list:
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return sessions[session_id]

def ask_model(history: list) -> str:
    """跑一次「请求模型 → 调工具 → 再生成」的 ReAct 循环，返回最终文本。"""
    for _ in range(5):
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json={"model": "deepseek-chat", "messages": history,
                  "tools": TOOLS_SCHEMA, "temperature": 0.3},
            timeout=60,
        )
        msg = resp.json()["choices"][0]["message"]
        if msg.get("tool_calls"):
            history.append(msg)  # 原样存回，模型才知道自己要调什么
            for call in msg["tool_calls"]:
                fn = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                result = TOOLS[fn](**args)
                history.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
            continue  # 带着工具结果再生成
        history.append({"role": "assistant", "content": msg["content"]})
        return msg["content"]
    return "（模型未给出最终答复，请重试）"

# ---------------- 5) FastAPI 应用 ----------------
app = FastAPI(title="公司制度问答机器人")

class ChatReq(BaseModel):
    message: str
    session_id: str | None = None

@app.post("/chat")
def chat(req: ChatReq):
    session_id = req.session_id or str(uuid.uuid4())
    history = get_history(session_id)
    # RAG：先检索相关资料，拼进这一轮的问题
    hits = kb.query(query_texts=[req.message], n_results=3)
    context = "\n".join(hits["documents"][0])
    augmented = f"【相关资料】\n{context}\n【用户问题】{req.message}"
    history.append({"role": "user", "content": augmented})
    reply = ask_model(history)
    return {"reply": reply, "session_id": session_id}

@app.get("/health")
def health():
    return {"status": "ok"}

# 最简单的网页前端：纯 HTML + 一点点 JS，直接调 /chat 接口
PAGE = """
<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>公司制度问答机器人</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
 #log{height:360px;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}
 .u{color:#15539e;margin:8px 0}.a{color:#0a7d3b;margin:8px 0;white-space:pre-wrap}
 input{flex:1;padding:10px;border:1px solid #ccc;border-radius:8px}
 button{padding:10px 16px;border:0;border-radius:8px;background:#15539e;color:#fff;cursor:pointer}
 .row{display:flex;gap:8px;margin-top:12px}
</style></head><body>
<h2>公司制度问答机器人</h2>
<div id="log"></div>
<div class="row"><input id="inp" placeholder="输入问题，回车发送"><button onclick="send()">发送</button></div>
<script>
let sid = null;
const log = document.getElementById('log');
const inp = document.getElementById('inp');
function add(who, text){const d=document.createElement('div');d.className=who==='你'?'u':'a';d.textContent=who+'：'+text;log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function send(){
  const q = inp.value.trim(); if(!q) return;
  add('你', q); inp.value='';
  const r = await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:q, session_id:sid})});
  const data = await r.json(); sid = data.session_id;
  add('助手', data.reply);
}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
