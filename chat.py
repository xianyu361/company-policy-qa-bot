"""
Step 2: 多轮对话 + 流式输出（打字机效果）
运行: python chat.py
命令: exit 退出  |  clear 清空记忆
"""
import json
import os
import sys

import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请先设置环境变量 DEEPSEEK_API_KEY")
    sys.exit(1)

API_URL = "https://api.deepseek.com/chat/completions"

# "记忆"的本质：一个消息列表。system 永远放第一位（人设）
history = [
    {"role": "system", "content": "你是一个友好的中文助手，回答简洁准确。"},
]

print("AI 助手已启动 | 输入 exit 退出 | 输入 clear 清空记忆")
print("=" * 40)

while True:
    try:
        user_input = input("\n你: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n再见！")
        break

    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("再见！")
        break
    if user_input.lower() == "clear":
        history = history[:1]  # 只保留 system，忘掉之前所有对话
        print("（记忆已清空）")
        continue

    # 1) 先把用户这句话存进记忆，再连同历史一起发给模型
    history.append({"role": "user", "content": user_input})

    # 2) 发起流式请求
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": "deepseek-chat",
                "messages": history,
                "temperature": 0.7,
                "stream": True,   # 关键1: 让服务器流式返回
            },
            stream=True,           # 关键2: 让 requests 边收边读，不等全部结束
            timeout=60,
        )
    except requests.RequestException as e:
        print("网络错误:", e)
        history.pop()              # 回滚这条没发成功的消息
        continue

    if resp.status_code != 200:
        print("请求失败 HTTP", resp.status_code)
        print(resp.text[:300])
        history.pop()
        continue

    # 3) 逐行读 SSE（Server-Sent Events），拼出打字机效果
    print("助手: ", end="", flush=True)
    full_reply = ""
    for raw in resp.iter_lines():          # 默认返回 bytes，手动按 utf-8 解码最稳
        if not raw:
            continue
        line = raw.decode("utf-8")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()          # 去掉 "data:" 前缀
        if payload == "[DONE]":             # 流结束标记
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"]
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        piece = delta.get("content", "")
        if piece:
            print(piece, end="", flush=True)   # 收到一小段立刻打印
            full_reply += piece
    print()

    # 4) 把模型的完整回答也存进记忆 —— 下一轮它才知道自己说过什么
    history.append({"role": "assistant", "content": full_reply})

    # 5) 记忆太长会又贵又容易超上下文窗口：只留 system + 最近 20 条
    if len(history) > 21:
        history = [history[0]] + history[-20:]
