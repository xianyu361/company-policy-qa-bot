import json
import os
import sys

import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请先设置环境变量 DEEPSEEK_API_KEY")
    sys.exit(1)

API_URL = "https://api.deepseek.com/chat/completions"

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
        history = history[:1]                       
        print("（记忆已清空）")
        continue

    history.append({"role": "user", "content": user_input})

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
                "stream": True,                  
            },
            stream=True,                                        
            timeout=60,
        )
    except requests.RequestException as e:
        print("网络错误:", e)
        history.pop()                           
        continue

    if resp.status_code != 200:
        print("请求失败 HTTP", resp.status_code)
        print(resp.text[:300])
        history.pop()
        continue

    print("助手: ", end="", flush=True)
    full_reply = ""
    for raw in resp.iter_lines():                                     
        if not raw:
            continue
        line = raw.decode("utf-8")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()                         
        if payload == "[DONE]":                    
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"]
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        piece = delta.get("content", "")
        if piece:
            print(piece, end="", flush=True)              
            full_reply += piece
    print()

    history.append({"role": "assistant", "content": full_reply})

    if len(history) > 21:
        history = [history[0]] + history[-20:]
