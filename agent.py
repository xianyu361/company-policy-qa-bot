import datetime
import json
import os
import sys

import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请先设置环境变量 DEEPSEEK_API_KEY")
    sys.exit(1)

API_URL = "https://api.deepseek.com/chat/completions"

def get_current_time() -> str:
    """返回当前本地时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """计算一个四则运算表达式，如 '23 * 7 + 1'。"""
    try:

        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "非法表达式"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算出错: {e}"

TOOLS_IMPL = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个四则运算表达式，例如 '23 * 7 + 1'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要算的算术表达式",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

history = [
    {"role": "system", "content": "你是一个会用工具的中文助手。需要实时信息或计算时，优先调用工具。"},
]

print("Agent 已启动（输入 exit 退出）")
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

    history.append({"role": "user", "content": user_input})

    MAX_ROUNDS = 5
    for _ in range(MAX_ROUNDS):
        resp = requests.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": "deepseek-chat",
                "messages": history,
                "tools": TOOLS_SCHEMA,                    
                "temperature": 0.7,
            },
            timeout=60,
        )
        data = resp.json()

        msg = data["choices"][0]["message"]

        if msg.get("tool_calls"):
            print("助手: [准备调用工具]", flush=True)
            history.append(msg)                                   

            for call in msg["tool_calls"]:
                fn_name = call["function"]["name"]
                fn_args = json.loads(call["function"]["arguments"])
                print(f"  -> {fn_name}({fn_args})", flush=True)

                result = TOOLS_IMPL[fn_name](**fn_args)

                history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result),
                })

            continue

        print("助手:", msg["content"])
        history.append({"role": "assistant", "content": msg["content"]})
        break
    else:
        print("助手: [超出最大工具调用轮数]")
