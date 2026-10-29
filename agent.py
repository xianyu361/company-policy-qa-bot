"""
Step 3: Function Calling（让模型调用工具）
运行: python agent.py
原理: 模型并不真的执行代码，而是返回"我要调用哪个函数、参数是什么"；
      由你的代码真正执行，再把结果喂回模型，它才能基于真实数据回答。
"""
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

# ---------- 1) 你本地真实可用的"工具" ----------
def get_current_time() -> str:
    """返回当前本地时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """计算一个四则运算表达式，如 '23 * 7 + 1'。"""
    try:
        # 只放行数字和运算符，避免执行任意代码（安全底线）
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "非法表达式"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算出错: {e}"

# 工具名 -> 真实函数
TOOLS_IMPL = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

# ---------- 2) 告诉模型的"工具说明书"（JSON Schema） ----------
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

    # ---------- 3) 多轮：模型可能要求调工具，我们执行后再问一次 ----------
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
                "tools": TOOLS_SCHEMA,        # 把工具说明书交给模型
                "temperature": 0.7,
            },
            timeout=60,
        )
        data = resp.json()

        # 模型回复：要么直接给答案，要么要求调用工具
        msg = data["choices"][0]["message"]

        # 情况A：模型要求调用工具
        if msg.get("tool_calls"):
            print("助手: [准备调用工具]", flush=True)
            history.append(msg)               # 必须原样存回，模型才知道自己要调什么

            for call in msg["tool_calls"]:
                fn_name = call["function"]["name"]
                fn_args = json.loads(call["function"]["arguments"])
                print(f"  -> {fn_name}({fn_args})", flush=True)

                # 真实执行本地函数
                result = TOOLS_IMPL[fn_name](**fn_args)

                # 把结果作为 tool 消息喂回模型（带上 tool_call_id 配对）
                history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result),
                })
            # 带着工具结果，再让模型生成最终回答（进入下一轮循环）
            continue

        # 情况B：模型直接给出最终回答
        print("助手:", msg["content"])
        history.append({"role": "assistant", "content": msg["content"]})
        break
    else:
        print("助手: [超出最大工具调用轮数]")
