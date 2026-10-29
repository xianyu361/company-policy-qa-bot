# llm_test.py —— 调用 DeepSeek API 快速测试
#
# 用法（三种任选，都不用改代码）：
#   1) 先把 Key 写进同目录下的 .env 文件：DEEPSEEK_API_KEY=sk-xxxx
#      然后在 VS Code 终端运行： python llm_test.py "用一句话解释什么是向量数据库。"
#   2) 或者在终端临时设环境变量： $env:DEEPSEEK_API_KEY="sk-xxxx"
#   3) 或直接用 VS Code 按 F5 运行（用默认问题）
#
# Key 优先级：系统环境变量 > .env 文件。密钥不会写进代码，也不会提交到 git。

import sys
import os
import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


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


def main():
    # 1) 取 API Key
    load_env_file()  # 先尝试从 .env 读取
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("未找到 DEEPSEEK_API_KEY。")
        print("请二选一：")
        print("  A) 在项目根目录的 .env 里写一行： DEEPSEEK_API_KEY=你的key")
        print('  B) 在终端执行： $env:DEEPSEEK_API_KEY="你的key"')
        sys.exit(1)

    # 2) 取命令行里的提问（没传就用默认）
    prompt = sys.argv[1] if len(sys.argv) > 1 else "用一句话解释什么是向量数据库。"

    # 3) 组装请求（DeepSeek 兼容 OpenAI 格式）
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    # 4) 发送请求
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"请求失败（HTTP {resp.status_code}）：{e}")
        print("  401 = Key 无效；402 = 余额不足；429 = 请求太频繁")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"请求失败（网络问题）：{e}")
        sys.exit(1)

    # 5) 解析并打印回复
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    print("DeepSeek：")
    print(answer)


if __name__ == "__main__":
    main()
