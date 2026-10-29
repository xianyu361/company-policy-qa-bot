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

    load_env_file()                
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("未找到 DEEPSEEK_API_KEY。")
        print("请二选一：")
        print("  A) 在项目根目录的 .env 里写一行： DEEPSEEK_API_KEY=你的key")
        print('  B) 在终端执行： $env:DEEPSEEK_API_KEY="你的key"')
        sys.exit(1)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "用一句话解释什么是向量数据库。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

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

    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    print("DeepSeek：")
    print(answer)

if __name__ == "__main__":
    main()
