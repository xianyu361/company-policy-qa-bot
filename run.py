"""
一键启动脚本：替你处理好 gradio 界面启动的两个坑。
用法（先激活虚拟环境）：
    python run.py
然后浏览器打开 http://127.0.0.1:7860

为什么需要这个脚本：
1) 你机器上有 HTTPS_PROXY 等代理变量，会把本地 127.0.0.1 请求投到代理导致连不上。
   这里在 Python 进程内清掉这些变量（连同大小写两种写法）。
2) bge 嵌入模型已缓存到本地，但 gradio/sentence-transformers 启动时仍会尝试连
   HuggingFace 拉额外文件，国内网络会卡死。设 HF_HUB_OFFLINE=1 让它完全离线，
   几秒就能起，不再卡在重试。
"""
import os

# —— 坑1：清掉所有代理变量（大小写都清），否则本地端口被代理拦截 ——
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
          "http_proxy", "https_proxy", "all_proxy", "no_proxy"]:
    os.environ.pop(k, None)

# —— 坑2：离线模式，避免启动卡在连 HuggingFace ——
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# 必须在 import gradio_app 之前设好上面的环境变量
import gradio_app

if __name__ == "__main__":
    # 模型加载约需 20~30 秒，看到 "Running on" 后再开浏览器
    # server_name="0.0.0.0" 绑定所有网卡，支持用局域网 IP 访问（可绕过某些代理对 localhost 的劫持）
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        print(f"局域网访问地址（如代理拦截 localhost，可试这个）: http://{lan_ip}:7860")
    except Exception:
        pass
    gradio_app.demo.launch(server_port=7860, server_name="0.0.0.0", inbrowser=False, share=False)
