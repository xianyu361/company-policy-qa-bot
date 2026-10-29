import os

for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
          "http_proxy", "https_proxy", "all_proxy", "no_proxy"]:
    os.environ.pop(k, None)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio_app

if __name__ == "__main__":

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
