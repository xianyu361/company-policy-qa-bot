"""
项目打磨点 2：漂亮网页界面（gradio）
复用 app.py 里已持久化的知识库 + RAG + 工具调用 + 多轮对话逻辑，
套一个自带聊天气泡、主题美观的网页界面，比 app.py 的纯 HTML 更适合演示给面试官看。

运行前: pip install gradio
运行:   python gradio_app.py
然后浏览器打开终端给出的地址（默认 http://127.0.0.1:7860）
"""
import gradio as gr
from app import kb, SYSTEM_PROMPT, ask_model

# —— 版本自适应：Gradio 4+ 用 messages 格式，否则用旧版元组格式 ——
GRADIO_MAJOR = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 3
USE_MESSAGES = GRADIO_MAJOR >= 4

def respond(message, chat_history, model_messages):
    # model_messages 是喂给大模型的完整 messages（含 system + 每轮），用 gr.State 隔离每个会话
    if model_messages is None:
        model_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # RAG：先检索相关资料
    hits = kb.query(query_texts=[message], n_results=3)
    context = "\n".join(hits["documents"][0])
    augmented = f"【相关资料】\n{context}\n【用户问题】{message}"
    model_messages.append({"role": "user", "content": augmented})
    # 跑 ReAct 循环（含工具调用），返回最终回复
    reply = ask_model(model_messages)
    # 按 Gradio 版本追加对话：4+ 用 messages 字典，3.x 用 (用户, 机器人) 元组
    if USE_MESSAGES:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
    else:
        chat_history.append((message, reply))
    return "", chat_history, model_messages

with gr.Blocks(title="公司制度问答机器人") as demo:
    gr.Markdown("# 公司制度问答机器人\n基于 RAG 检索增强 + Function Calling 工具调用的智能问答")
    # Gradio 3.x 默认元组格式；4.x 默认 messages 格式；5.x 移除了 type 参数。
    # 故统一不传 type（靠各版本默认行为），仅 chat_history 格式按版本区分。
    chatbot = gr.Chatbot(label="对话", height=420)
    state = gr.State(None)  # 存放每个会话的 model_messages
    with gr.Row():
        txt = gr.Textbox(placeholder="输入问题，回车发送（如：我入职2年，年假几天？）", scale=4, label="你的问题")
        clear = gr.Button("清空对话", scale=1)
    txt.submit(respond, [txt, chatbot, state], [txt, chatbot, state])
    clear.click(lambda: ([], None), None, [chatbot, state])

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=False, theme=gr.themes.Soft())
