import gradio as gr
from app import kb, SYSTEM_PROMPT, ask_model

GRADIO_MAJOR = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 3
USE_MESSAGES = GRADIO_MAJOR >= 4

def respond(message, chat_history, model_messages):

    if model_messages is None:
        model_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    hits = kb.query(query_texts=[message], n_results=3)
    context = "\n".join(hits["documents"][0])
    augmented = f"【相关资料】\n{context}\n【用户问题】{message}"
    model_messages.append({"role": "user", "content": augmented})

    reply = ask_model(model_messages)

    if USE_MESSAGES:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
    else:
        chat_history.append((message, reply))
    return "", chat_history, model_messages

with gr.Blocks(title="公司制度问答机器人") as demo:
    gr.Markdown("# 公司制度问答机器人\n基于 RAG 检索增强 + Function Calling 工具调用的智能问答")

    chatbot = gr.Chatbot(label="对话", height=420)
    state = gr.State(None)                          
    with gr.Row():
        txt = gr.Textbox(placeholder="输入问题，回车发送（如：我入职2年，年假几天？）", scale=4, label="你的问题")
        clear = gr.Button("清空对话", scale=1)
    txt.submit(respond, [txt, chatbot, state], [txt, chatbot, state])
    clear.click(lambda: ([], None), None, [chatbot, state])

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=False, theme=gr.themes.Soft())
