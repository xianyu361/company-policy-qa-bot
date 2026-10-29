# 企业制度智能问答系统（RAG + Function Calling）

基于大语言模型（DeepSeek）的企业内部制度问答机器人，融合 **RAG 检索增强** 与 **Function Calling 工具调用**，支持多轮对话，并以 FastAPI + Gradio 封装为可在浏览器访问的 Web 服务。

## 功能特性
- **RAG 知识库问答**：制度文档切块 → 向量化 → 语义检索 → 模型基于「资料 + 问题」作答，显著降低幻觉。
- **Function Calling 工具调用**：年假计算、当前时间查询等工具，模型自主发起调用，代码本地执行后将真实结果回灌对话（会算数、不瞎编）。
- **多轮对话**：通过 `messages` 历史列表维护记忆；服务端按 `session_id` 隔离不同用户会话。
- **工程化部署**：FastAPI 暴露 `POST /chat` 接口，Gradio 提供网页界面，浏览器可直接访问演示。

## 技术栈
Python 3 · DeepSeek Chat（OpenAI 兼容接口） · Chroma 向量库 + BGE-small-zh 中文嵌入模型 · 原生 `requests` + ReAct 工具循环 · FastAPI · Gradio

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置密钥：复制 `.env.example` 为 `.env`，填入你的 Key：
   ```bash
   DEEPSEEK_API_KEY=你的key
   ```
3. 运行：
   ```bash
   python run.py
   ```
4. 浏览器打开 http://127.0.0.1:7860

## 目录说明
| 文件 | 说明 |
|------|------|
| `app.py` | FastAPI 服务 + 持久化知识库 + 工具 + ReAct 循环 |
| `gradio_app.py` | Gradio 网页界面（自动适配 Gradio 3/4/5 版本） |
| `run.py` | 一键启动（自动清除代理变量 + 离线模式，规避本地连不上） |
| `rag.py` / `agent.py` / `chat.py` / `llm_test.py` | 分阶段学习脚本（RAG / 工具调用 / 多轮对话 / 首测） |
| `rag_rerank.py` | rerank 重排优化演示 |
| `policy_bot.py` | 早期整合小项目 |

> 注意：`.env`（含 API Key）与 `chroma_db/`（向量库）已通过 `.gitignore` 排除，不会入库。
