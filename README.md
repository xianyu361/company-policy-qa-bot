# 企业制度智能问答系统

基于大语言模型的企业内部制度问答服务，融合 **RAG 检索增强** 与 **Function Calling 工具调用**，支持多轮对话，并以 FastAPI + Gradio 提供 Web 访问。

## 功能

- **RAG 知识库问答**：制度文档切块 → 向量化 → 语义检索 → 模型基于检索内容作答，降低幻觉。
- **Function Calling**：年假计算、当前时间等工具由模型自主调用，结果本地执行后回灌对话，保证关键数据准确。
- **多轮对话**：`messages` 历史维护上下文，服务端按 `session_id` 隔离会话。
- **服务化部署**：FastAPI 提供 `POST /chat` 接口，Gradio 提供网页界面。

## 架构

```
用户请求
  └─ Gradio 界面 / FastAPI 接口
       └─ 对话编排（ReAct 工具循环）
            ├─ RAG 检索（Chroma + BGE-small-zh 向量库）
            └─ 工具调用（年假计算 / 时间查询，本地执行）
       └─ DeepSeek Chat 生成回复
```

## 技术栈

Python 3 · DeepSeek Chat（OpenAI 兼容接口） · Chroma（向量库）+ BGE-small-zh（中文嵌入） · requests + ReAct 循环 · FastAPI · Gradio

## 部署

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置密钥：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`。
3. 启动：
   ```bash
   python run.py
   ```
4. 访问 http://127.0.0.1:7860

## 目录

| 文件 | 说明 |
|------|------|
| `app.py` | FastAPI 服务、持久化知识库、工具定义与 ReAct 循环 |
| `gradio_app.py` | Gradio 网页界面（兼容 Gradio 3/4/5） |
| `run.py` | 启动入口 |
| `rag.py` / `agent.py` / `chat.py` / `llm_test.py` | 检索、工具调用、多轮对话与基础调用的实现示例 |
| `rag_rerank.py` | 重排（rerank）检索优化示例 |
| `policy_bot.py` | 早期整合原型 |

`.env`（含 API Key）与 `chroma_db/`（向量库）已被 `.gitignore` 排除，不纳入版本管理。
