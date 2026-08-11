# 📚 PaperMate — 论文知识库 RAG 智能体

基于 **ReAct Agent + 混合检索 + Reranker** 的个人论文知识库问答系统。中英文论文 PDF 入库，自然语言问答，支持联网兜底、临时文档上传、Web 流式界面。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 ReAct Agent | DeepSeek Function Calling 自主决策，防死循环保护 |
| 🔍 混合检索 | 向量 + TF-IDF 双路召回 → Reranker Cross-Encoder 精排(Top-5) |
| 🌐 联网兜底 | 知识库不足时自动 Tavily 搜索 |
| 📄 PDF 解析 | pdfplumber 解析中英文双栏排版、公式/图表标题提取 |
| 📤 临时文档 | 上传 PDF → 自动解析 + 内存向量化 → 会话内问答，无需入库 |
| 💾 本地向量库 | ChromaDB 嵌入式 + 内容哈希去重 |
| 🌍 Web 界面 | FastAPI + SSE 流式输出 + Markdown/KaTeX 数学公式渲染 |
| 🛡️ 生产加固 | 工具调用配额、死循环检测、ChatCompletionMessage 序列化兼容 |

## 🏗️ 架构

```
用户输入 → ReAct Agent (DeepSeek Function Calling)
    ├── search_kb()     → 向量检索 + TF-IDF → Reranker → Top-5
    ├── search_web()    → Tavily 联网搜索
    ├── parse_document() → PDF 解析 + 内存向量化（会话内可用）
    └── ingest_document() → 正式入库 + TF-IDF 索引重建
```

## 📂 项目结构

```
agentP2/
├── pdf_db/                  # 正式论文库 PDF
├── uploads/                 # 临时上传区（启动清空）
├── chroma_db/               # ChromaDB 持久化
├── logs/                    # 日志（RotatingFileHandler，5MB×3）
├── static/                  # Web 前端
│   └── index.html           # SSE 流式 Chat UI
├── src/
│   ├── ingestion/           # PDF 解析 + 语义分块
│   ├── embedding/           # BAAI/bge-small-zh-v1.5 向量化
│   ├── storage/             # ChromaDB 增删查 + 去重
│   ├── retrieval/           # 混合检索 + Reranker + Tavily 联网
│   ├── tools/               # Function Calling 工具注册 + 执行
│   ├── agent.py             # ReAct 流式循环引擎 + 防死循环
│   └── app.py               # 应用层（会话管理、文档上传）
├── config.py                # 全局配置
├── ingest.py                # 批量入库脚本
├── main.py                  # CLI 入口（prompt_toolkit）
├── server.py                # FastAPI 入口（SSE 流式）
├── requirements.txt
└── .env.example
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- DeepSeek API Key + Tavily API Key（免费 1000 次/月）

### 安装

```bash
# 1. 创建环境
conda create -n agentp2 python=3.12 -y
conda activate agentp2

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置密钥
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 和 TAVILY_API_KEY
# 国内用户加 HF_HUB_OFFLINE=1（模型已在 .cache 缓存时）

# 4. 放入论文 PDF 到 pdf_db/，然后：
python ingest.py              # 批量入库（自动去重）

# 5. 启动
python main.py                # CLI 交互
python server.py              # Web 界面 http://localhost:8000
```

### CLI 用法

```
$ python main.py

> 知识库有哪些关于 OTFS 的论文        # 自动检索知识库，列出结果
> 这篇论文的通讯作者是谁              # 上传后指代消解，精准定位
> /upload ~/paper.pdf                # 上传 + 自动解析 + 通知 Agent
> /status                            # 查看知识库统计
> /quit                              # 退出（自动清理临时文件）
```

### Web 界面

- **流式输出**：SSE 逐 token 推送，实时显示工具调用状态
- **Markdown + 数学公式**：marked.js 渲染表格/代码块，KaTeX 渲染 LaTeX
- **上传即解析**：上传 PDF 后自动向量化，可立即问答
- **对话恢复**：刷新页面自动加载历史记录
- **输入法兼容**：中文输入法 Enter 不会误触发发送

### 工具调用配额

| 工具 | 单轮对话上限 |
|------|-------------|
| search_kb | 4 次 |
| search_web | 2 次 |
| parse_document | 2 次 |
| 相同参数重复 | 1 次（第 2 次拦截） |