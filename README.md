# 📚 PaperMate — 论文知识库 RAG 智能体（未完成版本）

基于 **ReAct Agent + 混合检索 + Reranker** 的个人论文知识库问答系统。中英文论文 PDF 自动入库，自然语言问答，支持联网兜底、文档入库、Web 界面。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 ReAct Agent | DeepSeek Function Calling 自主决策，Think → Act → Observe 循环 |
| 🔍 三级检索管道 | Query 改写 → 混合检索(Top-20) → Reranker 精排(Top-5) |
| 🌐 联网兜底 | 知识库不足时自动 Tavily 搜索，可一键入库 |
| 📄 PDF 解析 | 中英文双栏排版、公式、图表标题提取 |
| 💾 本地向量库 | ChromaDB 嵌入式，无额外服务 |
| 🔬 模型微调 | Embedding 对比学习 + LLM DPO 偏好对齐 |
| 🌍 Web 界面 | FastAPI + 静态前端 |

## 🏗️ 架构

```
用户输入 → ReAct Agent (DeepSeek) 
    ├── search_kb()     → Query改写 → 向量+TF-IDF → Reranker
    ├── search_web()    → Tavily 联网搜索
    ├── parse_document() → PDF 解析 + 分块
    └── ingest_document() → 论文永久入库
```

## 📂 项目结构

```
agentP2/
├── pdf_db/                  # 论文 PDF
├── src/
│   ├── ingestion/           # PDF 解析 + 分块
│   ├── embedding/           # bge-small-zh 向量化
│   ├── storage/             # ChromaDB 增删查
│   ├── retrieval/           # 混合检索 + Tavily 联网
│   ├── tools/               # Function Calling 工具注册
│   ├── agent.py             # ReAct 循环引擎
│   └── app.py               # 应用核心
├── finetune/                # 模型微调（Embedding + DPO）
├── config.py                # 全局配置
├── main.py                  # CLI 入口
└── server.py                # FastAPI 入口
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 网络（DeepSeek API + Tavily）

### 安装

```bash
# 1. 创建环境
conda create -n agentp2 python=3.12 -y
conda activate agentp2

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置密钥（复制 .env.example 并填入）
cp .env.example .env
# 填入 OPENAI_API_KEY（DeepSeek）和 TAVILY_API_KEY

# 4. 放入论文
# 将 PDF 论文放入 pdf_db/

# 5. 启动
python main.py               # CLI 交互
python server.py             # Web 界面 (http://localhost:8000)
```

### CLI 用法

```
$ python main.py

> BERT 的预训练任务有哪些？        # 自动检索知识库
> 最新的多模态大模型进展？          # 知识库不足 → 自动联网
> 分析一下 attention 论文          # 解析指定 PDF
> 把 attention 论文加进知识库      # 永久入库
```