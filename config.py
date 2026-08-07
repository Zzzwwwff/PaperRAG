"""
============================================================================
 论文知识库 RAG 智能体 —— 全局配置文件
============================================================================
 所有模块的统一配置入口。优先级: 环境变量 > 默认值
 使用:  from config import PDF_DIR, LLM_MODEL, ...
============================================================================
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---- 加载 .env 文件 ----
load_dotenv()

# ============================================================================
#  项目路径
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
PDF_DIR = PROJECT_ROOT / os.getenv("PDF_DIR", "pdf_db")
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"
LOG_DIR = PROJECT_ROOT / "logs"

# 自动创建必要目录
for d in [PDF_DIR, VECTOR_DB_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
#  LLM 配置
# ============================================================================
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# ============================================================================
#  Embedding 模型 (本地)
# ============================================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


# ============================================================================
#  Reranker 模型 (本地)
# ============================================================================
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")


# ============================================================================
#  PDF 解析
# ============================================================================
PDF_PARSER = os.getenv("PDF_PARSER", "pdfplumber")    # "pdfplumber" | "pymupdf"
MIN_PARAGRAPH_LEN = int(os.getenv("MIN_PARAGRAPH_LEN", "20"))
REMOVE_HEADER_FOOTER = os.getenv("REMOVE_HEADER_FOOTER", "true").lower() == "true"
REMOVE_REFERENCES = os.getenv("REMOVE_REFERENCES", "true").lower() == "true"


# ============================================================================
#  文档分块
# ============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
# 分隔符优先级: 段落 → 句子 → 字符
CHUNK_SEPARATORS = ["\n\n", "\n", "。", ". ", "！", "？", "；", ";", " ", ""]


# ============================================================================
#  检索管道
# ============================================================================
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "20"))      # 向量检索粗排数量
TOP_K_TFIDF = int(os.getenv("TOP_K_TFIDF", "20"))        # TF-IDF 粗排数量
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))         # Reranker 后最终数量
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))   # 向量:TF-IDF 融合权重
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.35"))  # 最低相似度


# ============================================================================
#  ReAct Agent 循环
# ============================================================================
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "5"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))        # 单次 LLM 调用超时(秒)
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "15"))      # 单次工具调用超时(秒)


# ============================================================================
#  上下文窗口 (滑动窗口)
# ============================================================================
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "8"))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "6000"))


# ============================================================================
#  联网搜索
# ============================================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))


# ============================================================================
#  日志
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

CONSOLE_LOG_FMT = (
    "%(asctime)s | %(levelname)-7s | %(name)-20s | "
    "%(funcName)-15s | %(message)s"
)
FILE_LOG_FMT = (
    "%(asctime)s | %(levelname)-7s | %(name)-24s | "
    "%(funcName)s:%(lineno)-3d | %(message)s"
)
LOG_DATE_FMT = "%m-%d %H:%M:%S"


def setup_logging():
    """初始化日志：控制台 + 文件（文件始终记 DEBUG）"""
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_LEVEL)
    console.setFormatter(logging.Formatter(CONSOLE_LOG_FMT, LOG_DATE_FMT))

    handlers = [console]

    if LOG_DIR:
        file_handler = logging.FileHandler(
            LOG_DIR / "agent.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_LOG_FMT, LOG_DATE_FMT))
        handlers.append(file_handler)

    logging.basicConfig(level=LOG_LEVEL, handlers=handlers, force=True)

    logger = logging.getLogger(__name__)
    logger.info(f"配置加载完成 | 模型: {LLM_MODEL} | PDF目录: {PDF_DIR}")


# ============================================================================
#  System Prompt
# ============================================================================
SYSTEM_PROMPT = """你是一个论文知识库助手。你可以使用以下工具:

- search_kb(query): 搜索本地论文知识库
- search_web(query): 联网搜索最新信息
- parse_document(file_ref): 解析指定PDF论文内容
- ingest_document(file_ref): 将论文永久加入知识库

工作方式(ReAct):
1. 收到用户问题 → 思考需要什么信息
2. 调用合适的工具获取信息
3. 观察工具返回结果
4. 如果信息不够，继续调用工具
5. 信息足够后，给出最终答案

规则:
- 已知的常识性问题可以直接回答，但涉及实时信息（天气、新闻、最新进展）应调用 search_web
- 学术问题优先搜索知识库，结果不理想再联网
- 涉及具体论文名称时，先用 parse_document 解析
- 入库操作必须在最终答案中向用户确认
- 如果知识库和网络都找不到答案，诚实告知
"""

RAG_PROMPT_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请依据上述参考资料回答，注明来源编号。"""


# ============================================================================
#  启动时自动执行
# ============================================================================
setup_logging()

