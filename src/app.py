"""
应用核心
========
CLI 和 Web 共用的应用层：会话管理、文档上传、临时文档方案 A。
"""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from config import UPLOAD_DIR, PDF_DIR
from src.agent import run_agent
from src.storage.vector_store import get_stats
from src.tools.registry import execute_tool

logger = logging.getLogger(__name__)


class PaperAgent:
    """论文知识库 Agent 应用层"""

    def __init__(self):
        self.messages = None        # 对话历史（滑动窗口管理）
        self.session_docs = {}      # 本次会话上传的文档索引 {关键词: 路径}

    # ===== 文档上传 =====
    def upload_pdf(self, filename: str, content: bytes) -> dict:
        """接收上传 PDF → 存到 uploads/ 临时区"""
        # 处理重名：xxx.pdf → xxx_1.pdf
        safe_name = Path(filename).name
        dest = UPLOAD_DIR / safe_name
        counter = 1
        while dest.exists():
            dest = UPLOAD_DIR / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
            counter += 1

        dest.write_bytes(content)
        # 建立会话索引（去掉后缀做关键词）
        key = dest.stem.lower()
        self.session_docs[key] = str(dest)
        logger.info(f"上传: {dest.name} → 会话文档索引 {len(self.session_docs)} 篇")

        return {"status": "ok", "filename": dest.name,
                "session_docs": list(self.session_docs.keys())}

    def clear_uploads(self):
        """进程启动时清空临时上传区"""
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.session_docs = {}
        logger.info("临时上传区已清空")

    # ===== 对话 =====
    def ask(self, user_input: str) -> tuple:
        """处理用户输入，返回 (answer, rounds)"""
        answer, self.messages = run_agent(user_input, self.messages)
        return answer

    # ===== 工具封装（供 CLI/Web 直接调用） =====
    def tool(self, name: str, args: dict) -> dict:
        """执行工具（如前端手动触发入库）"""
        return execute_tool(name, args)

    # ===== 状态 =====
    def get_stats(self) -> dict:
        """知识库统计"""
        stats = get_stats()
        stats["uploaded_files"] = len(self.session_docs)
        return stats

    def get_available_docs(self) -> dict:
        """返回可用的文档列表（正式库 + 临时区）"""
        return {
            "kb_papers": get_stats().get("papers", []),
            "uploaded": list(self.session_docs.keys()),
        }


# ===== 全局单例 =====
_agent = None


def get_agent() -> PaperAgent:
    global _agent
    if _agent is None:
        _agent = PaperAgent()
    return _agent

