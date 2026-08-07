"""
工具注册
========
Function Calling Schema 定义 + 工具执行分发。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from src.retrieval.retriever import search_kb
from src.retrieval.web_search import search_web
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_paper
from src.embedding.embedder import embed_chunks
from src.storage.vector_store import add_chunks, get_stats, compute_paper_hash, hash_exists

logger = logging.getLogger(__name__)

# 已解析的文档缓存（parse_document 用）
_parsed_docs = {}

# ===== Function Calling Schema =====
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "搜索本地论文知识库。用于学术问题、论文内容查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "联网搜索最新信息。知识库结果不足或需要实时信息时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_document",
            "description": "解析指定PDF论文。用户提及某篇论文并想了解内容时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ref": {"type": "string", "description": "文件名或关键词"},
                },
                "required": ["file_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_document",
            "description": "将PDF论文加入知识库。用户明确要求保存或入库时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ref": {"type": "string", "description": "文件名或关键词"},
                },
                "required": ["file_ref"],
            },
        },
    },
]


def execute_tool(name, args):
    """执行工具调用，始终返回结构化结果 {success, data/error}"""
    try:
        if name == "search_kb":
            hits = search_kb(args["query"])
            if not hits:
                return {"success": False, "error": {"code": "KB_EMPTY",
                        "message": "未找到相关内容"}}
            return {"success": True, "data": [
                {"text": h["text"][:300], "source": h.get("metadata", {}).get("paper_title", ""),
                 "score": round(h["score"], 3)} for h in hits
            ]}

        elif name == "search_web":
            return search_web(args["query"])

        elif name == "parse_document":
            return _parse_document(args["file_ref"])

        elif name == "ingest_document":
            return _ingest_document(args["file_ref"])

        else:
            return {"success": False, "error": {"code": "UNKNOWN_TOOL",
                    "message": f"未知工具: {name}"}}

    except Exception as e:
        logger.error(f"工具 {name} 异常: {e}")
        return {"success": False, "error": {"code": "INTERNAL_ERROR",
                "message": str(e)}}


def _parse_document(file_ref):
    from config import PDF_DIR
    hits = list(PDF_DIR.glob(f"*{file_ref}*.pdf")) or list(PDF_DIR.glob("*.pdf"))
    if not hits:
        return {"success": False, "error": {"code": "FILE_NOT_FOUND",
                "message": f"未找到匹配 '{file_ref}' 的PDF",
                "suggestion": f"可用论文: {[f.stem[:40] for f in sorted(PDF_DIR.glob('*.pdf'))[:5]]}"}}

    path = hits[0]
    paper = parse_pdf(path)
    if not paper:
        return {"success": False, "error": {"code": "FILE_PARSE_ERROR",
                "message": f"解析失败: {path.name}"}}

    _parsed_docs[file_ref] = {"paper": paper, "chunks": chunk_paper(paper)}
    return {"success": True, "data": {
        "title": paper["title"][:100],
        "sections": len(paper.get("sections", [])),
        "chunks": len(_parsed_docs[file_ref]["chunks"]),
        "message": f"已解析 {path.name}, 共 {len(_parsed_docs[file_ref]['chunks'])} 个段落"
    }}


def _ingest_document(file_ref):
    from config import PDF_DIR
    hits = list(PDF_DIR.glob(f"*{file_ref}*.pdf"))
    if not hits:
        return {"success": False, "error": {"code": "FILE_NOT_FOUND",
                "message": f"未找到匹配 '{file_ref}' 的PDF"}}

    path = hits[0]
    # 检查重复（文件名）
    stats = get_stats()
    if path.name in stats.get("papers", []):
        return {"success": False, "error": {"code": "DUPLICATE_INGEST",
                "message": f"'{path.name}' 已在知识库中"}}

    paper = parse_pdf(path)
    if not paper:
        return {"success": False, "error": {"code": "FILE_PARSE_ERROR"}}

    # 内容级查重（文件名不同但内容相同）
    paper_hash = compute_paper_hash(paper["full_text"])
    if hash_exists(paper_hash):
        return {"success": False, "error": {"code": "DUPLICATE_INGEST",
                "message": "内容相同的论文已在知识库中"}}

    chunks = chunk_paper(paper)
    embeddings = embed_chunks(chunks)
    add_chunks(chunks, embeddings, paper_hash)

    return {"success": True, "data": {
        "title": paper["title"][:100],
        "chunks": len(chunks),
        "message": f"已入库: {paper['title'][:50]}, 共 {len(chunks)} 个片段"
    }}


# ===== quick test =====
if __name__ == "__main__":
    print("=== 1. search_kb（空库测试）===")
    r = execute_tool("search_kb", {"query": "相位噪声"})
    print(r)

    print("\n=== 2. search_web ===")
    r = execute_tool("search_web", {"query": "oscillator phase noise"})
    print(f"success: {r.get('success')}, 结果数: {len(r.get('data', [])) if r.get('success') else 0}")

    print("\n=== 3. parse_document ===")
    r = execute_tool("parse_document", {"file_ref": "Demir"})
    print(r)

    print("\n=== 4. 未知工具 ===")
    r = execute_tool("nonexistent", {})
    print(r)

