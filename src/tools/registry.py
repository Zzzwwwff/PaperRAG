"""
工具注册
========
Function Calling Schema 定义 + 工具执行分发。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import numpy as np
from src.retrieval.retriever import search_kb
from src.retrieval.web_search import search_web
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_paper
from src.embedding.embedder import embed_chunks, encode_query
from src.storage.vector_store import add_chunks, get_stats, compute_paper_hash, hash_exists

logger = logging.getLogger(__name__)

# 已解析的文档缓存（parse_document 用）
_parsed_docs = {}


def clear_parsed_docs():
    """清空所有临时解析文档（释放内存 + 向量）"""
    global _parsed_docs
    count = len(_parsed_docs)
    _parsed_docs = {}
    logger.info(f"已清空 {count} 个临时解析文档")

# ===== Function Calling Schema =====
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "搜索本地论文知识库。用户提及'这篇''那篇'等指代时，从对话上下文推断具体论文，传入 file_ref 限定范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询关键词"},
                    "file_ref": {"type": "string", "description": "可选，限定搜索某篇论文的文件名或关键词"},
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
            "description": "【仅在用户明确要求分析/介绍某篇具体论文时使用】解析论文全文。严禁在搜索、浏览、列举文献时调用。search_kb 的结果已经足够回答大多数问题。",
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


def _search_parsed_docs(query: str, file_ref: str = None, top_k: int = 10) -> list[dict]:
    """在已解析的临时文档中检索（内存余弦相似度）
    file_ref: 可选，限定搜索特定文档的文件名或关键词
    """
    if not _parsed_docs:
        return []
    qv = encode_query(query)
    hits = []
    for key, doc in _parsed_docs.items():
        # 如果指定了 file_ref，只搜匹配的文档
        if file_ref and file_ref.lower() not in key.lower():
            continue
        if "embeddings" not in doc:
            continue
        sims = np.dot(doc["embeddings"], qv)  # 归一化后点积=余弦相似度
        top_idx = sims.argsort()[-top_k:][::-1]
        for i in top_idx:
            if sims[i] > 0.3:  # 相似度阈值
                chunk = doc["chunks"][i]
                hits.append({
                    "text": chunk["text"][:300],
                    "source": doc["paper"]["title"][:80],
                    "score": round(float(sims[i]), 3),
                    "from_upload": True,
                })
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]


def execute_tool(name, args):
    """执行工具调用，始终返回结构化结果 {success, data/error}"""
    try:
        if name == "search_kb":
            file_ref = args.get("file_ref", None)
            # 优先检索上传文档（可指定 file_ref 限定范围）
            upload_hits = _search_parsed_docs(args["query"], file_ref=file_ref)
            if upload_hits and upload_hits[0]["score"] > 0.35:
                return {"success": True, "data": [
                    {"text": h["text"][:600], "source": h["source"],
                     "score": h["score"]} for h in upload_hits[:5]
                ]}
            # 上传文档无结果 → 搜整个知识库
            kb_hits = search_kb(args["query"])
            if not kb_hits:
                return {"success": False, "error": {"code": "KB_EMPTY",
                        "message": "未找到相关内容"}}
            return {"success": True, "data": [
                {"text": h["text"][:600],
                 "source": h.get("metadata", {}).get("paper_title", ""),
                 "section": h.get("metadata", {}).get("section", ""),
                 "score": round(h["score"], 3)} for h in kb_hits
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


def _find_pdf(file_ref):
    """在 uploads/ 和 pdf_db/ 中查找匹配的 PDF"""
    from config import PDF_DIR, UPLOAD_DIR
    # 去掉 .pdf 扩展名，避免 glob 模式重复匹配（如 "x.pdf*.pdf"）
    ref = file_ref
    if ref.lower().endswith(".pdf"):
        ref = ref[:-4]
    return list(UPLOAD_DIR.glob(f"*{ref}*.pdf")) \
        + list(PDF_DIR.glob(f"*{ref}*.pdf"))


def _parse_document(file_ref):
    # 缓存命中：已解析过直接返回
    if file_ref in _parsed_docs:
        doc = _parsed_docs[file_ref]
        return {"success": True, "data": {
            "file_ref": file_ref,
            "title": doc["paper"]["title"][:100],
            "authors": doc["paper"].get("authors", "Unknown")[:100],
            "sections": len(doc["paper"].get("sections", [])),
            "chunks": len(doc["chunks"]),
            "abstract": doc["chunks"][0]["text"][:500] if doc["chunks"] else "",
            "message": f"(缓存) {doc['paper'].get('authors', '')[:30]}, 共 {len(doc['chunks'])} 个段落",
            "cached": True,
        }}

    hits = _find_pdf(file_ref)
    if not hits:
        from config import PDF_DIR, UPLOAD_DIR
        all_pdfs = list(UPLOAD_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.pdf"))
        return {"success": False, "error": {"code": "FILE_NOT_FOUND",
                "message": f"未找到匹配 '{file_ref}' 的PDF",
                "suggestion": f"可用论文: {[f.stem[:40] for f in sorted(all_pdfs)[:5]]}"}}

    path = hits[0]
    paper = parse_pdf(path)
    if not paper:
        return {"success": False, "error": {"code": "FILE_PARSE_ERROR",
                "message": f"解析失败: {path.name}"}}

    chunks = chunk_paper(paper)
    embeddings = embed_chunks(chunks)  # 临时向量化
    _parsed_docs[file_ref] = {
        "paper": paper,
        "chunks": chunks,
        "embeddings": embeddings,
    }

    # 返回标题 + 作者 + 摘要片段，方便 Agent 直接回答问题
    abstract = ""
    for sec in paper.get("sections", []):
        if "abstract" in sec.get("heading", "").lower() or "摘要" in sec.get("heading", ""):
            abstract = sec.get("content", "")[:500]
            break
    if not abstract and chunks:
        abstract = chunks[0]["text"][:500]

    return {"success": True, "data": {
        "file_ref": file_ref,
        "title": paper["title"][:100],
        "authors": paper.get("authors", "Unknown")[:100],
        "sections": len(paper.get("sections", [])),
        "chunks": len(chunks),
        "abstract": abstract,
        "message": f"已解析 {path.name}, 作者: {paper.get('authors', 'Unknown')[:50]}, 共 {len(chunks)} 个段落"
    }}


def _ingest_document(file_ref):
    import shutil
    from config import PDF_DIR, UPLOAD_DIR
    hits = _find_pdf(file_ref)
    if not hits:
        return {"success": False, "error": {"code": "FILE_NOT_FOUND",
                "message": f"未找到匹配 '{file_ref}' 的PDF"}}

    path = hits[0]
    # 若在临时上传区，先移到正式库
    if path.parent == UPLOAD_DIR:
        dest = PDF_DIR / path.name
        dest.write_bytes(path.read_bytes())
        path.unlink()
        path = dest

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

    # 入库后重建 TF-IDF 索引，确保新文档可被混合检索命中
    from src.retrieval.retriever import rebuild_tfidf
    rebuild_tfidf()

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

