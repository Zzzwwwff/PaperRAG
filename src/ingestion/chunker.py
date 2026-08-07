"""
论文感知分块器
==============
按论文章节 → 段落 → 句子 递进切分，相邻 chunk 有重叠。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

logger = logging.getLogger(__name__)


def _recursive_split(text, separators=None,
                     chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """递归语义分割：段落→句子→字符 递进切分"""
    if separators is None:
        separators = CHUNK_SEPARATORS
    if not separators or not text.strip():
        return [text] if text.strip() else []

    sep = separators[0]
    remaining = separators[1:]

    if sep == "":
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - chunk_overlap
            if start >= len(text):
                break
        return chunks

    parts = text.split(sep)
    result = []
    buf = ""

    for part in parts:
        candidate = buf + (sep if buf else "") + part
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf.strip():
                result.append(buf.strip())
            if len(part) > chunk_size:
                result.extend(_recursive_split(part, remaining, chunk_size, chunk_overlap))
                buf = ""
            else:
                buf = part

    if buf.strip():
        result.append(buf.strip())
    return result


def chunk_paper(paper):
    """对单篇论文分块，保留元数据"""
    chunks = []
    idx = 0
    sections = paper.get("sections", [])

    for sec in (sections or [{"heading": "", "content": paper.get("full_text", ""), "has_formula": False}]):
        content = sec.get("content", sec.get("full_text", ""))
        if not content.strip():
            continue
        heading = sec.get("heading", "")
        has_f = sec.get("has_formula", False)
        for text in _recursive_split(content):
            idx += 1
            chunks.append({
                "text": text.strip(),
                "paper_title": paper.get("title", "")[:100],
                "authors": paper.get("authors", "")[:100],
                "source_file": paper.get("source_file", ""),
                "section": heading[:80],
                "chunk_index": idx,
                "has_formula": has_f,
                "char_count": len(text),
            })
    return chunks


def chunk_stats(chunks):
    if not chunks:
        return {}
    sizes = [c["char_count"] for c in chunks]
    return {
        "total_chunks": len(chunks),
        "papers": len(set(c["paper_title"] for c in chunks)),
        "avg_size": round(sum(sizes) / len(sizes)),
        "min_size": min(sizes),
        "max_size": max(sizes),
    }


# ===== 快速测试 =====
if __name__ == "__main__":
    from src.ingestion.pdf_parser import parse_pdf
    from config import PDF_DIR

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("no PDF")
        exit(1)

    paper = parse_pdf(pdfs[30])
    if not paper:
        print("parse failed")
        exit(1)

    chunks = chunk_paper(paper)
    stats = chunk_stats(chunks)
    print(f"论文: {paper['title'][:50]}")
    print(f"分块: {stats['total_chunks']} chunks, 平均 {stats['avg_size']} 字\n")
    for c in chunks[:3]:
        p = c["text"][:80].replace("\n", " ")
        print(f"  [{c['section'][:20]}] {p}...")
    if len(chunks) > 3:
        print(f"  ... 共 {len(chunks)} 个")
