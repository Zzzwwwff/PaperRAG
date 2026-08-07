"""
ChromaDB 向量存储
=================
增、删、查、统计的简单封装。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import chromadb
from chromadb.config import Settings
from config import VECTOR_DB_DIR, EMBED_DIM

logger = logging.getLogger(__name__)

# 抑制 ChromaDB 遥测日志（模块加载时生效）
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

COLLECTION_NAME = "papers"

_client = None


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks, embeddings):
    """批量写入 chunks（文本 + 向量 + 元数据）"""
    col = _get_collection()
    ids = []
    docs = []
    metas = []
    vecs = []
    for chunk, emb in zip(chunks, embeddings):
        cid = f"{chunk['source_file']}_{chunk['chunk_index']}"
        ids.append(cid)
        docs.append(chunk["text"])
        metas.append({
            "paper_title": chunk.get("paper_title", ""),
            "authors": chunk.get("authors", ""),
            "source_file": chunk.get("source_file", ""),
            "section": chunk.get("section", ""),
            "chunk_index": chunk["chunk_index"],
            "has_formula": chunk.get("has_formula", False),
        })
        vecs.append(emb.tolist())
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
    logger.info(f"入库: {len(chunks)} chunks")


def search(query_embedding, top_k=20):
    """向量检索，返回 [{id, text, metadata, score}, ...]"""
    col = _get_collection()
    results = col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    if not results["ids"][0]:
        return hits
    for i, cid in enumerate(results["ids"][0]):
        dist = results["distances"][0][i]
        hits.append({
            "id": cid,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - dist,  # cosine distance → similarity
        })
    return hits


def get_all_texts():
    """获取所有 chunk 文本（给 TF-IDF 建索引用）"""
    col = _get_collection()
    return col.get(include=["documents"])["documents"] or []


def get_all_metas():
    """获取所有 chunk 元数据"""
    col = _get_collection()
    return col.get(include=["metadatas"])["metadatas"] or []


def delete_paper(source_file):
    """删除某篇论文的全部 chunks"""
    col = _get_collection()
    all_ids = col.get()["ids"]
    target = [i for i in all_ids if i.startswith(source_file)]
    if target:
        col.delete(ids=target)
        logger.info(f"删除: {source_file} ({len(target)} chunks)")


def get_stats():
    """知识库统计"""
    col = _get_collection()
    metas = col.get(include=["metadatas"])["metadatas"] or []
    papers = set(m.get("source_file", "") for m in metas)
    return {"total_chunks": len(metas), "total_papers": len(papers), "papers": sorted(papers)}


def clear():
    """清空知识库"""
    col = _get_collection()
    all_ids = col.get()["ids"]
    if all_ids:
        col.delete(ids=all_ids)
    logger.info("知识库已清空")


# ===== quick test =====
if __name__ == "__main__":
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_paper
    from src.embedding.embedder import embed_chunks, encode_query
    from config import PDF_DIR

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    paper = parse_pdf(pdfs[0])
    chunks = chunk_paper(paper)
    embeddings = embed_chunks(chunks)

    add_chunks(chunks, embeddings)
    stats = get_stats()
    print(f"库: {stats['total_papers']} 篇, {stats['total_chunks']} chunks")

    qv = encode_query(paper['title'][:30])
    hits = search(qv, top_k=3)
    print(f"\n查询: {paper['title'][:40]}")
    for h in hits:
        print(f"  [{h['score']:.4f}] {h['text'][:60]}")

    # 清理测试数据
    clear()
    print("\n测试数据已清空")

