"""
混合检索引擎
============
向量检索 + TF-IDF + Reranker 三级管道。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

from config import (
    TOP_K_VECTOR, TOP_K_TFIDF, FINAL_TOP_K,
    HYBRID_ALPHA, SIMILARITY_FLOOR, RERANK_MODEL,
)
from src.storage.vector_store import search as vs_search, get_all_texts, get_all_metas
from src.embedding.embedder import encode_query

logger = logging.getLogger(__name__)

_tfidf_vec = None
_tfidf_matrix = None
_reranker = None


def _init_tfidf():
    global _tfidf_vec, _tfidf_matrix
    if _tfidf_vec is not None:
        return
    texts = get_all_texts()
    if not texts:
        return
    _tfidf_vec = TfidfVectorizer(max_features=10000)
    _tfidf_matrix = _tfidf_vec.fit_transform(texts)
    logger.info(f"TF-IDF 索引: {_tfidf_matrix.shape[0]} docs, {_tfidf_matrix.shape[1]} terms")


def _get_reranker():
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def search_kb(query, final_k=None, alpha=None, floor=None):
    """
    混合检索主入口。

    流程: 向量检索 Top-20 + TF-IDF Top-20 → 加权融合 → Reranker → Top-5
    返回: [{text, paper_title, section, score, source_file}, ...]
    """
    if final_k is None:
        final_k = FINAL_TOP_K
    if alpha is None:
        alpha = HYBRID_ALPHA
    if floor is None:
        floor = SIMILARITY_FLOOR

    # 1. 向量检索
    qv = encode_query(query)
    vec_hits = vs_search(qv, top_k=TOP_K_VECTOR)

    if not vec_hits:
        return []

    # 2. TF-IDF 检索
    _init_tfidf()
    tfidf_hits = _tfidf_search(query, top_k=TOP_K_TFIDF) if _tfidf_vec else []

    # 3. 融合（先归一化到同一尺度）
    vec_hits = _normalize_scores(vec_hits)
    tfidf_hits = _normalize_scores(tfidf_hits)
    merged = _merge(vec_hits, tfidf_hits, alpha)

    # 4. 相似度过滤
    candidates = [m for m in merged if m["score"] >= floor]

    if not candidates:
        return []

    # 5. Reranker 精排
    reranked = _rerank(query, candidates)
    return reranked[:final_k]


def _tfidf_search(query, top_k):
    """TF-IDF 关键词检索"""
    q_vec = _tfidf_vec.transform([query])
    scores = cosine_similarity(q_vec, _tfidf_matrix)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    metas = get_all_metas()
    texts = get_all_texts()
    hits = []
    for i in top_idx:
        if scores[i] > 0:
            meta = metas[i] if i < len(metas) else {}
            hits.append({
                "text": texts[i] if i < len(texts) else "",
                "metadata": meta,
                "score": float(scores[i]),
                "source": "tfidf",
            })
    return hits


def _normalize_scores(hits):
    """Min-Max 归一化到 [0, 1]"""
    if not hits:
        return hits
    scores = [h["score"] for h in hits]
    smin, smax = min(scores), max(scores)
    if smax == smin:
        return hits
    for h in hits:
        h["score"] = (h["score"] - smin) / (smax - smin)
    return hits


def _merge(vec_hits, tfidf_hits, alpha):
    """加权合并两路结果"""
    merged = {}
    # 向量分数 × alpha
    for h in vec_hits:
        cid = h["id"]
        merged[cid] = {"text": h["text"], "metadata": h["metadata"],
                       "score": h["score"] * alpha, "source": "vector"}

    # TF-IDF 分数 × (1-alpha)
    for i, h in enumerate(tfidf_hits):
        cid = h["metadata"].get("source_file", "") + f"_tfidf_{i}"
        tfidf_score = h["score"] * (1 - alpha)
        if cid in merged:
            merged[cid]["score"] += tfidf_score
            merged[cid]["source"] = "hybrid"
        else:
            merged[cid] = {"text": h["text"], "metadata": h["metadata"],
                           "score": tfidf_score, "source": "tfidf"}

    sorted_items = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return sorted_items


def _rerank(query, candidates):
    """Cross-Encoder 精排"""
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ===== quick test =====
if __name__ == "__main__":
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_paper
    from src.embedding.embedder import embed_chunks
    from src.storage.vector_store import add_chunks, clear, get_stats
    from config import PDF_DIR

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    paper = parse_pdf(pdfs[0])
    chunks = chunk_paper(paper)
    embeddings = embed_chunks(chunks)
    add_chunks(chunks, embeddings)

    # query = paper['title'][:30]
    query = "低比特量化技术"
    hits = search_kb(query)
    print(f"查询: {query}")
    for h in hits:
        print(f"  [{h['score']:.4f}] {h['text'][:60].replace(chr(10),' ')}")

    clear()

