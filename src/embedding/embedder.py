"""
向量化引擎
==========
加载 bge-small-zh-v1.5，批量编码和单条查询编码。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, EMBED_DIM, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def encode(texts, batch_size=None):
    if batch_size is None:
        batch_size = EMBED_BATCH_SIZE
    model = get_model()
    return model.encode(
        texts, batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )


def encode_query(query):
    return encode([query])[0]


def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    if not texts:
        return np.empty((0, EMBED_DIM))
    embeddings = encode(texts)
    logger.info(f"Embedded {len(texts)} chunks -> {embeddings.shape}")
    return embeddings


# ===== quick test =====
if __name__ == "__main__":
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_paper, chunk_stats
    from config import PDF_DIR

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    paper = parse_pdf(pdfs[30])
    chunks = chunk_paper(paper)
    stats = chunk_stats(chunks)

    print(f"论文: {paper['title'][:50]}")
    print(f"分块: {stats['total_chunks']} chunks, 平均 {stats['avg_size']} 字")

    embeddings = embed_chunks(chunks)
    print(f"向量: {embeddings.shape}")

    # 相似度测试
    q = "Message Passing"
    qv = encode_query(q)
    sims = np.dot(embeddings, qv)
    top = np.argsort(sims)[-3:][::-1]
    print(f"\nQuery: \"{q}\" → Top-3:")
    for i in top:
        print(f"  [{sims[i]:.4f}] {chunks[i]['text'][:80].replace(chr(10),' ')}")
