"""
知识库构建脚本
==============
将 pdf_db/ 下所有 PDF 解析 → 分块 → 向量化 → 写入 ChromaDB。

用法:
    python ingest.py                  # 全量入库
    python ingest.py --skip-existing  # 跳过已入库的论文
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from config import PDF_DIR
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_paper, chunk_stats
from src.embedding.embedder import embed_chunks
from src.storage.vector_store import add_chunks, get_stats, clear, compute_paper_hash, hash_exists

logger = logging.getLogger(__name__)


def ingest_all(skip_existing: bool = True):
    """批量入库所有 PDF"""
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"没有 PDF: {PDF_DIR}")
        return

    # 已入库的论文（文件名）
    existing_names = set(get_stats().get("papers", [])) if skip_existing else set()

    logger.info(f"共 {len(pdfs)} 篇 PDF，已入库 {len(existing_names)} 篇")
    total_chunks = 0
    processed = 0
    skipped = 0

    for i, path in enumerate(pdfs, 1):
        if skip_existing and path.name in existing_names:
            logger.info(f"[{i}/{len(pdfs)}] 跳过(已入库): {path.name}")
            skipped += 1
            continue

        logger.info(f"[{i}/{len(pdfs)}] 解析: {path.name}")
        paper = parse_pdf(path)
        if not paper:
            logger.warning(f"  ✗ 解析失败: {path.name}")
            continue

        # 内容级查重（文件名不同但内容相同也能拦下）
        paper_hash = compute_paper_hash(paper["full_text"])
        if hash_exists(paper_hash):
            logger.info(f"  ✗ 跳过(内容重复): {path.name}")
            skipped += 1
            continue

        chunks = chunk_paper(paper)
        if not chunks:
            logger.warning(f"  ✗ 无内容: {path.name}")
            continue

        embeddings = embed_chunks(chunks)
        add_chunks(chunks, embeddings, paper_hash)

        total_chunks += len(chunks)
        processed += 1

    stats = get_stats()
    logger.info("=" * 50)
    logger.info(f"入库完成: 新增 {processed} 篇, 跳过 {skipped} 篇")
    logger.info(f"知识库总计: {stats['total_papers']} 篇, {stats['total_chunks']} 片段")


def main():
    parser = argparse.ArgumentParser(description="构建论文知识库")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已入库的论文")
    parser.add_argument("--no-skip", action="store_true", help="强制全量入库")
    args = parser.parse_args()

    skip = not args.no_skip
    ingest_all(skip_existing=skip)


if __name__ == "__main__":
    main()
