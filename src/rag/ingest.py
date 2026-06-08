"""Load prepared chunks into the configured vector store."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from .config import load_config
from .embeddings import create_embeddings
from .retrieval.bm25 import build_bm25_index_from_documents
from .vectorstore import (
    chunks_to_documents,
    create_vectorstore,
    deduplicate_chunks,
    filter_stop_chunks,
    ingest_to_vectorstore,
    load_chunks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Load KOMPAS-3D documentation chunks into a vector store."
    )
    parser.add_argument("chunks_path", help="Path to chunks.jsonl")
    parser.add_argument("--config", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--bm25-index-path", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--backend", choices=["chroma", "qdrant"], default=None)
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only the persisted BM25 index from chunks without touching the vector store.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing local vector store directory before ingest.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config.vectorstore.backend = args.backend
    if args.persist_dir:
        config.vectorstore.chroma_persist_dir = args.persist_dir
    if args.bm25_index_path:
        config.retriever.bm25_index_path = args.bm25_index_path
    if args.collection:
        config.vectorstore.collection_name = args.collection

    chunks = load_chunks(args.chunks_path)
    if not chunks:
        logger.error("No chunks to ingest")
        sys.exit(1)

    chunks = filter_stop_chunks(chunks)
    chunks = deduplicate_chunks(chunks)
    documents, embedding_texts, ids = chunks_to_documents(chunks)
    bm25_index = build_bm25_index_from_documents(documents)
    if args.bm25_only:
        bm25_index.save(config.retriever.bm25_index_path)
        return

    embeddings = create_embeddings(config.embedding)

    if args.reset and config.vectorstore.backend == "chroma":
        db_path = Path(config.vectorstore.chroma_persist_dir)
        if db_path.exists():
            shutil.rmtree(db_path)
            logger.info("Deleted vector store directory: %s", db_path)

    vectorstore = create_vectorstore(embeddings, config.vectorstore)

    started_at = time.time()
    ingest_to_vectorstore(
        vectorstore,
        documents,
        embedding_texts,
        ids,
        args.batch_size,
    )
    elapsed = time.time() - started_at

    logger.info("Ingest time: %.1f s", elapsed)
    if elapsed > 0:
        logger.info("Throughput: %.0f docs/s", len(documents) / elapsed)

    bm25_index.save(config.retriever.bm25_index_path)


if __name__ == "__main__":
    main()
