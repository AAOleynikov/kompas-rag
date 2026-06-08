"""Single assembly point for the RAG runtime pipeline."""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

from .chain import KompasRAGChain, create_llm
from .config import RAGConfig
from .embeddings import create_embeddings
from .query_rewriter import rewrite_query
from .retriever import KompasRetriever, load_bm25_index
from .vectorstore import create_vectorstore, get_vectorstore_count

logger = logging.getLogger(__name__)


@dataclass
class BuiltRAGPipeline:
    """Container with the assembled RAG runtime and its reusable components."""

    config: RAGConfig
    chain: KompasRAGChain
    retriever: KompasRetriever
    llm: object
    vectorstore: object
    embeddings: object
    bm25_index: object | None = None
    reranker: object | None = None


def build_rag_pipeline(
    config: Optional[RAGConfig] = None,
    *,
    require_non_empty_store: bool = True,
) -> BuiltRAGPipeline:
    """Build the runtime RAG pipeline from one config object.

    This is intentionally scoped to runtime RAG only. It does not parse HTML,
    rebuild chunks, mutate Chroma artifacts, or touch evaluation outputs.
    """
    config = config or RAGConfig()

    embeddings = create_embeddings(config.embedding)
    vectorstore = create_vectorstore(embeddings, config.vectorstore)
    count = get_vectorstore_count(vectorstore)
    if require_non_empty_store and count == 0:
        logger.error(
            "Vector store is empty. Load chunks first, for example: "
            "python -m rag.ingest data/processed/chunks.jsonl"
        )
        sys.exit(1)

    try:
        bm25_index = load_bm25_index(
            config.retriever.bm25_index_path,
            expected_doc_count=count,
        )
    except Exception as exc:
        logger.error(
            "Could not load BM25 index from %s: %s. Run rag.ingest to build "
            "both Chroma and BM25 indexes.",
            config.retriever.bm25_index_path,
            exc,
        )
        sys.exit(1)

    llm = create_llm(config.llm)
    query_rewriter_fn = lambda query: rewrite_query(
        query,
        llm,
        max_queries=getattr(config.query_rewrite, "max_queries", 3),
    )
    if not getattr(config.query_rewrite, "enabled", True):
        query_rewriter_fn = None

    reranker = None
    if config.reranker.enabled:
        from .reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model_name=config.reranker.model_name,
            device=config.reranker.device,
            max_length=config.reranker.max_length,
            batch_size=config.reranker.batch_size,
        )

    retriever = KompasRetriever(
        vectorstore=vectorstore,
        config=config.retriever,
        bm25_index=bm25_index,
        query_rewriter=query_rewriter_fn,
        reranker=reranker,
        rrf_k=config.retriever.rrf_k,
        merge_chunks_per_file=config.retriever.merge_chunks_per_file,
        rerank_candidates=config.reranker.candidates,
    )
    chain = KompasRAGChain(
        retriever=retriever,
        llm=llm,
        config=config,
        docs_base_url=config.docs_base_url,
    )

    logger.info(
        "RAG ready: %s documents, LLM=%s, reranker=%s",
        count,
        config.llm.model_name,
        "on" if reranker else "off",
    )

    return BuiltRAGPipeline(
        config=config,
        chain=chain,
        retriever=retriever,
        llm=llm,
        vectorstore=vectorstore,
        embeddings=embeddings,
        bm25_index=bm25_index,
        reranker=reranker,
    )
