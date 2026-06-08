"""Retrieval helpers used by ablation experiments."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document

from .bm25 import BM25Index
from .rrf import reciprocal_rank_fusion

RetrieveFn = Callable[[str], list[Document]]


def retrieve_bm25(bm25_index: BM25Index, query: str, top_k: int = 20) -> list[Document]:
    """Return BM25-ranked documents."""
    pairs = bm25_index.search(query, top_k=top_k)
    return [doc for doc, _score in pairs]


def retrieve_vector(vectorstore, query: str, top_k: int = 20) -> list[Document]:
    """Return vector-ranked documents."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    return [doc for doc, _score in results]


def retrieve_hybrid(
    bm25_index: BM25Index,
    vectorstore,
    query: str,
    top_k: int = 20,
) -> list[Document]:
    """Return RRF-fused vector and BM25 documents."""
    vec_results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    vec_pairs = [(doc, score) for doc, score in vec_results]
    bm25_pairs = bm25_index.search(query, top_k=top_k)
    fused = reciprocal_rank_fusion([vec_pairs, bm25_pairs])
    return [doc for doc, _score in fused]


def dedup_by_file(docs: list[Document], top_k: int = 5) -> list[Document]:
    """Keep the first document per source file."""
    seen = set()
    result = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "")
        if source_file in seen:
            continue
        seen.add(source_file)
        result.append(doc)
        if len(result) >= top_k:
            break
    return result


def multi_query_retrieve(
    retrieve_fn: RetrieveFn,
    queries: list[str],
    top_k_final: int = 5,
) -> list[Document]:
    """Fuse ranked lists from multiple query rewrites and deduplicate by page."""
    all_lists = []
    for query in queries:
        docs = retrieve_fn(query)
        pairs = [(doc, 1.0 / (rank + 1)) for rank, doc in enumerate(docs)]
        all_lists.append(pairs)

    if len(all_lists) == 1:
        return dedup_by_file([doc for doc, _score in all_lists[0]], top_k_final)

    fused = reciprocal_rank_fusion(all_lists)
    return dedup_by_file([doc for doc, _score in fused], top_k_final)


def make_retrieve_fn(mode: str, bm25_index: BM25Index, vectorstore) -> RetrieveFn:
    """Build a retrieval function for an ablation mode."""
    if mode == "bm25":
        return lambda query: retrieve_bm25(bm25_index, query)
    if mode == "vector":
        return lambda query: retrieve_vector(vectorstore, query)
    if mode == "hybrid":
        return lambda query: retrieve_hybrid(bm25_index, vectorstore, query)
    raise ValueError(f"Unknown retrieval mode: {mode}")
