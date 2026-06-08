"""Backward-compatible retrieval facade.

The implementation lives in ``src.rag.retrieval``. This module keeps existing
imports such as ``from .retriever import KompasRetriever`` working.
"""

from .retrieval.bm25 import (
    BM25Index,
    build_bm25_index,
    build_bm25_index_from_documents,
    load_bm25_index,
    tokenize_for_bm25,
)
from .retrieval.hybrid import KompasRetriever, QueryRewriterFn
from .retrieval.rrf import reciprocal_rank_fusion

_tokenize = tokenize_for_bm25

__all__ = [
    "BM25Index",
    "KompasRetriever",
    "QueryRewriterFn",
    "_tokenize",
    "build_bm25_index",
    "build_bm25_index_from_documents",
    "load_bm25_index",
    "reciprocal_rank_fusion",
]
