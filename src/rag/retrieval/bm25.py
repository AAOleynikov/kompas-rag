"""BM25 index helpers for lexical retrieval over RAG chunks."""

import logging
import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from rank_bm25 import BM25Okapi

from ..vectorstore import get_vectorstore_count, iter_vectorstore_documents

logger = logging.getLogger(__name__)

BM25_CACHE_VERSION = 1


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for the lightweight Russian/Latin BM25 baseline."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [token for token in text.split() if len(token) > 2]


class BM25Index:
    """In-memory BM25 index over RAG chunks."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        corpus = [tokenize_for_bm25(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info("BM25 index: %s documents", len(documents))

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            (self.documents[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    def save(self, path: str | Path) -> None:
        """Persist BM25 documents and the rank-bm25 model to a pickle file."""
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": BM25_CACHE_VERSION,
            "doc_count": self.doc_count,
            "documents": self.documents,
            "bm25": self.bm25,
        }
        with cache_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index: %s documents, path=%s", self.doc_count, cache_path)


def load_bm25_index(path: str | Path, expected_doc_count: int | None = None) -> BM25Index:
    """Load a persisted BM25 index and validate its basic compatibility."""
    cache_path = Path(path)
    with cache_path.open("rb") as stream:
        payload: dict[str, Any] = pickle.load(stream)

    if payload.get("version") != BM25_CACHE_VERSION:
        raise ValueError(
            f"Unsupported BM25 cache version: {payload.get('version')!r}"
        )

    doc_count = int(payload.get("doc_count", -1))
    if expected_doc_count is not None and doc_count != expected_doc_count:
        raise ValueError(
            f"BM25 cache document count mismatch: cache={doc_count}, "
            f"vectorstore={expected_doc_count}"
        )

    index = BM25Index.__new__(BM25Index)
    index.documents = payload["documents"]
    index.bm25 = payload["bm25"]
    logger.info("Loaded BM25 index: %s documents, path=%s", index.doc_count, cache_path)
    return index


def build_bm25_index_from_documents(
    documents: list[Document],
    persist_path: str | Path | None = None,
) -> BM25Index:
    """Build BM25 directly from prepared chunk documents and optionally save it."""
    index = BM25Index(documents)

    if persist_path:
        try:
            index.save(persist_path)
        except Exception as exc:
            logger.warning("Could not save BM25 index to %s: %s", persist_path, exc)

    return index


def build_bm25_index(
    vectorstore: VectorStore,
    persist_path: str | Path | None = None,
) -> BM25Index:
    """Build BM25 from all documents in a Chroma/Qdrant-compatible vector store.

    This compatibility helper is used by research scripts. The runtime pipeline
    loads a saved BM25 index produced during ``rag.ingest`` instead of rebuilding
    it from Chroma.
    """
    count = get_vectorstore_count(vectorstore)

    if persist_path:
        cache_path = Path(persist_path)
        if cache_path.exists():
            try:
                return load_bm25_index(cache_path, expected_doc_count=count)
            except Exception as exc:
                logger.warning("Could not load BM25 index from %s: %s", cache_path, exc)

    logger.info("Loading %s documents from vector store for BM25...", count)

    all_docs = list(iter_vectorstore_documents(vectorstore))
    logger.info("Loaded %s documents for BM25", len(all_docs))
    return build_bm25_index_from_documents(all_docs, persist_path=persist_path)
