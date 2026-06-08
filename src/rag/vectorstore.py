"""Vector store and chunk ingestion helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .config import VectorStoreConfig
from .embeddings import make_embedding_text

logger = logging.getLogger(__name__)


def load_chunks(chunks_path: str) -> list[dict]:
    """Load JSONL chunks produced by the existing corpus pipeline."""
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %s chunks from %s", len(chunks), chunks_path)
    return chunks


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove exact text duplicates while preserving alternate sources."""
    seen: dict[str, dict] = {}

    for chunk in chunks:
        text = chunk["text"]
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in seen:
            existing = seen[text_hash]
            new_file = chunk["metadata"].get("source_file", "")
            alt_sources = existing["metadata"].get("alt_sources", "")
            existing["metadata"]["alt_sources"] = (
                f"{alt_sources};{new_file}" if alt_sources else new_file
            )
        else:
            seen[text_hash] = chunk

    result = list(seen.values())
    removed = len(chunks) - len(result)
    if removed:
        logger.info("Deduplication removed %s duplicates, kept %s", removed, len(result))
    return result


STOP_PATTERNS = [
    "Информация в Интернете",
]


def filter_stop_chunks(chunks: list[dict]) -> list[dict]:
    """Remove known low-value chunks from retrieval indexing."""
    result: list[dict] = []
    removed = 0

    for chunk in chunks:
        text = chunk["text"]
        if any(pattern in text for pattern in STOP_PATTERNS):
            removed += 1
            continue
        result.append(chunk)

    if removed:
        logger.info("Stop filter removed %s chunks", removed)
    return result


def chunks_to_documents(chunks: list[dict]) -> tuple[list[Document], list[str], list[str]]:
    """Convert JSONL chunks to LangChain documents, embedding texts and IDs."""
    documents: list[Document] = []
    embedding_texts: list[str] = []
    ids: list[str] = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        chunk_id = chunk.get("id", "")
        if not chunk_id:
            source_file = meta.get("source_file", "unknown")
            chunk_index = meta.get("chunk_index", 0)
            raw_id = f"{source_file}::{chunk_index}::{chunk['text'][:200]}"
            chunk_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        embedding_text = make_embedding_text(
            text=chunk["text"],
            page_title=meta.get("page_title", ""),
            breadcrumbs=meta.get("breadcrumbs", ""),
        )

        document = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk_id,
                "source_file": meta.get("source_file", ""),
                "page_title": meta.get("page_title", ""),
                "breadcrumbs": meta.get("breadcrumbs", ""),
                "section_title": meta.get("section_title", ""),
                "section_path": meta.get("section_path", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "char_count": meta.get("char_count", 0),
                "has_table": meta.get("has_table", False),
                "has_procedure": meta.get("has_procedure", False),
                "block_types": ",".join(meta.get("block_types", [])),
                "alt_sources": meta.get("alt_sources", ""),
            },
        )
        documents.append(document)
        embedding_texts.append(embedding_text)
        ids.append(chunk_id)

    logger.info("Prepared %s documents", len(documents))
    return documents, embedding_texts, ids


def create_vectorstore(
    embeddings: Embeddings,
    config: Optional[VectorStoreConfig] = None,
) -> VectorStore:
    """Create or load the configured vector store."""
    config = config or VectorStoreConfig()

    if config.backend == "chroma":
        return _create_chroma(embeddings, config)
    if config.backend == "qdrant":
        return _create_qdrant(embeddings, config)
    raise ValueError(f"Unknown vector store backend: {config.backend}")


def get_vectorstore_count(vectorstore: VectorStore) -> int:
    """Return document count for stores exposing a Chroma-like collection."""
    return _get_collection(vectorstore).count()


def iter_vectorstore_documents(
    vectorstore: VectorStore,
    batch_size: int = 5000,
) -> Iterator[Document]:
    """Yield all stored documents from a Chroma-backed vector store."""
    collection = _get_collection(vectorstore)
    count = collection.count()

    for offset in range(0, count, batch_size):
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc_text, meta in zip(result["documents"], result["metadatas"]):
            yield Document(page_content=doc_text, metadata=meta or {})


def ingest_to_vectorstore(
    vectorstore: VectorStore,
    documents: list[Document],
    embedding_texts: list[str],
    ids: list[str],
    batch_size: int = 100,
) -> None:
    """Add documents to the vector store in batches with precomputed IDs."""
    total = len(documents)
    logger.info("Uploading %s documents to vector store...", total)

    embedding_function = _get_embedding_function(vectorstore)
    collection = _get_collection(vectorstore)

    for offset in range(0, total, batch_size):
        batch_docs = documents[offset : offset + batch_size]
        batch_texts = embedding_texts[offset : offset + batch_size]
        batch_ids = ids[offset : offset + batch_size]

        embeddings = embedding_function.embed_documents(batch_texts)
        collection.add(
            documents=[doc.page_content for doc in batch_docs],
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in batch_docs],
            ids=batch_ids,
        )

        loaded = min(offset + batch_size, total)
        if loaded % 500 == 0 or loaded == total:
            logger.info("  Uploaded %s/%s", loaded, total)

    logger.info("Upload complete. Vector store documents: %s", collection.count())


def _create_chroma(
    embeddings: Embeddings,
    config: VectorStoreConfig,
) -> VectorStore:
    from langchain_chroma import Chroma

    persist_dir = Path(config.chroma_persist_dir)
    store = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    logger.info(
        "Chroma: collection=%s, documents=%s, path=%s",
        config.collection_name,
        get_vectorstore_count(store),
        persist_dir,
    )
    return store


def _create_qdrant(
    embeddings: Embeddings,
    config: VectorStoreConfig,
) -> VectorStore:
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "For Qdrant install: pip install langchain-qdrant qdrant-client"
        ) from exc

    client = QdrantClient(url=config.qdrant_url)
    store = QdrantVectorStore(
        client=client,
        collection_name=config.qdrant_collection,
        embedding=embeddings,
    )
    logger.info("Qdrant: collection=%s", config.qdrant_collection)
    return store


def _get_collection(vectorstore: VectorStore):
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise TypeError(
            "This operation requires a vector store exposing a Chroma-like "
            "'_collection'."
        )
    return collection


def _get_embedding_function(vectorstore: VectorStore):
    embedding_function = getattr(vectorstore, "_embedding_function", None)
    if embedding_function is None:
        raise TypeError(
            "This operation requires a vector store exposing '_embedding_function'."
        )
    return embedding_function
