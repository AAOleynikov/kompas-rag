"""Runtime configuration for the KOMPAS-3D RAG system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""

    model_name: str = "intfloat/multilingual-e5-large-instruct"
    device: str = "cuda:2"
    query_instruction: str = (
        "Instruct: Дан вопрос, необходимо найти абзац текста с ответом\n"
        "Query: "
    )
    batch_size: int = 64
    max_length: int = 512


@dataclass
class VectorStoreConfig:
    """Vector store settings."""

    backend: str = "chroma"
    chroma_persist_dir: str = "/storage/chroma_db"
    collection_name: str = "kompas_docs"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kompas_docs"


@dataclass
class RetrieverConfig:
    """Hybrid retrieval settings."""

    top_k: int = 15
    top_k_final: int = 5
    bm25_index_path: str = "data/indexes/bm25_index.pkl"
    score_threshold: float = 0.3
    filter_empty: bool = True
    rrf_k: int = 60
    merge_chunks_per_file: int = 2


@dataclass
class RerankerConfig:
    """Cross-encoder reranker settings."""

    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    candidates: int = 20
    top_k: int = 5
    device: str | None = None
    max_length: int = 512
    batch_size: int = 16


@dataclass
class QueryRewriteConfig:
    """LLM query rewriting settings."""

    enabled: bool = True
    max_queries: int = 3


@dataclass
class LLMConfig:
    """OpenAI-compatible LLM endpoint settings."""

    base_url: str = "http://localhost:8000/v1"
    model_name: str = "llm"
    api_key: str = "not-needed"
    temperature: float = 0.3
    max_tokens: int = 16000
    top_p: float | None = None
    disable_thinking: bool = True


@dataclass
class RAGConfig:
    """Top-level runtime RAG configuration."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    query_rewrite: QueryRewriteConfig = field(default_factory=QueryRewriteConfig)
    docs_base_url: str = "https://help.ascon.ru/KOMPAS/24/ru-RU"
    chunks_path: str = "data/processed/chunks.jsonl"
    debug: bool = False


T = TypeVar("T")


def load_config(path: str | Path | None = None) -> RAGConfig:
    """Load config from defaults, optional YAML and environment overrides."""
    load_dotenv()
    config = RAGConfig()

    if path:
        config_path = Path(path)
        if config_path.exists():
            import yaml

            with config_path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
            _update_dataclass(config, data)

    apply_env_overrides(config)
    return config


def apply_env_overrides(config: RAGConfig) -> RAGConfig:
    """Apply KOMPAS_RAG_* environment variables to a config object."""
    env_map = {
        "KOMPAS_RAG_PERSIST_DIR": ("vectorstore", "chroma_persist_dir", str),
        "KOMPAS_RAG_COLLECTION": ("vectorstore", "collection_name", str),
        "KOMPAS_RAG_BACKEND": ("vectorstore", "backend", str),
        "KOMPAS_RAG_EMBEDDING_MODEL": ("embedding", "model_name", str),
        "KOMPAS_RAG_EMBEDDING_DEVICE": ("embedding", "device", str),
        "KOMPAS_RAG_LLM_URL": ("llm", "base_url", str),
        "KOMPAS_RAG_LLM_MODEL": ("llm", "model_name", str),
        "KOMPAS_RAG_LLM_API_KEY": ("llm", "api_key", str),
        "KOMPAS_RAG_DOCS_URL": (None, "docs_base_url", str),
        "KOMPAS_RAG_TOP_K": ("retriever", "top_k", int),
        "KOMPAS_RAG_TOP_K_FINAL": ("retriever", "top_k_final", int),
        "KOMPAS_RAG_BM25_INDEX_PATH": ("retriever", "bm25_index_path", str),
        "KOMPAS_RAG_RERANK": ("reranker", "enabled", _parse_bool),
        "KOMPAS_RAG_REWRITE": ("query_rewrite", "enabled", _parse_bool),
    }

    for env_name, (section, attr, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        target = getattr(config, section) if section else config
        setattr(target, attr, caster(raw))

    return config


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass instance, got {type(instance)!r}")

    field_names = {item.name for item in fields(instance)}
    for key, value in values.items():
        if key not in field_names:
            continue

        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}
