"""Embedding model adapters."""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def make_embedding_text(text: str, page_title: str = "", breadcrumbs: str = "") -> str:
    """Build the text sent to the embedding model."""
    parts = []
    if breadcrumbs:
        parts.append(breadcrumbs)
    if page_title:
        parts.append(page_title)
    parts.append(text)
    return "\n".join(parts)


class E5InstructEmbeddings:
    """Adapter for intfloat/multilingual-e5-large-instruct."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cuda",
        batch_size: int = 64,
        task: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.task = task
        self.batch_size = batch_size
        dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            "E5-instruct embeddings: model=%s, dim=%s, device=%s",
            model_name,
            dim,
            device,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        formatted = f"Instruct: {self.task}\nQuery: {text}"
        embedding = self.model.encode([formatted], normalize_embeddings=True)
        return embedding[0].tolist()


def create_embeddings(config):
    """Create an embedding adapter from config."""
    model_name = config.model_name

    if "e5" in model_name.lower() and "instruct" in model_name.lower():
        return E5InstructEmbeddings(
            model_name=model_name,
            device=config.device,
            batch_size=config.batch_size,
        )

    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": config.device,
            "torch_dtype": torch.float16,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": config.batch_size,
        },
    )
    dim = len(embeddings.embed_query("test"))
    logger.info("Embeddings: model=%s, dim=%s, device=%s", model_name, dim, config.device)
    return embeddings
