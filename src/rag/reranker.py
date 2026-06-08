"""Cross-encoder reranker."""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import torch
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker based on a sequence-classification cross-encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading reranker: %s (device=%s)", model_name, self.device)
        started_at = time.time()

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            self.model.half()

        logger.info("Reranker loaded in %.1f s", time.time() - started_at)

    @torch.no_grad()
    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset : offset + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend(1 / (1 + math.exp(-score)) for score in logits)
        return scores

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """Rerank documents by cross-encoder relevance to the query."""
        if not documents:
            return []

        started_at = time.time()
        pairs = [
            [query, document.page_content[: self.max_length * 4]]
            for document in documents
        ]
        scores = self._score_pairs(pairs)

        scored_docs = []
        for document, score in zip(documents, scores):
            doc_copy = Document(
                page_content=document.page_content,
                metadata={**document.metadata, "reranker_score": round(score, 4)},
            )
            scored_docs.append((score, doc_copy))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        result = [document for _, document in scored_docs[:top_k]]

        logger.debug(
            "Rerank: %s -> %s docs in %.2f s",
            len(documents),
            len(result),
            time.time() - started_at,
        )
        return result
