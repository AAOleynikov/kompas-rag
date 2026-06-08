"""Hybrid retriever: vector search, BM25, RRF, page deduplication, rerank."""

import logging
from collections.abc import Callable
from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from ..config import RetrieverConfig
from .bm25 import BM25Index
from .rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

QueryRewriterFn = Optional[Callable[[str], list[str]]]


class KompasRetriever:
    """Hybrid retriever for KOMPAS-3D documentation chunks."""

    def __init__(
        self,
        vectorstore: VectorStore,
        config: Optional[RetrieverConfig] = None,
        bm25_index: Optional[BM25Index] = None,
        query_rewriter: QueryRewriterFn = None,
        reranker=None,
        rrf_k: int = 60,
        merge_chunks_per_file: int = 2,
        rerank_candidates: int = 15,
    ):
        self.vectorstore = vectorstore
        self.config = config or RetrieverConfig()
        self.bm25_index = bm25_index
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.merge_chunks_per_file = merge_chunks_per_file
        self.rerank_candidates = rerank_candidates

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[dict] = None,
    ) -> list[Document]:
        """Run rewrite -> vector/BM25 -> RRF -> file-level dedup -> rerank."""
        top_k_final = top_k or self.config.top_k_final
        fetch_k = self.config.top_k * 2

        queries = self.query_rewriter(query) if self.query_rewriter else [query]
        all_result_lists: list[list[tuple[Document, float]]] = []

        for rewritten_query in queries:
            vector_results = self.vectorstore.similarity_search_with_relevance_scores(
                query=rewritten_query,
                k=fetch_k,
                **({"filter": filter_dict} if filter_dict else {}),
            )
            all_result_lists.append([(doc, score) for doc, score in vector_results])

            if self.bm25_index:
                bm25_pairs = self.bm25_index.search(rewritten_query, top_k=fetch_k)
                if bm25_pairs:
                    all_result_lists.append(bm25_pairs)

        fused = reciprocal_rank_fusion(all_result_lists, k=self.rrf_k)
        logger.info(
            "Retrieval: queries=%s, lists=%s, fused=%s",
            queries,
            len(all_result_lists),
            len(fused),
        )

        ordered_files, docs_by_file = self._group_by_source_file(fused)
        final = self._take_unique_files(ordered_files, docs_by_file, top_k_final)

        logger.info(
            "  unique_files=%s, candidates=%s",
            len(ordered_files),
            len(final),
        )

        if self.reranker is not None and final:
            candidates = self._take_unique_files(
                ordered_files,
                docs_by_file,
                self.rerank_candidates,
            )
            final = self.reranker.rerank(query, candidates, top_k=top_k_final)
            logger.info("  rerank: %s -> %s", len(candidates), len(final))

        return final

    def _group_by_source_file(
        self,
        fused: list[tuple[Document, float]],
    ) -> tuple[list[str], dict[str, list[Document]]]:
        docs_by_file: dict[str, list[Document]] = {}
        ordered_files: list[str] = []

        for doc, score in fused:
            source_file = doc.metadata.get("source_file", "")
            doc.metadata["fusion_score"] = round(score, 4)

            if source_file not in docs_by_file:
                docs_by_file[source_file] = []
                ordered_files.append(source_file)
            docs_by_file[source_file].append(doc)

        return ordered_files, docs_by_file

    def _take_unique_files(
        self,
        ordered_files: list[str],
        docs_by_file: dict[str, list[Document]],
        limit: int,
    ) -> list[Document]:
        result: list[Document] = []
        for source_file in ordered_files:
            if len(result) >= limit:
                break

            docs_from_file = docs_by_file[source_file]
            if len(docs_from_file) == 1:
                result.append(docs_from_file[0])
            else:
                result.append(
                    self._merge_page_chunks(
                        docs_from_file,
                        max_chunks=self.merge_chunks_per_file,
                    )
                )
        return result

    def _merge_page_chunks(
        self,
        docs: list[Document],
        max_chunks: int = 2,
    ) -> Document:
        docs_sorted = sorted(
            docs[:max_chunks],
            key=lambda doc: doc.metadata.get("chunk_index", 0),
        )
        merged_text = "\n\n".join(doc.page_content for doc in docs_sorted)
        best_meta = docs[0].metadata.copy()
        best_meta["merged_chunks"] = len(docs_sorted)
        return Document(page_content=merged_text, metadata=best_meta)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Return vector-only scores for debugging."""
        k = top_k or self.config.top_k
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )
        return [
            (doc, score)
            for doc, score in results
            if score >= self.config.score_threshold
        ]
