"""Reciprocal Rank Fusion utilities."""

from langchain_core.documents import Document


def document_key(doc: Document) -> str:
    """Return a stable key for deduplicating retrieval results."""
    key = doc.metadata.get("chunk_id", "")
    if key:
        return str(key)
    return f"{doc.metadata.get('source_file', '')}_{doc.metadata.get('chunk_index', 0)}"


def reciprocal_rank_fusion(
    results_list: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse ranked result lists with RRF.

    RRF score = sum(1 / (k + rank_i)).
    """
    scores: dict[str, tuple[Document, float]] = {}

    for results in results_list:
        for rank, (doc, _original_score) in enumerate(results):
            key = document_key(doc)
            if key not in scores:
                scores[key] = (doc, 0.0)

            doc_obj, current_score = scores[key]
            scores[key] = (doc_obj, current_score + 1.0 / (k + rank + 1))

    return sorted(scores.values(), key=lambda item: item[1], reverse=True)
