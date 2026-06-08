"""Retrieval evaluation metrics for annotated ranked results."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

LABELS_STRICT = {"relevant": 1.0, "partial": 0.0, "irrelevant": 0.0}
LABELS_LENIENT = {"relevant": 1.0, "partial": 1.0, "irrelevant": 0.0}
LABELS_GRADED = {"relevant": 2.0, "partial": 1.0, "irrelevant": 0.0}


def load_annotations(path: str | Path) -> tuple[list[str], dict[int, list[tuple[int, str]]]]:
    """Load query-level annotation labels ordered by ranked document index."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    queries = data["queries"]
    annotations = data["annotations"]

    by_query: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for item in annotations.values():
        by_query[item["query_idx"]].append((item["doc_idx"], item["label"]))

    for query_idx in by_query:
        by_query[query_idx].sort(key=lambda item: item[0])

    return queries, dict(by_query)


def precision_at_k(relevances: list[float], k: int) -> float:
    """Compute Precision@k for binary relevance values."""
    if not relevances:
        return 0.0
    return sum(1 for relevance in relevances[:k] if relevance > 0) / k


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute discounted cumulative gain at k."""
    return sum(relevance / math.log2(rank + 2) for rank, relevance in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute normalized discounted cumulative gain at k."""
    dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(relevances: list[float], k: int, total_relevant: float) -> float:
    """Compute Recall@k for binary relevance values."""
    if total_relevant <= 0:
        return 0.0
    return sum(1 for relevance in relevances[:k] if relevance > 0) / total_relevant


def reciprocal_rank(relevances: list[float]) -> float:
    """Compute reciprocal rank of the first relevant result."""
    for rank, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            return 1.0 / rank
    return 0.0


def compute_query_metrics(
    query_idx: int,
    query: str,
    labels: list[str],
) -> dict[str, Any]:
    """Compute all retrieval metrics for one query."""
    rels_strict = [LABELS_STRICT.get(label, 0.0) for label in labels]
    rels_lenient = [LABELS_LENIENT.get(label, 0.0) for label in labels]
    rels_graded = [LABELS_GRADED.get(label, 0.0) for label in labels]

    total_rel_strict = sum(rels_strict)
    total_rel_lenient = sum(rels_lenient)

    return {
        "query_idx": query_idx,
        "query": query,
        "labels": labels,
        "p@3_strict": precision_at_k(rels_strict, 3),
        "p@5_strict": precision_at_k(rels_strict, 5),
        "recall@3_strict": recall_at_k(rels_strict, 3, total_rel_strict),
        "recall@5_strict": recall_at_k(rels_strict, 5, total_rel_strict),
        "mrr_strict": reciprocal_rank(rels_strict),
        "p@3_lenient": precision_at_k(rels_lenient, 3),
        "p@5_lenient": precision_at_k(rels_lenient, 5),
        "recall@3_lenient": recall_at_k(rels_lenient, 3, total_rel_lenient),
        "recall@5_lenient": recall_at_k(rels_lenient, 5, total_rel_lenient),
        "mrr_lenient": reciprocal_rank(rels_lenient),
        "ndcg@3": ndcg_at_k(rels_graded, 3),
        "ndcg@5": ndcg_at_k(rels_graded, 5),
    }


def compute_all_metrics(annotations_path: str | Path) -> list[dict[str, Any]]:
    """Compute retrieval metrics from an annotation JSON file."""
    queries, by_query = load_annotations(annotations_path)
    results_per_query = []

    for query_idx in sorted(by_query):
        docs = by_query[query_idx]
        query = queries[query_idx] if query_idx < len(queries) else f"q{query_idx}"
        labels = [label for _, label in docs]
        results_per_query.append(compute_query_metrics(query_idx, query, labels))

    return results_per_query


def compute_ablation_metrics(labels_list: list[list[str]]) -> dict[str, float]:
    """Compute compact metrics used by retrieval ablation studies."""
    if not labels_list:
        return {"P@3": 0.0, "P@5": 0.0, "NDCG@5": 0.0, "MRR": 0.0}

    p3_values: list[float] = []
    p5_values: list[float] = []
    ndcg5_values: list[float] = []
    mrr_values: list[float] = []

    for labels in labels_list:
        rels_strict = [LABELS_STRICT.get(label, 0.0) for label in labels]
        rels_graded = [LABELS_GRADED.get(label, 0.0) for label in labels]
        p3_values.append(precision_at_k(rels_strict, 3))
        p5_values.append(precision_at_k(rels_strict, 5))
        ndcg5_values.append(ndcg_at_k(rels_graded, 5))
        mrr_values.append(reciprocal_rank(rels_strict))

    n = len(labels_list)
    return {
        "P@3": sum(p3_values) / n,
        "P@5": sum(p5_values) / n,
        "NDCG@5": sum(ndcg5_values) / n,
        "MRR": sum(mrr_values) / n,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Average query-level retrieval metrics."""
    if not results:
        return {}

    metric_names = [
        "p@3_strict",
        "p@5_strict",
        "p@3_lenient",
        "p@5_lenient",
        "recall@5_strict",
        "recall@5_lenient",
        "mrr_strict",
        "mrr_lenient",
        "ndcg@3",
        "ndcg@5",
    ]
    return {
        metric: round(sum(row[metric] for row in results) / len(results), 4)
        for metric in metric_names
    }


def save_metrics_report(
    results: list[dict[str, Any]],
    summary: dict[str, float],
    output_path: str | Path,
) -> None:
    """Save retrieval metrics report as JSON."""
    output = {
        "summary": summary,
        "per_query": results,
        "n_queries": len(results),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
