"""Evaluation helpers for the RAG runtime."""

from .retrieval import (
    compute_ablation_metrics,
    compute_all_metrics,
    compute_query_metrics,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    save_metrics_report,
    summarize_results,
)

__all__ = [
    "compute_ablation_metrics",
    "compute_all_metrics",
    "compute_query_metrics",
    "dcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "save_metrics_report",
    "summarize_results",
]
