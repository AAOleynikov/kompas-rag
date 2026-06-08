"""CLI wrapper for retrieval metrics computed by ``rag.evaluation.retrieval``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src" / "rag").exists()
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.evaluation.retrieval import (  # noqa: E402
    compute_all_metrics,
    save_metrics_report,
    summarize_results,
)

COMPACT_METRIC_KEYS = ("P@3", "P@5", "NDCG@5", "MRR")


def print_report(results: list[dict], summary: dict[str, float]) -> None:
    if not results:
        print("No annotations found.")
        return

    print("=" * 70)
    print(f"RAG RETRIEVAL METRICS - {len(results)} queries")
    print("=" * 70)
    print(f"Precision@3 strict : {summary['p@3_strict']:.3f}")
    print(f"Precision@5 strict : {summary['p@5_strict']:.3f}")
    print(f"Precision@3 lenient: {summary['p@3_lenient']:.3f}")
    print(f"Precision@5 lenient: {summary['p@5_lenient']:.3f}")
    print(f"Recall@5 strict    : {summary['recall@5_strict']:.3f}")
    print(f"Recall@5 lenient   : {summary['recall@5_lenient']:.3f}")
    print(f"MRR strict         : {summary['mrr_strict']:.3f}")
    print(f"MRR lenient        : {summary['mrr_lenient']:.3f}")
    print(f"NDCG@3 graded      : {summary['ndcg@3']:.3f}")
    print(f"NDCG@5 graded      : {summary['ndcg@5']:.3f}")

    weak_queries = [row for row in results if row["p@5_strict"] < 0.2]
    if weak_queries:
        print()
        print(f"Weak queries (P@5 strict < 0.2): {len(weak_queries)}")
        for row in weak_queries:
            print(f"  q{row['query_idx']}: {row['query'][:80]}")


def load_ablation_results(path: str | Path) -> dict[str, dict[str, float]]:
    """Load compact ablation metrics from either raw or documented JSON files."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: expected object with metrics")

    normalized: dict[str, dict[str, float]] = {}
    for name, values in metrics.items():
        if not isinstance(values, dict):
            raise ValueError(f"{path}: {name!r} is not a metric object")

        missing = [key for key in COMPACT_METRIC_KEYS if key not in values]
        if missing:
            raise ValueError(f"{path}: {name!r} missing metrics: {', '.join(missing)}")

        normalized[name] = {key: float(values[key]) for key in COMPACT_METRIC_KEYS}
        if "avg_time" in values:
            normalized[name]["avg_time"] = float(values["avg_time"])

    return normalized


def print_ablation_results(paths: list[str]) -> None:
    """Print a compact comparison table for stored ablation result files."""
    for path in paths:
        metrics = load_ablation_results(path)
        print("=" * 100)
        print(f"{Path(path)} - {len(metrics)} configurations")
        print("=" * 100)

        has_time = any("avg_time" in values for values in metrics.values())
        header = f"{'Configuration':<62} {'P@3':>7} {'P@5':>7} {'NDCG@5':>8} {'MRR':>7}"
        if has_time:
            header += f" {'Time':>7}"
        print(header)
        print("-" * len(header))

        for name, values in sorted(metrics.items(), key=lambda item: item[1]["MRR"], reverse=True):
            line = (
                f"{name:<62} {values['P@3']:>7.3f} {values['P@5']:>7.3f} "
                f"{values['NDCG@5']:>8.3f} {values['MRR']:>7.3f}"
            )
            if has_time:
                avg_time = values.get("avg_time")
                line += f" {avg_time:>6.2f}s" if avg_time is not None else f" {'-':>7}"
            print(line)

        best_mrr = max(metrics.items(), key=lambda item: item[1]["MRR"])
        best_ndcg = max(metrics.items(), key=lambda item: item[1]["NDCG@5"])
        print()
        print(f"Best MRR    : {best_mrr[0]} ({best_mrr[1]['MRR']:.3f})")
        print(f"Best NDCG@5 : {best_ndcg[0]} ({best_ndcg[1]['NDCG@5']:.3f})")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute retrieval metrics from annotations.")
    parser.add_argument("--annotations", default="/storage/annotations.json")
    parser.add_argument("--output", default="/storage/metrics_report.json")
    parser.add_argument(
        "--results",
        nargs="+",
        default=None,
        help=(
            "Print compact ablation result JSON files. Supports both raw "
            "ablation outputs and documented files with a top-level metrics key."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.results:
        print_ablation_results(args.results)
        return

    results = compute_all_metrics(args.annotations)
    summary = summarize_results(results)
    print_report(results, summary)
    save_metrics_report(results, summary, args.output)
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
