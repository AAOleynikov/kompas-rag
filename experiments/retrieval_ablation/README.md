# Retrieval Ablation Experiments

This directory contains retrieval evaluation scripts and saved research metrics
for comparing search strategies.

## Layout

- `scripts/` - executable experiment and reporting scripts.
- `results/` - saved JSON metrics from research runs.

## Scripts

- `scripts/run_e5_ablation.py` - main 12-mode retrieval ablation for the current E5 index.
- `scripts/run_giga_ablation.py` - Giga-Embeddings variant of the ablation.
- `scripts/summarize_metrics.py` - retrieval metrics over annotation files and compact summaries for saved result JSON files.

## Saved Metrics

- `results/hybrid_search_metrics.json` - BM25 and hybrid search measurements.
- `results/vector_search_metrics.json` - vector-only measurements across embedding models.

Print saved research metrics:

```bash
python experiments/retrieval_ablation/scripts/summarize_metrics.py \
  --results experiments/retrieval_ablation/results/hybrid_search_metrics.json \
            experiments/retrieval_ablation/results/vector_search_metrics.json
```
