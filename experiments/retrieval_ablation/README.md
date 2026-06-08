# Retrieval Ablation Experiments

В этом каталоге находятся скрипты retrieval-оценки и сохраненные research-метрики
для сравнения стратегий поиска по инженерной документации КОМПАС-3D.

## Структура

- `scripts/` - исполняемые скрипты экспериментов и отчетов.
- `results/` - сохраненные JSON-метрики research-запусков.

## Скрипты

- `scripts/run_e5_ablation.py` - основной ablation-эксперимент для текущего E5-индекса.
- `scripts/run_giga_ablation.py` - вариант ablation-эксперимента для Giga-Embeddings.
- `scripts/summarize_metrics.py` - расчет retrieval-метрик по аннотациям и компактные сводки по сохраненным JSON-результатам.

## Сохраненные метрики

- `results/hybrid_search_metrics.json` - замеры BM25 и hybrid search.
- `results/vector_search_metrics.json` - замеры vector-only поиска по разным embedding-моделям.

Печать сохраненных research-метрик:

```bash
python experiments/retrieval_ablation/scripts/summarize_metrics.py \
  --results experiments/retrieval_ablation/results/hybrid_search_metrics.json \
            experiments/retrieval_ablation/results/vector_search_metrics.json
```
