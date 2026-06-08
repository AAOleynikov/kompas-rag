# KOMPAS-3D v24 RAG

RAG-система информационной поддержки пользователя КОМПАС-3D v24 по официальной HTML-документации.

Проект разделен на три зоны:

- `src/corpus` - подготовка корпуса из HTML-документации, очистка и chunking.
- `src/rag` - runtime RAG: embeddings, vectorstore, BM25, hybrid retrieval, rerank, LLM generation, ссылки.
- `experiments/retrieval_ablation` - экспериментальные retrieval ablation и расчет IR-метрик.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e .
```

Для локальных секретов и путей:

```powershell
Copy-Item .env.example .env
```

## Основные переменные

Runtime-конфиг загружается из `src/rag/config.py`, optional YAML через `--config` и env overrides:

- `KOMPAS_RAG_PERSIST_DIR` - путь к ChromaDB.
- `KOMPAS_RAG_COLLECTION` - имя коллекции Chroma.
- `KOMPAS_RAG_BM25_INDEX_PATH` - путь к сохраненному BM25-индексу.
- `KOMPAS_RAG_EMBEDDING_MODEL` - embedding-модель.
- `KOMPAS_RAG_EMBEDDING_DEVICE` - устройство embedding-модели.
- `KOMPAS_RAG_LLM_URL` - OpenAI-compatible endpoint.
- `KOMPAS_RAG_LLM_MODEL` - модель генерации.
- `KOMPAS_RAG_LLM_API_KEY` - API key.
- `KOMPAS_RAG_TOP_K`, `KOMPAS_RAG_TOP_K_FINAL` - retrieval limits.
- `KOMPAS_RAG_RERANK`, `KOMPAS_RAG_REWRITE` - включение rerank/rewrite.

Базовый YAML-конфиг проекта: `configs/rag.yaml`.

## Подготовка корпуса

HTML-парсинг и preprocessing сейчас оставлены как отдельный слой. Каноническая точка входа:

```powershell
.venv\Scripts\python.exe -m corpus.cli `
  data\raw\prepare_doc_v24\kompas_v2_doc_welcome_page\help.ascon.ru\KOMPAS\24\ru-RU `
  --output data\processed\chunks.jsonl `
  --tree data\raw\prepare_doc_v24\kompas_v2_doc_welcome_page\help.ascon.ru\KOMPAS\24\ru-RU\js\hmcontent.js `
  --no-images `
  --stats
```

Совместимый wrapper:

```powershell
.venv\Scripts\python.exe src\run_chunker.py path\to\html_docs -o data\processed\chunks.jsonl
```

## Индексация

Построение/обновление векторного индекса ChromaDB и полнотекстового индекса BM25:

```powershell
.venv\Scripts\python.exe -m rag.ingest `
  --config configs\rag.yaml `
  data\processed\chunks.jsonl `
  --persist-dir data\indexes\chroma_db `
  --bm25-index-path data\indexes\bm25_index.pkl
```

Оба индекса строятся из одного подготовленного `chunks.jsonl`: документы
батчами загружаются в ChromaDB, а BM25 строится параллельно из тех же
подготовленных `Document` и сохраняется в `data/indexes/bm25_index.pkl`.
При запуске RAG BM25 не пересобирается из Chroma, а загружается из сохраненной
копии. Если число документов в BM25 не совпадает с Chroma, запуск остановится и
потребует заново выполнить `rag.ingest`.

Если ChromaDB уже построен, а нужно создать только сохраненный BM25 без
перезагрузки векторного индекса:

```powershell
.venv\Scripts\python.exe -m rag.ingest `
  --config configs\rag.yaml `
  data\processed\chunks.jsonl `
  --bm25-index-path data\indexes\bm25_index.pkl `
  --bm25-only
```

## Запуск RAG

CLI-запрос:

```powershell
.venv\Scripts\python.exe -m rag.app `
  --config configs\rag.yaml `
  --persist-dir data\indexes\chroma_db `
  --sources `
  -q "Как построить массив элементов по сетке?"
```

Web UI:

```powershell
.venv\Scripts\python.exe -m rag.web_ui `
  --config configs\rag.yaml `
  --persist-dir data\indexes\chroma_db `
  --port 7860
```

### Docker

Локальный запуск Gradio через Docker Compose:

```bash
cp .env.example .env
# заполнить KOMPAS_RAG_LLM_URL, KOMPAS_RAG_LLM_MODEL, KOMPAS_RAG_LLM_API_KEY
docker compose up --build gradio
```

После старта UI доступен на `http://localhost:7860`.

Compose монтирует локальный каталог `data` внутрь контейнера read-only, поэтому индекс должен уже лежать в `data/indexes/chroma_db`. По умолчанию в контейнере используется CPU:

```bash
KOMPAS_RAG_EMBEDDING_DEVICE=cpu docker compose up --build gradio
```

Сборка образа без Compose:

```bash
docker build -t kompas-rag:local .
docker run --rm -p 7860:7860 --env-file .env -v "$PWD/data:/app/data:ro" kompas-rag:local
```

Annotator UI:

```powershell
.venv\Scripts\python.exe -m rag.annotator `
  --config configs\rag.yaml `
  --persist-dir data\indexes\chroma_db `
  --port 7861
```

## Оценка

Retrieval metrics по размеченному JSON:

```powershell
.venv\Scripts\python.exe experiments\retrieval_ablation\scripts\summarize_metrics.py `
  --annotations data\eval\annotations_llm_v2.json `
  --output data\eval\metrics_report.json
```

Generation evaluation:

```powershell
.venv\Scripts\python.exe -m rag.eval_generation `
  --persist-dir data\indexes\chroma_db `
  --api-key $env:JUDGE_API_KEY `
  --api-base $env:JUDGE_API_BASE `
  --model $env:JUDGE_MODEL `
  --limit 29
```

## Retrieval Ablation

E5/current RAG index:

```powershell
.venv\Scripts\python.exe experiments\retrieval_ablation\scripts\run_e5_ablation.py `
  --config configs\rag.yaml `
  --persist-dir data\indexes\chroma_db `
  --annotations data\eval\annotations_llm_v2.json `
  --output data\eval\annotations_llm_v2.json `
  --results-output data\eval\ablation_v2_results.json `
  --api-key $env:JUDGE_API_KEY `
  --api-base $env:JUDGE_API_BASE `
  --model $env:JUDGE_MODEL
```

Giga-Embeddings index:

```powershell
.venv\Scripts\python.exe experiments\retrieval_ablation\scripts\run_giga_ablation.py `
  --config configs\rag.yaml `
  --persist-dir data\indexes\chroma_db_giga `
  --collection kompas_docs `
  --annotations data\eval\annotations_llm_v2.json `
  --output data\eval\annotations_llm_v2.json `
  --results-output data\eval\ablation_giga_results.json `
  --api-key $env:JUDGE_API_KEY `
  --api-base $env:JUDGE_API_BASE `
  --model $env:JUDGE_MODEL
```

Просмотр сохранённых research-замеров по стратегиям поиска:

```powershell
.venv\Scripts\python.exe experiments\retrieval_ablation\scripts\summarize_metrics.py `
  --results experiments\retrieval_ablation\results\hybrid_search_metrics.json `
            experiments\retrieval_ablation\results\vector_search_metrics.json
```

## Структура

```text
src/
  corpus/                 HTML -> cleaned chunks
  rag/
    retrieval/            BM25, RRF, hybrid retrieval
    generation/           prompts, LLM client, RAG chain, references
    evaluation/           retrieval and generation metrics/helpers
    config.py             runtime config and env overrides
    vectorstore.py        Chroma/Qdrant creation helpers
experiments/
  retrieval_ablation/
    scripts/              retrieval ablation scripts and metrics CLI
    results/              saved research metrics for search strategy selection
data/
  raw/                    source data, not committed
  processed/              generated corpus/chunks, not committed
  indexes/                ChromaDB and indexes, not committed
  eval/                   reports/annotations, curated manually
```

## Проверка синтаксиса

```powershell
.venv\Scripts\python.exe -m compileall src experiments\retrieval_ablation
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

## CI/CD

- `.github/workflows/ci.yml` - установка зависимостей, `compileall`, smoke-test загрузки конфига и Docker build.
- `.github/workflows/docker-publish.yml` - публикация Docker image в GHCR по тегам `v*` или ручному `workflow_dispatch`.
