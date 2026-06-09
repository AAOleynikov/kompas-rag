"""Ablation study v2: 12 конфигураций с E5-large-instruct."""

import json
import logging
import time
import argparse
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src" / "rag").exists()
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

for name in [
    "httpx", "httpcore", "urllib3", "chromadb",
    "sentence_transformers", "FlagEmbedding", "openai",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


# ── Judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """Ты — эксперт-оценщик релевантности для системы поиска по документации САПР КОМПАС-3D.

## Вопрос пользователя
{question}

## Фрагмент документации
Файл: {source_file}

{content}

---

"relevant" — фрагмент ПОЛЕЗЕН для ответа: описывает нужную функцию/команду/операцию, содержит инструкцию или параметры.
"partial" — фрагмент КОСВЕННО связан: упоминает тему, но без конкретной полезной информации.
"irrelevant" — фрагмент НЕ ПОЛЕЗЕН: другая тема.

Ответь СТРОГО в формате JSON (без markdown):
{{"label": "relevant|partial|irrelevant", "reason": "краткое обоснование"}}
"""


def create_judge(api_key, api_base, model):
    import httpx
    from langchain_openai import ChatOpenAI

    http_client = httpx.Client(base_url=api_base, verify=False, timeout=120.0)
    return ChatOpenAI(
        base_url=api_base, api_key=api_key, model=model,
        max_tokens=200, temperature=0, http_client=http_client,
    )


def judge_single(judge, question, doc_content, source_file):
    from langchain_core.messages import HumanMessage

    prompt = JUDGE_PROMPT.format(
        question=question, source_file=source_file, content=doc_content[:3000],
    )
    for attempt in range(5):
        try:
            resp = judge.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"    judge attempt {attempt+1}/5: {e}")
            time.sleep(5 * (attempt + 1))
    return {"label": "irrelevant", "reason": "retries exhausted"}

# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML RAG config.")
    parser.add_argument("--persist-dir", default=None, help="Override Chroma persist directory.")
    parser.add_argument("--bm25-index-path", default=None, help="Override persisted BM25 index path.")
    parser.add_argument("--annotations", default="/storage/annotations_llm_v2.json")
    parser.add_argument("--output", default="/storage/annotations_llm_v2.json")
    parser.add_argument("--results-output", default="/storage/ablation_v2_results.json")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default="tgpt/qwen3-coder-480b-a35b-instruct")
    args = parser.parse_args()

    # ── Загрузка компонентов ──
    from rag.config import load_config
    from rag.embeddings import create_embeddings
    from rag.vectorstore import create_vectorstore, get_vectorstore_count
    from rag.retrieval.bm25 import load_bm25_index
    from rag.generation.llm import create_llm
    from rag.evaluation.retrieval import compute_ablation_metrics
    from rag.query_rewriter import rewrite_query
    from rag.reranker import CrossEncoderReranker
    from rag.retrieval.ablation import make_retrieve_fn, multi_query_retrieve

    config = load_config(args.config)
    if args.persist_dir:
        config.vectorstore.chroma_persist_dir = args.persist_dir
    if args.bm25_index_path:
        config.retriever.bm25_index_path = args.bm25_index_path
    logger.info(f"Embedder: {config.embedding.model_name}")

    embeddings = create_embeddings(config.embedding)
    vectorstore = create_vectorstore(embeddings, config.vectorstore)
    bm25_index = load_bm25_index(
        config.retriever.bm25_index_path,
        expected_doc_count=get_vectorstore_count(vectorstore),
    )
    llm = create_llm(config.llm)
    reranker = CrossEncoderReranker(
        model_name=config.reranker.model_name,
        device=config.reranker.device,
        max_length=config.reranker.max_length,
        batch_size=config.reranker.batch_size,
    )

    # ── Аннотации ──
    with open(args.annotations) as f:
        data = json.load(f)
    queries = data["queries"]
    annotations = data["annotations"]

    gt = defaultdict(dict)
    for a in annotations.values():
        gt[a["query_idx"]][a["source_file"]] = a["label"]

    annotated_qi = sorted(set(a["query_idx"] for a in annotations.values()))
    logger.info(f"Запросов: {len(annotated_qi)}, аннотаций: {len(annotations)}")

    # ── Rewrites ──
    logger.info("Генерируем query rewrites...")
    rewrites = {}
    for qi in annotated_qi:
        try:
            rw = rewrite_query(queries[qi], llm)
        except:
            rw = [queries[qi]]
        rewrites[qi] = rw
        logger.info(f"  q{qi}: {len(rw)} вариантов")

    # ── Judge ──
    judge = create_judge(args.api_key, args.api_base, args.model)
    new_ann_count = 0

    def get_label(qi, doc):
        nonlocal new_ann_count
        sf = doc.metadata.get("source_file", "")
        if sf in gt[qi]:
            return gt[qi][sf]
        result = judge_single(judge, queries[qi], doc.page_content, sf)
        label = result.get("label", "irrelevant")
        reason = result.get("reason", "")
        gt[qi][sf] = label
        existing = [a for a in annotations.values() if a["query_idx"] == qi]
        next_idx = max((a.get("doc_idx", 0) for a in existing), default=-1) + 1
        key = f"{qi}_{next_idx}"
        annotations[key] = {
            "query_idx": qi, "doc_idx": next_idx,
            "source_file": sf, "label": label,
            "auto": True, "llm_reason": reason,
        }
        new_ann_count += 1
        icon = {"relevant": "✅", "partial": "⚠️", "irrelevant": "❌"}
        logger.info(f"    AUTO: {icon.get(label, '?')} {sf[:50]} → {label}")
        time.sleep(0.3)
        return label

    # ── 12 конфигураций ──
    # Для closure capture
    _bm25 = bm25_index
    _vs = vectorstore

    configs = [
        # BM25
        ("1. BM25 only",                    "bm25",   False, False),
        ("2. BM25 + rewrite",               "bm25",   True,  False),
        ("3. BM25 + reranker",              "bm25",   False, True),
        ("4. BM25 + rewrite + reranker",    "bm25",   True,  True),
        # Vector
        ("5. Vector only",                   "vector", False, False),
        ("6. Vector + rewrite",              "vector", True,  False),
        ("7. Vector + reranker",             "vector", False, True),
        ("8. Vector + rewrite + reranker",   "vector", True,  True),
        # Hybrid
        ("9. Hybrid (BM25+Vec)",             "hybrid", False, False),
        ("10. Hybrid + rewrite",             "hybrid", True,  False),
        ("11. Hybrid + reranker",            "hybrid", False, True),
        ("12. Hybrid + rewrite + reranker",  "hybrid", True,  True),
    ]

    all_results = {}

    for cfg_name, mode, use_rewrite, use_rerank in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"  {cfg_name}")
        logger.info(f"{'='*60}")

        retrieve_fn = make_retrieve_fn(mode, _bm25, _vs)
        labels_per_query = []
        times_list = []

        for qi in annotated_qi:
            t0 = time.time()

            q_list = rewrites[qi] if use_rewrite else [queries[qi]]
            top_k = 15 if use_rerank else 5

            docs = multi_query_retrieve(retrieve_fn, q_list, top_k_final=top_k)

            if use_rerank:
                docs = reranker.rerank(queries[qi], docs, top_k=5)

            elapsed = time.time() - t0
            times_list.append(elapsed)

            labels = []
            for doc in docs[:5]:
                labels.append(get_label(qi, doc))
            while len(labels) < 5:
                labels.append("irrelevant")

            labels_per_query.append(labels)

            icon = {"relevant": "✅", "partial": "⚠️", "irrelevant": "❌"}
            ls = " ".join(icon.get(l, "?") for l in labels)
            logger.info(f"  q{qi:>2} [{elapsed:.1f}s]: {ls}")

        metrics = compute_ablation_metrics(labels_per_query)
        metrics["avg_time"] = sum(times_list) / len(times_list)
        all_results[cfg_name] = metrics

        logger.info(
            f"  → P@3={metrics['P@3']:.3f} P@5={metrics['P@5']:.3f} "
            f"NDCG@5={metrics['NDCG@5']:.3f} MRR={metrics['MRR']:.3f} "
            f"t={metrics['avg_time']:.2f}s"
        )

        # Промежуточное сохранение
        data["annotations"] = annotations
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Финал ──
    logger.info(f"\nДоразмечено: {new_ann_count}")
    logger.info(f"Всего аннотаций: {len(annotations)}")

    results_output = Path(args.results_output)
    results_output.parent.mkdir(parents=True, exist_ok=True)
    with results_output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 100)
    print(f"  ABLATION STUDY v2 — E5-large-instruct — {len(annotated_qi)} запросов, {len(annotations)} аннотаций")
    print("=" * 100)
    print(f"{'#':<4} {'Конфигурация':<40} {'P@3':>7} {'P@5':>7} {'NDCG@5':>8} {'MRR':>7} {'Time':>7}")
    print("-" * 100)

    prev_group = ""
    for cfg_name, m in all_results.items():
        group = cfg_name.split(".")[1].strip().split()[0]  # BM25/Vector/Hybrid
        if group != prev_group:
            if prev_group:
                print("-" * 100)
            prev_group = group
        num = cfg_name.split(".")[0]
        name = cfg_name.split(".", 1)[1].strip()
        print(
            f"{num:>3}. {name:<39} {m['P@3']:>7.3f} {m['P@5']:>7.3f} "
            f"{m['NDCG@5']:>8.3f} {m['MRR']:>7.3f} {m['avg_time']:>6.2f}s"
        )
    print("=" * 100)

    # Лучшая конфигурация
    best = max(all_results.items(), key=lambda x: x[1]["NDCG@5"])
    print(f"\n🏆 Лучшая по NDCG@5: {best[0]}")
    print(f"   P@3={best[1]['P@3']:.3f} P@5={best[1]['P@5']:.3f} NDCG@5={best[1]['NDCG@5']:.3f} MRR={best[1]['MRR']:.3f}")

    best_p = max(all_results.items(), key=lambda x: x[1]["P@5"])
    print(f"\n🎯 Лучшая по P@5: {best_p[0]}")
    print(f"   P@3={best_p[1]['P@3']:.3f} P@5={best_p[1]['P@5']:.3f} NDCG@5={best_p[1]['NDCG@5']:.3f} MRR={best_p[1]['MRR']:.3f}")


if __name__ == "__main__":
    main()
