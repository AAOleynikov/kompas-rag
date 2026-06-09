"""LLM-as-a-judge generation evaluation."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..chain import KompasRAGChain, postprocess_references, preprocess_context
from ..config import load_config
from ..pipeline import build_rag_pipeline

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
Ты эксперт по оценке качества ответов справочной RAG-системы по САПР КОМПАС-3D.

Тебе будут даны:
1. вопрос пользователя;
2. найденные системой фрагменты документации;
3. ответ системы.

Оцени ответ по четырем критериям: faithfulness, completeness, relevance, helpfulness.
Для каждого критерия поставь "yes", "partial" или "no" и дай краткое обоснование.
"""

JUDGE_PROMPT = """\
## Вопрос пользователя
{question}

## Найденные фрагменты документации
{context}

## Ответ системы
{answer}

---

Критерии:

1. faithfulness: ответ основан только на предоставленных фрагментах?
2. completeness: ответ достаточно полно покрывает вопрос с учетом контекста?
3. relevance: ответ по делу, без лишней нерелевантной информации?
4. helpfulness: поможет ли ответ пользователю КОМПАС-3D решить задачу?

Ответь строго JSON без markdown:
{{
  "faithfulness": {{"score": "yes|partial|no", "reason": "..."}},
  "completeness": {{"score": "yes|partial|no", "reason": "..."}},
  "relevance": {{"score": "yes|partial|no", "reason": "..."}},
  "helpfulness": {{"score": "yes|partial|no", "reason": "..."}}
}}
"""

SCORE_MAP = {"yes": 1.0, "partial": 0.5, "no": 0.0, "error": None}
CRITERIA = ["faithfulness", "completeness", "relevance", "helpfulness"]
CRITERIA_LABELS_RU = {
    "faithfulness": "Точность",
    "completeness": "Полнота",
    "relevance": "Релевантность",
    "helpfulness": "Полезность",
}


def init_rag_for_generation_eval(
    persist_dir: str,
    config_path: str | None = None,
) -> KompasRAGChain:
    """Build RAG for generation eval, preserving the old no-rerank behavior."""
    config = load_config(config_path)
    config.vectorstore.chroma_persist_dir = persist_dir
    config.reranker.enabled = False
    return build_rag_pipeline(config).chain


def run_rag_for_judge(chain: KompasRAGChain, question: str) -> dict:
    started_at = time.time()
    docs = chain.retriever.retrieve(question)

    if docs:
        context, refs = preprocess_context(docs, chain.docs_base_url)
        answer_raw = chain.chain.invoke({"context": context, "question": question})
        answer = postprocess_references(answer_raw, refs)
    else:
        refs = []
        answer = "Не найдено информации."

    context_for_judge = ""
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata
        context_for_judge += f"\n--- Фрагмент {idx} ---\n"
        context_for_judge += f"Файл: {meta.get('source_file', '?')}\n"
        context_for_judge += f"Раздел: {meta.get('page_title', '?')}\n"
        context_for_judge += doc.page_content[:1500]
        context_for_judge += "\n"

    return {
        "question": question,
        "answer": answer,
        "context_for_judge": context_for_judge,
        "n_docs": len(docs),
        "n_refs": len(refs),
        "time": round(time.time() - started_at, 1),
    }


def create_judge_llm(api_key: str, api_base: str, model: str) -> ChatOpenAI:
    http_client = httpx.Client(
        base_url=api_base,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        verify=False,
        timeout=120.0,
    )
    return ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=model,
        max_tokens=1000,
        temperature=0,
        http_client=http_client,
    )


def judge_answer(
    judge_llm: ChatOpenAI,
    question: str,
    context: str,
    answer: str,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:6000],
        answer=answer[:4000],
    )
    response = judge_llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Judge JSON parse error: %s", raw[:300])
        return {
            criterion: {"score": "error", "reason": raw[:200]}
            for criterion in CRITERIA
        }


def compute_generation_metrics(evaluations: list[dict]) -> dict:
    stats: dict[str, dict | float] = {}

    for criterion in CRITERIA:
        scores = []
        counts = {"yes": 0, "partial": 0, "no": 0, "error": 0}

        for evaluation in evaluations:
            score = evaluation.get("judgment", {}).get(criterion, {}).get("score", "error")
            counts[score] = counts.get(score, 0) + 1
            if score in SCORE_MAP and SCORE_MAP[score] is not None:
                scores.append(SCORE_MAP[score])

        average = sum(scores) / len(scores) if scores else 0
        stats[criterion] = {
            "avg": round(average, 3),
            "yes": counts["yes"],
            "partial": counts["partial"],
            "no": counts["no"],
            "error": counts["error"],
            "n": len(scores),
        }

    averages = [stats[criterion]["avg"] for criterion in CRITERIA]
    stats["overall"] = round(sum(averages) / len(averages), 3) if averages else 0
    return stats


def print_generation_report(evaluations: list[dict], stats: dict) -> None:
    print()
    print("=" * 70)
    print(f"RAG GENERATION METRICS (LLM-as-a-Judge) - {len(evaluations)} запросов")
    print("=" * 70)
    print()

    for criterion in CRITERIA:
        criterion_stats = stats[criterion]
        print(
            f"{CRITERIA_LABELS_RU[criterion]:<14} "
            f"avg={criterion_stats['avg']:.3f} "
            f"yes={criterion_stats['yes']} "
            f"partial={criterion_stats['partial']} "
            f"no={criterion_stats['no']} "
            f"n={criterion_stats['n']}"
        )
    print(f"OVERALL={stats['overall']:.3f}")
    print()

    for idx, evaluation in enumerate(evaluations):
        judgment = evaluation.get("judgment", {})
        marks = " ".join(
            judgment.get(criterion, {}).get("score", "error")[:1]
            for criterion in CRITERIA
        )
        print(f"{idx:>3} {marks}  {evaluation['question'][:60]}")


def evaluate_generation(
    *,
    persist_dir: str,
    annotations_path: str,
    api_key: str,
    api_base: str,
    model: str,
    output_path: str,
    limit: int | None = None,
    config_path: str | None = None,
) -> dict:
    with open(annotations_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    queries = data["queries"]
    if limit:
        queries = queries[:limit]

    logger.info("Queries: %s", len(queries))
    logger.info("Judge: %s @ %s", model, api_base)

    chain = init_rag_for_generation_eval(persist_dir, config_path)
    judge_llm = create_judge_llm(api_key, api_base, model)

    evaluations = []
    for idx, question in enumerate(queries):
        logger.info("[%s/%s] %s...", idx + 1, len(queries), question[:50])
        rag_result = run_rag_for_judge(chain, question)

        try:
            judgment = judge_answer(
                judge_llm,
                question=rag_result["question"],
                context=rag_result["context_for_judge"],
                answer=rag_result["answer"],
            )
        except Exception as exc:
            logger.error("Judge error: %s", exc)
            judgment = {
                criterion: {"score": "error", "reason": str(exc)}
                for criterion in CRITERIA
            }

        evaluations.append(
            {
                "idx": idx,
                "question": question,
                "answer": rag_result["answer"],
                "n_docs": rag_result["n_docs"],
                "rag_time": rag_result["time"],
                "judgment": judgment,
            }
        )
        logger.info(
            "  -> %s",
            " | ".join(
                f"{criterion[:5]}={judgment.get(criterion, {}).get('score', '?')}"
                for criterion in CRITERIA
            ),
        )
        time.sleep(1)

    stats = compute_generation_metrics(evaluations)
    print_generation_report(evaluations, stats)

    output = {
        "created": datetime.now().isoformat(),
        "model_judge": model,
        "n_queries": len(queries),
        "stats": stats,
        "evaluations": evaluations,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", path)
    return output
