"""Gradio relevance annotator for retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

from .config import load_config
from .pipeline import build_rag_pipeline
from .retriever import KompasRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

for noisy_logger in ["httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class AnnotatorState:
    """State holder for query-level retrieval annotations."""

    def __init__(self):
        self.queries: list[str] = []
        self.retrieval_results: dict[int, list[dict]] = {}
        self.annotations: dict[str, dict] = {}
        self.current_idx: int = 0
        self.retriever: Optional[KompasRetriever] = None
        self.output_path = Path("/storage/annotations.json")
        self.ready = False

    def initialize(
        self,
        persist_dir: str | None = None,
        output_path: str | None = None,
        config_path: str | None = None,
    ):
        if self.ready:
            return

        config = load_config(config_path)
        if persist_dir:
            config.vectorstore.chroma_persist_dir = persist_dir
        if output_path:
            self.output_path = Path(output_path)

        pipeline = build_rag_pipeline(config)
        self.retriever = pipeline.retriever
        self.ready = True
        logger.info("Annotator ready")

    def load_queries(self, text: str) -> int:
        self.queries = [query.strip() for query in text.strip().splitlines() if query.strip()]
        self.current_idx = 0
        self.retrieval_results = {}

        if self.output_path.exists():
            with self.output_path.open("r", encoding="utf-8") as stream:
                saved = json.load(stream)
            self.annotations = saved.get("annotations", {})
            logger.info("Loaded %s existing annotations", len(self.annotations))
        return len(self.queries)

    def get_docs_for_query(self, idx: int) -> list[dict]:
        if idx in self.retrieval_results:
            return self.retrieval_results[idx]

        if not self.ready or self.retriever is None or idx >= len(self.queries):
            return []

        query = self.queries[idx]
        started_at = time.time()
        docs = self.retriever.retrieve(query, top_k=5)
        elapsed = time.time() - started_at

        results = []
        for doc_idx, doc in enumerate(docs):
            meta = doc.metadata
            results.append(
                {
                    "doc_idx": doc_idx,
                    "text": doc.page_content[:1500],
                    "full_text": doc.page_content,
                    "source_file": meta.get("source_file", ""),
                    "page_title": meta.get("page_title", ""),
                    "section_path": meta.get("section_path", ""),
                    "breadcrumbs": meta.get("breadcrumbs", ""),
                    "fusion_score": meta.get("fusion_score", 0),
                    "reranker_score": meta.get("reranker_score", ""),
                    "retrieval_time": round(elapsed, 2),
                }
            )

        self.retrieval_results[idx] = results
        return results

    def annotate(self, query_idx: int, doc_idx: int, label: str, comment: str = ""):
        key = f"q{query_idx}_d{doc_idx}"
        self.annotations[key] = {
            "query_idx": query_idx,
            "query": self.queries[query_idx] if query_idx < len(self.queries) else "",
            "doc_idx": doc_idx,
            "source_file": "",
            "label": label,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }

        docs = self.retrieval_results.get(query_idx, [])
        if doc_idx < len(docs):
            self.annotations[key]["source_file"] = docs[doc_idx]["source_file"]
            self.annotations[key]["page_title"] = docs[doc_idx]["page_title"]

    def save(self) -> str:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "created": datetime.now().isoformat(),
            "queries": self.queries,
            "annotations": self.annotations,
            "stats": self._compute_stats(),
        }
        with self.output_path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        logger.info("Saved annotations to %s", self.output_path)
        return str(self.output_path)

    def _compute_stats(self) -> dict:
        if not self.annotations:
            return {}

        labels = [annotation["label"] for annotation in self.annotations.values()]
        total = len(labels)
        return {
            "total_annotations": total,
            "relevant": labels.count("relevant"),
            "partial": labels.count("partial"),
            "irrelevant": labels.count("irrelevant"),
            "relevant_pct": round(100 * labels.count("relevant") / total, 1) if total else 0,
            "queries_total": len(self.queries),
            "queries_annotated": len(
                {annotation["query_idx"] for annotation in self.annotations.values()}
            ),
        }

    def get_progress_stats(self) -> str:
        stats = self._compute_stats()
        return (
            f"**Прогресс:** {stats.get('queries_annotated', 0)}/{len(self.queries)} "
            f"запросов | {stats.get('total_annotations', 0)} аннотаций | "
            f"релевантных: {stats.get('relevant', 0)} | "
            f"частичных: {stats.get('partial', 0)} | "
            f"нерелевантных: {stats.get('irrelevant', 0)}"
        )


state = AnnotatorState()


def render_query_page(idx: int):
    """Render one query and its top retrieved fragments."""
    if idx < 0 or idx >= len(state.queries):
        return "Нет запросов", "", "", "", "", "", "", idx

    query = state.queries[idx]
    docs = state.get_docs_for_query(idx)

    header = f"## Запрос {idx + 1} / {len(state.queries)}\n\n"
    header += f"### «{query}»\n\n"
    if docs:
        header += f"_Найдено {len(docs)} фрагментов за {docs[0].get('retrieval_time', '?')} c_"

    fragments = []
    for doc_idx, doc in enumerate(docs):
        key = f"q{idx}_d{doc_idx}"
        existing_label = state.annotations.get(key, {}).get("label", "")
        badge = {
            "relevant": " [релевантен]",
            "partial": " [частично]",
            "irrelevant": " [нерелевантен]",
        }.get(existing_label, "")

        score_parts = [f"fusion={doc['fusion_score']}"]
        if doc.get("reranker_score") != "":
            score_parts.append(f"reranker={doc['reranker_score']}")

        text_preview = doc["text"]
        if len(text_preview) > 800:
            text_preview = text_preview[:800] + "\n\n_[...обрезано]_"

        fragment = f"""---

### Фрагмент {doc_idx + 1}{badge}

**Файл:** `{doc['source_file']}`

**Раздел:** {doc['page_title']}{' > ' + doc['section_path'] if doc['section_path'] else ''}

**Breadcrumbs:** _{doc['breadcrumbs'] or '-'}_

**Score:** {', '.join(score_parts)}

```
{text_preview}
```
"""
        fragments.append(fragment)

    panels = [fragments[i] if i < len(fragments) else "" for i in range(5)]
    return header, panels[0], panels[1], panels[2], panels[3], panels[4], state.get_progress_stats(), idx


def handle_annotation(idx, doc_idx, label, comment=""):
    if idx < len(state.queries):
        state.annotate(idx, doc_idx, label, comment)
        state.save()
    return render_query_page(idx)


DEFAULT_QUERIES = """Создание штампа
Согнуть прокатный профиль и получить развертку
Можно ли обрезать полигональный объект в КОМПАС?
Как преобразовать деталь в листовую"""


def create_annotator_ui():
    with gr.Blocks(title="RAG Annotator - КОМПАС-3D") as app:
        current_idx = gr.State(0)

        gr.HTML(
            """
            <div style="text-align:center; padding:8px 0;">
                <h1>Разметка релевантности RAG</h1>
                <p style="color:#666;">Оцените top-5 найденных фрагментов для каждого запроса.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                queries_input = gr.Textbox(
                    label="Запросы, по одному на строку",
                    lines=8,
                    value=DEFAULT_QUERIES,
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("Загрузить запросы", variant="primary", size="lg")
                status_text = gr.Markdown("")

        progress_bar = gr.Markdown("**Загрузите запросы для начала.**")

        with gr.Row():
            prev_btn = gr.Button("Назад", size="sm")
            query_slider = gr.Slider(
                minimum=1,
                maximum=1,
                step=1,
                value=1,
                label="Запрос N",
                interactive=True,
            )
            next_btn = gr.Button("Далее", size="sm", variant="primary")
            save_btn = gr.Button("Сохранить", size="sm", variant="secondary")

        query_header = gr.Markdown("")
        fragment_panels = []
        button_groups = []

        for doc_idx in range(5):
            with gr.Group():
                fragment_panel = gr.Markdown("", elem_id=f"fragment_{doc_idx}")
                fragment_panels.append(fragment_panel)
                with gr.Row():
                    btn_rel = gr.Button("Релевантен", size="sm", variant="primary")
                    btn_part = gr.Button("Частично", size="sm", variant="secondary")
                    btn_irr = gr.Button("Нерелевантен", size="sm", variant="stop")
                    button_groups.append((btn_rel, btn_part, btn_irr))

        all_outputs = [
            query_header,
            fragment_panels[0],
            fragment_panels[1],
            fragment_panels[2],
            fragment_panels[3],
            fragment_panels[4],
            progress_bar,
            current_idx,
        ]

        def on_load(queries_text):
            n_queries = state.load_queries(queries_text)
            if n_queries == 0:
                return [gr.update(maximum=1, value=1), "Нет запросов"] + [""] * 7 + [0]

            return [
                gr.update(maximum=n_queries, value=1),
                f"Загружено запросов: {n_queries}",
            ] + list(render_query_page(0))

        load_btn.click(
            fn=on_load,
            inputs=[queries_input],
            outputs=[query_slider, status_text] + all_outputs,
        )

        def on_slider(value):
            return render_query_page(int(value) - 1)

        query_slider.change(fn=on_slider, inputs=[query_slider], outputs=all_outputs)

        def on_prev(idx):
            new_idx = max(0, idx - 1)
            return list(render_query_page(new_idx)) + [gr.update(value=new_idx + 1)]

        prev_btn.click(
            fn=on_prev,
            inputs=[current_idx],
            outputs=all_outputs + [query_slider],
        )

        def on_next(idx):
            new_idx = min(len(state.queries) - 1, idx + 1)
            return list(render_query_page(new_idx)) + [gr.update(value=new_idx + 1)]

        next_btn.click(
            fn=on_next,
            inputs=[current_idx],
            outputs=all_outputs + [query_slider],
        )

        save_btn.click(fn=lambda: f"Сохранено: `{state.save()}`", outputs=[status_text])

        for doc_idx, buttons in enumerate(button_groups):
            for button, label in [
                (buttons[0], "relevant"),
                (buttons[1], "partial"),
                (buttons[2], "irrelevant"),
            ]:
                button.click(
                    fn=lambda idx, label=label, doc_idx=doc_idx: handle_annotation(
                        idx,
                        doc_idx,
                        label,
                    ),
                    inputs=[current_idx],
                    outputs=all_outputs,
                )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    state.initialize(
        persist_dir=args.persist_dir,
        output_path=args.output,
        config_path=args.config,
    )

    app = create_annotator_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
