"""Gradio web UI for the KOMPAS-3D RAG assistant."""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import gradio as gr

from .chain import postprocess_references, preprocess_context
from .config import RAGConfig, load_config
from .pipeline import build_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

for noisy_logger in ["httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class RAGEngine:
    """Lazy holder for the shared runtime RAG pipeline."""

    def __init__(self):
        self.chain = None
        self.retriever = None
        self.llm = None
        self.config: Optional[RAGConfig] = None
        self.ready = False

    def initialize(self, config: Optional[RAGConfig] = None):
        if self.ready:
            return

        logger.info("Initializing RAG...")
        self.config = config or RAGConfig()
        pipeline = build_rag_pipeline(self.config)
        self.llm = pipeline.llm
        self.retriever = pipeline.retriever
        self.chain = pipeline.chain
        self.ready = True
        logger.info("RAG ready")


engine = RAGEngine()


@dataclass
class ChatSession:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "Новый чат"
    history: list = field(default_factory=list)
    metadata: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class ChatManager:
    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.current_id: Optional[str] = None
        self._create_session()

    def _create_session(self):
        session = ChatSession()
        self.sessions[session.id] = session
        self.current_id = session.id
        return session

    @property
    def current(self):
        if self.current_id not in self.sessions:
            self._create_session()
        return self.sessions[self.current_id]

    def new_chat(self):
        return self._create_session()

    def switch_to(self, session_id):
        if session_id in self.sessions:
            self.current_id = session_id

    def delete_chat(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
        if not self.sessions:
            self._create_session()
        elif self.current_id == session_id:
            self.current_id = list(self.sessions.keys())[-1]

    def get_choices(self):
        sessions = sorted(
            self.sessions.values(),
            key=lambda session: session.created_at,
            reverse=True,
        )
        return [(session.title, session.id) for session in sessions]


chat_mgr = ChatManager()


def process_query(message, history):
    """Run retrieval and generation for one chat message."""
    message = message.strip()
    if not message:
        return history, "", "", "", ""

    if not engine.ready:
        engine.initialize()

    session = chat_mgr.current
    started_at = time.time()

    rewrite_started = time.time()
    queries = (
        engine.retriever.query_rewriter(message)
        if engine.retriever.query_rewriter
        else [message]
    )
    rewrite_time = time.time() - rewrite_started

    retrieval_started = time.time()
    docs = engine.retriever.retrieve(message)
    retrieval_time = time.time() - retrieval_started

    generation_started = time.time()
    if docs:
        context, refs = preprocess_context(docs, engine.chain.docs_base_url)
        answer_raw = engine.chain.chain.invoke({"context": context, "question": message})
        answer = postprocess_references(answer_raw, refs)
    else:
        refs = []
        answer = (
            "К сожалению, я не нашел информации по этому вопросу "
            "в документации КОМПАС-3D."
        )
    generation_time = time.time() - generation_started
    total_time = time.time() - started_at

    if not session.history:
        session.title = message[:40] + ("..." if len(message) > 40 else "")

    session.history.append((message, answer))
    session.metadata.append(
        {"queries": queries, "n_docs": len(docs), "time": round(total_time, 1)}
    )

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    q_md = build_queries_panel(message, queries)
    s_md = build_sources_panel(refs)
    d_md = build_debug_panel(
        rewrite_time=rewrite_time,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time,
        n_queries=len(queries),
        n_docs=len(docs),
        n_refs=len(refs),
    )
    return history, q_md, s_md, d_md, ""


def build_queries_panel(original_query: str, queries: list[str]) -> str:
    lines = [
        "### Перефразирование запроса",
        "",
        f"**Оригинал:** {original_query}",
        "",
    ]
    if len(queries) > 1:
        lines.append("**Поисковые запросы:**")
        lines.append("")
        lines.extend(f"- `{query}`" for query in queries)
    else:
        lines.append("Запрос не перефразирован.")
    return "\n".join(lines)


def build_sources_panel(refs) -> str:
    lines = ["### Источники", ""]
    if not refs:
        lines.append("Ничего не найдено.")
        return "\n".join(lines)

    for ref in refs:
        title = ref.section_path or ref.page_title
        if ref.url:
            lines.append(f"**[{ref.number}]** [{title}]({ref.url})")
        else:
            lines.append(f"**[{ref.number}]** {title}")
        if ref.breadcrumbs:
            lines.append(f"_{ref.breadcrumbs}_")
        lines.append("")
    return "\n".join(lines)


def build_debug_panel(
    *,
    rewrite_time: float,
    retrieval_time: float,
    generation_time: float,
    total_time: float,
    n_queries: int,
    n_docs: int,
    n_refs: int,
) -> str:
    return f"""### Детали

| Параметр | Значение |
|---|---:|
| Rewrite | {rewrite_time:.1f} c |
| Retrieval | {retrieval_time:.1f} c |
| Generation | {generation_time:.1f} c |
| **Итого** | **{total_time:.1f} c** |
| Запросов | {n_queries} |
| Документов | {n_docs} |
| Источников | {n_refs} |
"""


def on_new_chat():
    chat_mgr.new_chat()
    return [], "", "", "", gr.update(
        choices=chat_mgr.get_choices(),
        value=chat_mgr.current_id,
    )


def on_switch(session_id):
    if not session_id:
        return [], "", "", ""
    chat_mgr.switch_to(session_id)
    history = []
    for user_message, assistant_message in chat_mgr.current.history:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
    return history, "", "", ""


def on_delete():
    chat_mgr.delete_chat(chat_mgr.current_id)
    return [], "", "", "", gr.update(
        choices=chat_mgr.get_choices(),
        value=chat_mgr.current_id,
    )


def on_submit(message, history):
    if not message.strip():
        return history, "", "", "", ""
    new_history, queries, sources, debug, _ = process_query(message, history)
    return new_history, queries, sources, debug, "", gr.update(
        choices=chat_mgr.get_choices(),
        value=chat_mgr.current_id,
    )


EXAMPLES = [
    "Какая видеокарта нужна для КОМПАС-3D?",
    "Как нарисовать отверстие в детали?",
    "Как сделать фаску на ребре?",
    "Как вставить деталь в сборку?",
    "Детали пересекаются, как проверить?",
    "Как проставить размеры на чертеже?",
    "Как сохранить в PDF?",
    "Открыть файл SolidWorks в КОМПАСе?",
    "Лагает компьютер в КОМПАСе",
    "Куда делась панель инструментов?",
]


def create_ui():
    with gr.Blocks(title="КОМПАС-3D Помощник") as app:
        gr.HTML(
            """
            <div style="text-align:center; padding:10px 0;">
                <h1>КОМПАС-3D Помощник</h1>
                <p style="color:#666;">RAG по документации КОМПАС-3D v24 | Vector + BM25 | Query rewriting</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### Чаты")
                new_btn = gr.Button("Новый чат", variant="primary", size="sm")
                chat_radio = gr.Radio(
                    choices=chat_mgr.get_choices(),
                    value=chat_mgr.current_id,
                    label="",
                    interactive=True,
                )
                del_btn = gr.Button("Удалить", variant="stop", size="sm")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    label="Диалог",
                    render_markdown=True,
                    type="messages",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Задайте вопрос по КОМПАС-3D...",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send = gr.Button("Отправить", variant="primary", scale=1)

                gr.Markdown("**Примеры:**")
                with gr.Row():
                    for i in range(0, len(EXAMPLES), 2):
                        with gr.Column():
                            for example in EXAMPLES[i : i + 2]:
                                gr.Button(example, size="sm").click(
                                    fn=lambda value=example: value,
                                    outputs=msg,
                                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Запросы"):
                        q_panel = gr.Markdown("Задайте вопрос.")
                    with gr.Tab("Источники"):
                        s_panel = gr.Markdown("Задайте вопрос.")
                    with gr.Tab("Детали"):
                        d_panel = gr.Markdown("Задайте вопрос.")

        submit_out = [chatbot, q_panel, s_panel, d_panel, msg, chat_radio]
        send.click(fn=on_submit, inputs=[msg, chatbot], outputs=submit_out)
        msg.submit(fn=on_submit, inputs=[msg, chatbot], outputs=submit_out)

        new_btn.click(
            fn=on_new_chat,
            outputs=[chatbot, q_panel, s_panel, d_panel, chat_radio],
        )
        chat_radio.change(
            fn=on_switch,
            inputs=[chat_radio],
            outputs=[chatbot, q_panel, s_panel, d_panel],
        )
        del_btn.click(
            fn=on_delete,
            outputs=[chatbot, q_panel, s_panel, d_panel, chat_radio],
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.persist_dir:
        config.vectorstore.chroma_persist_dir = args.persist_dir

    engine.initialize(config)
    app = create_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
