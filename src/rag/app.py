"""CLI application for the KOMPAS-3D RAG assistant."""

from __future__ import annotations

import argparse
import logging

from .chain import KompasRAGChain
from .config import RAGConfig, load_config
from .pipeline import build_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_rag(config: RAGConfig) -> KompasRAGChain:
    """Build the RAG chain through the shared runtime pipeline."""
    return build_rag_pipeline(config).chain


def interactive_mode(chain: KompasRAGChain):
    """Interactive question-answer loop."""
    print("\n" + "=" * 60)
    print("  КОМПАС-3D RAG - помощник по документации")
    print("=" * 60)
    print("Введите вопрос или 'выход' для завершения.")
    print("Команды: /debug, /sources, /raw")

    show_sources = False
    show_raw = False
    debug = False

    while True:
        try:
            question = input("\n? Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not question:
            continue

        if question.lower() in ("выход", "exit", "quit", "q"):
            print("До свидания!")
            break

        if question == "/debug":
            debug = not debug
            chain.config.debug = debug
            print(f"Отладка: {'вкл' if debug else 'выкл'}")
            continue

        if question == "/sources":
            show_sources = not show_sources
            print(f"Источники: {'вкл' if show_sources else 'выкл'}")
            continue

        if question == "/raw":
            show_raw = not show_raw
            print(f"Сырой ответ: {'вкл' if show_raw else 'выкл'}")
            continue

        try:
            result = chain.ask(question, return_sources=show_sources)
        except Exception as exc:
            print(f"Ошибка: {exc}")
            logger.exception("RAG error")
            continue

        print(f"\nОтвет:\n{result['answer']}")

        if show_raw and result.get("answer_raw"):
            print(f"\nСырой ответ LLM:\n{result['answer_raw']}")

        if show_sources and result["sources"]:
            print("\nИсточники из контекста:")
            for source in result["sources"]:
                print(
                    f"  [{source['number']}] {source['section_path']} "
                    f"-> {source['url']}"
                )

        if debug and result.get("context"):
            print(f"\nКонтекст ({len(result['context'])} символов):")
            print(result["context"][:500])
            print("...")


def single_query(chain: KompasRAGChain, question: str, show_sources: bool):
    """Run one query without entering interactive mode."""
    result = chain.ask(question, return_sources=show_sources)
    print(result["answer"])
    if show_sources and result["sources"]:
        print("\nИсточники:")
        for source in result["sources"]:
            print(f"  [{source['number']}] {source['section_path']} -> {source['url']}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-помощник по документации КОМПАС-3D"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--llm-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("-q", "--question")
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--backend", choices=["chroma", "qdrant"], default=None)
    parser.add_argument("--docs-url", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-rewrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config.vectorstore.backend = args.backend
    if args.persist_dir:
        config.vectorstore.chroma_persist_dir = args.persist_dir
    if args.collection:
        config.vectorstore.collection_name = args.collection
    if args.llm_url:
        config.llm.base_url = args.llm_url
    if args.llm_model:
        config.llm.model_name = args.llm_model
    if args.docs_url:
        config.docs_base_url = args.docs_url
    config.debug = args.debug
    if args.no_rerank:
        config.reranker.enabled = False
    if args.no_rewrite:
        config.query_rewrite.enabled = False

    chain = build_rag(config)

    if args.question:
        single_query(chain, args.question, args.sources)
    else:
        interactive_mode(chain)


if __name__ == "__main__":
    main()
