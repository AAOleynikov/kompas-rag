"""CLI wrapper for LLM-as-a-judge generation evaluation."""

import argparse
import logging

from .evaluation.generation import evaluate_generation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

for noisy_logger in ["httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge для RAG")
    parser.add_argument("--config", default=None)
    parser.add_argument("--persist-dir", default="/storage/chroma_db")
    parser.add_argument("--annotations", default="/storage/annotations.json")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default="anthropic/claude-opus-4-20250514")
    parser.add_argument("--output", default="/storage/generation_eval.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    evaluate_generation(
        persist_dir=args.persist_dir,
        annotations_path=args.annotations,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        output_path=args.output,
        limit=args.limit,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
