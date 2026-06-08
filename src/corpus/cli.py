"""CLI for preparing KOMPAS-3D HTML documentation chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HTML documentation -> JSONL chunks for KOMPAS-3D RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/run_chunker.py ./kompas_docs/ -o data/processed/chunks.jsonl
  python src/run_chunker.py ./kompas_docs/some_file.html --single --preview
  python src/run_chunker.py ./kompas_docs/ -o data/processed/chunks.jsonl --tree hmcontent.js
""",
    )

    parser.add_argument("input", help="HTML directory or one HTML file with --single")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--also-csv", default=None)
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--min-size", type=int, default=200)
    parser.add_argument("--max-size", type=int, default=3000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--no-see-also", action="store_true")
    parser.add_argument("--no-dropdowns", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--tree", default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> CorpusBuildConfig:
    from .pipeline import CorpusBuildConfig

    return CorpusBuildConfig(
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        tree_path=Path(args.tree) if args.tree else None,
        csv_path=Path(args.also_csv) if args.also_csv else None,
        single=args.single,
        target_size=args.target_size,
        min_size=args.min_size,
        max_size=args.max_size,
        overlap=args.overlap,
        include_see_also=not args.no_see_also,
        include_dropdowns=not args.no_dropdowns,
        include_images=not args.no_images,
        max_files=args.max_files,
        preview=args.preview,
        preview_count=args.preview_count,
        stats=args.stats,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .pipeline import build_chunks, write_outputs

    config = config_from_args(args)
    try:
        chunks = build_chunks(config)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        return 1

    if not chunks:
        logger.warning("No chunks were created")
        return 0

    write_outputs(config, chunks)
    print(f"\nDone. Created {len(chunks)} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
