"""HTML corpus preparation orchestration.

The heavy parsing and chunking algorithms remain in the existing modules:
``convert_kompas_html_to_md.py`` and ``chunker.py``. This module only owns the
pipeline glue: file traversal, component construction, preview, stats and
exports. Keeping that boundary stable protects the current chunks and metrics.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from chunker import (
    Chunk,
    ChunkerConfig,
    SemanticBlockParser,
    SemanticChunker,
    export_chunks_jsonl,
)
from convert_kompas_html_to_md import (
    ConversionConfig,
    HTMLToMarkdownConverter,
    parse_toc_tree,
)

logger = logging.getLogger(__name__)

SERVICE_HTML_NAME_PARTS = (
    "hmcontent",
    "hmftsearch",
    "hmkwindex",
    "hmcontextids",
    "zoomimage",
    "open.html",
)


@dataclass
class CorpusBuildConfig:
    """Options for preparing HTML documentation chunks."""

    input_path: Path
    output_path: Path | None = None
    tree_path: Path | None = None
    csv_path: Path | None = None
    single: bool = False
    target_size: int = 1500
    min_size: int = 200
    max_size: int = 3000
    overlap: int = 200
    include_see_also: bool = True
    include_dropdowns: bool = True
    include_images: bool = True
    max_files: int = 0
    preview: bool = False
    preview_count: int = 10
    stats: bool = False


@dataclass
class CorpusComponents:
    converter: HTMLToMarkdownConverter
    parser: SemanticBlockParser
    chunker: SemanticChunker


def create_components(config: CorpusBuildConfig) -> CorpusComponents:
    """Create parser/chunker components from one config object."""
    converter_config = ConversionConfig(
        include_images=config.include_images,
        include_dropdowns=config.include_dropdowns,
        include_see_also=config.include_see_also,
    )
    chunker_config = ChunkerConfig(
        target_chunk_size=config.target_size,
        min_chunk_size=config.min_size,
        max_chunk_size=config.max_size,
        overlap_size=config.overlap,
        include_see_also=config.include_see_also,
        include_command_invocation=config.include_dropdowns,
        include_images=config.include_images,
    )

    converter = HTMLToMarkdownConverter(converter_config)
    parser = SemanticBlockParser(converter)
    chunker = SemanticChunker(chunker_config)
    return CorpusComponents(converter=converter, parser=parser, chunker=chunker)


def load_toc_tree(tree_path: Path | None) -> dict | None:
    """Load HelpMaker TOC breadcrumbs if a tree path was provided."""
    if tree_path is None:
        return None

    if not tree_path.exists():
        logger.warning("TOC tree file not found: %s", tree_path)
        return None

    logger.info("Loading TOC tree: %s", tree_path)
    toc_tree = parse_toc_tree(tree_path)
    logger.info("TOC entries: %s", len(toc_tree))
    return toc_tree


def process_single_file(
    html_path: Path,
    converter: HTMLToMarkdownConverter,
    parser: SemanticBlockParser,
    chunker: SemanticChunker,
    toc_tree: dict | None = None,
) -> list[Chunk]:
    """Process one HTML page into semantic chunks."""
    try:
        html = html_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s: %s", html_path, exc)
        return []

    soup = BeautifulSoup(html, "html.parser")
    metadata, blocks = parser.parse_page(soup)

    if toc_tree and html_path.name in toc_tree:
        toc_info = toc_tree[html_path.name]
        if not metadata.get("breadcrumbs"):
            metadata["breadcrumbs"] = toc_info.get("breadcrumbs", "")
        if not metadata.get("title"):
            metadata["title"] = toc_info.get("title", "")

    if not blocks:
        return []

    return chunker.chunk_page(
        page_metadata=metadata,
        blocks=blocks,
        source_file=html_path.name,
    )


def collect_html_files(input_dir: Path, max_files: int = 0) -> list[Path]:
    """Collect documentation HTML files, excluding HelpMaker service pages."""
    html_files = sorted(input_dir.rglob("*.html"))
    html_files = [
        path
        for path in html_files
        if not any(part in path.name.lower() for part in SERVICE_HTML_NAME_PARTS)
    ]
    if max_files > 0:
        html_files = html_files[:max_files]
    return html_files


def process_directory(
    input_dir: Path,
    converter: HTMLToMarkdownConverter,
    parser: SemanticBlockParser,
    chunker: SemanticChunker,
    toc_tree: dict | None = None,
    max_files: int = 0,
) -> list[Chunk]:
    """Process all documentation HTML pages in a directory."""
    html_files = collect_html_files(input_dir, max_files=max_files)
    total = len(html_files)
    logger.info("Found %s HTML files", total)

    all_chunks: list[Chunk] = []
    skipped = 0
    errors = 0

    for index, path in enumerate(html_files, 1):
        if index % 100 == 0 or index == total:
            logger.info("  processed %s/%s, chunks=%s", index, total, len(all_chunks))

        try:
            chunks = process_single_file(path, converter, parser, chunker, toc_tree)
            if chunks:
                all_chunks.extend(chunks)
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Error while processing %s: %s", path.name, exc)
            errors += 1

    logger.info("Corpus build summary:")
    logger.info("  files processed: %s", total - skipped - errors)
    logger.info("  files skipped: %s", skipped)
    logger.info("  errors: %s", errors)
    logger.info("  chunks created: %s", len(all_chunks))
    return all_chunks


def build_chunks(config: CorpusBuildConfig) -> list[Chunk]:
    """Build chunks from the configured file or directory."""
    components = create_components(config)
    toc_tree = load_toc_tree(config.tree_path)

    if config.single:
        if not config.input_path.is_file():
            raise FileNotFoundError(f"HTML file not found: {config.input_path}")
        logger.info("Processing file: %s", config.input_path)
        return process_single_file(
            config.input_path,
            components.converter,
            components.parser,
            components.chunker,
            toc_tree,
        )

    if not config.input_path.is_dir():
        raise NotADirectoryError(f"HTML directory not found: {config.input_path}")

    return process_directory(
        config.input_path,
        components.converter,
        components.parser,
        components.chunker,
        toc_tree,
        config.max_files,
    )


def preview_chunks(chunks: list[Chunk], max_chunks: int = 10) -> None:
    """Print a compact preview of produced chunks."""
    print(f"\n{'=' * 70}")
    print(f"PREVIEW: {len(chunks)} chunks, showing up to {max_chunks}")
    print(f"{'=' * 70}")

    for index, chunk in enumerate(chunks[:max_chunks], 1):
        print(f"\n{'-' * 70}")
        print(f"CHUNK {index}/{len(chunks)}")
        print(f"  id:          {chunk.chunk_id}")
        print(f"  file:        {chunk.source_file}")
        print(f"  page:        {chunk.page_title}")
        print(f"  section:     {chunk.section_title}")
        print(f"  breadcrumbs: {chunk.breadcrumbs}")
        print(f"  block types: {', '.join(chunk.block_types)}")
        print(f"  chars:       {chunk.char_count}")
        print(f"  tokens ~:    {chunk.token_count_approx}")
        print(f"  table:       {chunk.has_table}")
        print(f"  procedure:   {chunk.has_procedure}")
        print(f"{'-' * 70}")

        text = chunk.text
        if len(text) > 500:
            text = text[:500] + f"\n... ({len(chunk.text) - 500} more chars)"
        print(text)

    if len(chunks) > max_chunks:
        print(f"\n... and {len(chunks) - max_chunks} more chunks")


def print_stats(chunks: list[Chunk]) -> None:
    """Print chunk size and block-type statistics."""
    if not chunks:
        print("No chunks for statistics")
        return

    sizes = [chunk.char_count for chunk in chunks]
    pages = {chunk.source_file for chunk in chunks}
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        for block_type in chunk.block_types:
            type_counts[block_type] = type_counts.get(block_type, 0) + 1

    print(f"\n{'=' * 50}")
    print("STATISTICS")
    print(f"{'=' * 50}")
    print(f"  chunks total:      {len(chunks)}")
    print(f"  unique pages:      {len(pages)}")
    print(f"  chunks per page:   {len(chunks) / len(pages):.1f}")
    print("\n  size, chars:")
    print(f"    min:     {min(sizes)}")
    print(f"    max:     {max(sizes)}")
    print(f"    mean:    {sum(sizes) / len(sizes):.0f}")
    print(f"    median:  {sorted(sizes)[len(sizes) // 2]}")
    print(f"    total:   {sum(sizes):,}")
    print("\n  block types:")
    for block_type, count in sorted(type_counts.items(), key=lambda item: -item[1]):
        print(f"    {block_type:30s} {count}")

    buckets = [0] * 6
    labels = ["<200", "200-500", "500-1000", "1000-1500", "1500-3000", ">3000"]
    for size in sizes:
        if size < 200:
            buckets[0] += 1
        elif size < 500:
            buckets[1] += 1
        elif size < 1000:
            buckets[2] += 1
        elif size < 1500:
            buckets[3] += 1
        elif size < 3000:
            buckets[4] += 1
        else:
            buckets[5] += 1

    print("\n  size distribution:")
    max_bucket = max(buckets) if buckets else 0
    for label, count in zip(labels, buckets):
        bar = "#" * (count * 40 // max_bucket) if max_bucket else ""
        print(f"    {label:>10s}: {count:5d} {bar}")


def export_csv_metadata(chunks: list[Chunk], csv_path: Path) -> None:
    """Export chunk metadata to CSV for inspection."""
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "chunk_id",
                "source_file",
                "page_title",
                "section_title",
                "breadcrumbs",
                "block_types",
                "char_count",
                "token_count_approx",
                "has_table",
                "has_procedure",
                "has_image",
                "chunk_index",
                "text_preview",
            ]
        )
        for chunk in chunks:
            writer.writerow(
                [
                    chunk.chunk_id,
                    chunk.source_file,
                    chunk.page_title,
                    chunk.section_title,
                    chunk.breadcrumbs,
                    ";".join(chunk.block_types),
                    chunk.char_count,
                    chunk.token_count_approx,
                    chunk.has_table,
                    chunk.has_procedure,
                    chunk.has_image,
                    chunk.chunk_index,
                    chunk.text[:150].replace("\n", " "),
                ]
            )
    logger.info("CSV metadata exported to %s", csv_path)


def write_outputs(config: CorpusBuildConfig, chunks: list[Chunk]) -> None:
    """Write configured JSONL/CSV outputs and optional console summaries."""
    if config.preview or (config.single and not config.output_path):
        preview_chunks(chunks, config.preview_count)

    if config.stats or not config.output_path:
        print_stats(chunks)

    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        export_chunks_jsonl(chunks, config.output_path)

    if config.csv_path:
        config.csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_csv_metadata(chunks, config.csv_path)
