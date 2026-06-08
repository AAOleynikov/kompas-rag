#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for corpus chunking.

The corpus preparation orchestration lives in ``corpus.cli`` and
``corpus.pipeline``. The heavy HTML conversion and semantic chunking algorithms
remain unchanged in ``convert_kompas_html_to_md.py`` and ``chunker.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from corpus.cli import main

_PIPELINE_EXPORTS = {
    "CorpusBuildConfig",
    "build_chunks",
    "export_csv_metadata",
    "preview_chunks",
    "print_stats",
    "process_directory",
    "process_single_file",
}

__all__ = ["main", *_PIPELINE_EXPORTS]


def __getattr__(name):
    if name in _PIPELINE_EXPORTS:
        from corpus import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)


if __name__ == "__main__":
    sys.exit(main())
