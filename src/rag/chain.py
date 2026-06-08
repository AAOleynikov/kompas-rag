"""Backward-compatible generation facade.

The implementation lives in ``src.rag.generation``. This module keeps existing
imports such as ``from .chain import KompasRAGChain`` working.
"""

from .generation.chain import KompasRAGChain
from .generation.llm import create_llm
from .generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .generation.references import (
    DOCS_BASE_URL,
    SourceReference,
    _strip_sources_section,
    build_source_url,
    postprocess_references,
    preprocess_context,
)

__all__ = [
    "DOCS_BASE_URL",
    "KompasRAGChain",
    "SYSTEM_PROMPT",
    "SourceReference",
    "USER_PROMPT_TEMPLATE",
    "_strip_sources_section",
    "build_source_url",
    "create_llm",
    "postprocess_references",
    "preprocess_context",
]
