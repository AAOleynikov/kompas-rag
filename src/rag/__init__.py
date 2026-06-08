"""RAG system for KOMPAS-3D documentation."""

from .config import RAGConfig

__all__ = ["RAGConfig", "KompasRAGChain", "KompasRetriever"]


def __getattr__(name: str):
    if name == "KompasRAGChain":
        from .generation.chain import KompasRAGChain

        return KompasRAGChain
    if name == "KompasRetriever":
        from .retrieval.hybrid import KompasRetriever

        return KompasRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
