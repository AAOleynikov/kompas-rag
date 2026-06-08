"""Corpus preparation pipeline for KOMPAS-3D documentation."""

__all__ = [
    "CorpusBuildConfig",
    "build_chunks",
    "export_csv_metadata",
    "print_stats",
    "process_directory",
    "process_single_file",
]


def __getattr__(name):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
