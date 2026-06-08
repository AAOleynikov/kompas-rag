"""Source numbering and reference post-processing."""

import re
from dataclasses import dataclass

from langchain_core.documents import Document

DOCS_BASE_URL = "https://help.ascon.ru/KOMPAS/24/ru-RU"


@dataclass
class SourceReference:
    """Reference to a documentation chunk shown to the LLM."""

    number: int
    source_file: str
    page_title: str
    section_path: str
    breadcrumbs: str
    url: str


def build_source_url(source_file: str, base_url: str = DOCS_BASE_URL) -> str:
    if not source_file:
        return ""
    return f"{base_url}/{source_file}"


def preprocess_context(
    documents: list[Document],
    base_url: str = DOCS_BASE_URL,
) -> tuple[str, list[SourceReference]]:
    """Build numbered context blocks and source references."""
    parts: list[str] = []
    refs: list[SourceReference] = []

    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        source_file = meta.get("source_file", "")
        page_title = meta.get("page_title", "")
        section_path = meta.get("section_path", "") or page_title
        breadcrumbs = meta.get("breadcrumbs", "")
        url = build_source_url(source_file, base_url)

        refs.append(
            SourceReference(
                number=i,
                source_file=source_file,
                page_title=page_title,
                section_path=section_path,
                breadcrumbs=breadcrumbs,
                url=url,
            )
        )

        header = f"[Источник {i}] {section_path}"
        if breadcrumbs and breadcrumbs != section_path:
            header += f"\n  Раздел: {breadcrumbs}"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts), refs


def _strip_sources_section(text: str) -> tuple[str, set[int]]:
    """Strip an LLM-generated sources section from the end of an answer."""
    patterns = [
        r"\n---\s*\n\s*\*{0,2}Источники?\*{0,2}",
        r"\n#{1,4}\s*Источники?",
        r"\n\*{2}Источники?\*{0,2}\s*:",
        r"\nИсточники?\s*:",
    ]

    earliest_pos = len(text)
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos < len(text):
        main_part = text[:earliest_pos]
        sources_part = text[earliest_pos:]
        nums = {int(match.group(1)) for match in re.finditer(r"\[(\d{1,2})\]", sources_part)}
        return main_part, nums

    return text, set()


def postprocess_references(
    answer: str,
    refs: list[SourceReference],
    link_style: str = "markdown",
) -> str:
    """Replace local [N] references with links and append a single sources block."""
    ref_map = {ref.number: ref for ref in refs}
    used_numbers: set[int] = set()

    main_part, nums_from_sources = _strip_sources_section(answer)
    used_numbers.update(nums_from_sources)

    def replace_ref(match: re.Match) -> str:
        num = int(match.group(1))
        if num not in ref_map:
            return match.group(0)

        used_numbers.add(num)
        ref = ref_map[num]
        if link_style == "markdown" and ref.url:
            return f"[[{num}]]({ref.url})"
        return f"[{num}: {ref.page_title}]"

    main_part = re.sub(r"(?<!\[)\[(\d{1,2})\](?!\]|\()", replace_ref, main_part)

    if not used_numbers:
        return main_part.strip()

    sources_block = "\n\n**Источники:**\n"
    for num in sorted(used_numbers):
        if num not in ref_map:
            continue
        ref = ref_map[num]
        title = ref.section_path or ref.page_title
        if ref.url:
            sources_block += f"- [{num}] [{title}]({ref.url})\n"
        else:
            sources_block += f"- [{num}] {title}\n"

    return (main_part.rstrip() + sources_block).strip()
