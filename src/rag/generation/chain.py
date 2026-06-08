"""RAG answer generation chain."""

import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import RAGConfig
from ..retrieval.hybrid import KompasRetriever
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .references import DOCS_BASE_URL, postprocess_references, preprocess_context

logger = logging.getLogger(__name__)


class KompasRAGChain:
    """Orchestrates retrieval, prompt construction, LLM call and references."""

    def __init__(
        self,
        retriever: KompasRetriever,
        llm: ChatOpenAI,
        config: Optional[RAGConfig] = None,
        docs_base_url: str = DOCS_BASE_URL,
    ):
        self.retriever = retriever
        self.llm = llm
        self.config = config or RAGConfig()
        self.docs_base_url = docs_base_url

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT_TEMPLATE),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(
        self,
        question: str,
        return_sources: bool = False,
        link_style: str = "markdown",
    ) -> dict:
        docs = self.retriever.retrieve(question)

        if not docs:
            return {
                "answer": (
                    "К сожалению, я не нашёл информации по этому вопросу "
                    "в документации КОМПАС-3D. Попробуйте переформулировать "
                    "вопрос или уточнить тему."
                ),
                "answer_raw": "",
                "sources": [],
                "context": "",
            }

        context, refs = preprocess_context(docs, self.docs_base_url)
        answer_raw = self.chain.invoke({"context": context, "question": question})
        answer = postprocess_references(answer_raw, refs, link_style)

        sources = []
        if return_sources:
            sources = [
                {
                    "number": ref.number,
                    "page_title": ref.page_title,
                    "section_path": ref.section_path,
                    "source_file": ref.source_file,
                    "url": ref.url,
                    "breadcrumbs": ref.breadcrumbs,
                }
                for ref in refs
            ]

        logger.info(
            "RAG: question='%s...', documents=%s, refs=%s, answer=%s chars",
            question[:50],
            len(docs),
            len(refs),
            len(answer),
        )

        return {
            "answer": answer,
            "answer_raw": answer_raw,
            "sources": sources,
            "context": context if self.config.debug else "",
        }

    def ask_simple(self, question: str) -> str:
        return self.ask(question)["answer"]
