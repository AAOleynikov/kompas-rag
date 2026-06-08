"""LLM query rewriting for retrieval."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
Ты помощник по поиску в документации САПР КОМПАС-3D.

Пользователь задал вопрос разговорным языком. Сформулируй 2-3 поисковых
запроса, которые помогут найти ответ в технической документации.

Правила:
1. Каждый запрос должен быть на отдельной строке и начинаться с "- ".
2. Используй техническую терминологию КОМПАС-3D.
3. Запросы должны покрывать разные аспекты вопроса.
4. Не добавляй пояснений, только запросы.
5. Если вопрос уже точный и технический, верни его как есть.

Примеры:

Вопрос: "лагает компьютер в компасе"
- системные требования КОМПАС-3D аппаратное обеспечение
- оптимизация производительности настройки отображения
- работа с большими сборками снижение нагрузки

Вопрос: "как нарисовать дырку в детали"
- создание отверстия в детали
- вырезать выдавливанием эскиз отверстия
- отверстие в листовом теле

Вопрос: "файл не открывается"
- открытие документа КОМПАС ошибка
- форматы файлов совместимость импорт
- восстановление поврежденного файла

Вопрос: "{question}"
"""


def rewrite_query(
    question: str,
    llm: ChatOpenAI,
    max_queries: int = 3,
) -> list[str]:
    """Rewrite a user question into search queries.

    The original question is always kept as the first query.
    """
    try:
        response = llm.invoke(REWRITE_PROMPT.format(question=question))
        raw = response.content.strip()

        if not raw:
            logger.warning("Query rewriter returned an empty response")
            return [question]

        queries = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                query = line[2:].strip()
                if query and len(query) > 5:
                    queries.append(query)

        if not queries:
            logger.warning("Could not parse rewritten queries: %s", raw[:100])
            return [question]

        queries = queries[:max_queries]
        result = [question] + [
            query for query in queries if query.lower() != question.lower()
        ]
        result = result[: max_queries + 1]

        logger.info(
            "Query rewrite: %r -> %s queries: %s",
            question,
            len(result),
            result,
        )
        return result

    except Exception as exc:
        logger.warning("Query rewrite error: %s", exc)
        return [question]
