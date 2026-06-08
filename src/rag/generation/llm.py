"""LLM client factory."""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from ..config import LLMConfig

logger = logging.getLogger(__name__)


def create_llm(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    """Create an OpenAI-compatible chat model client."""
    if config is None:
        config = LLMConfig()

    extra_body = {}
    if config.disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    kwargs = {
        "base_url": config.base_url,
        "model": config.model_name,
        "api_key": config.api_key,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if extra_body:
        kwargs["extra_body"] = extra_body

    llm = ChatOpenAI(**kwargs)

    logger.info(
        "LLM: %s @ %s (thinking=%s, max_tokens=%s)",
        config.model_name,
        config.base_url,
        "OFF" if config.disable_thinking else "ON",
        config.max_tokens,
    )
    return llm
