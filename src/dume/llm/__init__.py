"""LLM integrations for DUM-E.

Public surface:

- ``LLMClient``  — Protocol (runtime-checkable) for all LLM clients.
- ``LLMError``   — Exception raised when LLM calls fail.
- ``OpenAICompatClient`` — Concrete client for OpenAI-compatible endpoints.
- ``build_llm``  — Factory returning the configured ``LLMClient``.
"""

from dume.llm.base import LLMClient, LLMError
from dume.llm.factory import build_llm
from dume.llm.openai_compat import OpenAICompatClient

__all__ = [
    "LLMClient",
    "LLMError",
    "OpenAICompatClient",
    "build_llm",
]
