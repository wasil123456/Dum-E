"""LLM client factory for DUM-E.

Provides a single ``build_llm()`` entry-point that reads the configured
provider from Settings and returns the appropriate LLMClient implementation.
Currently only the ``"google"`` / OpenAI-compat path is supported, but the
factory is structured to accommodate additional providers without changing
call sites.
"""

from __future__ import annotations

import logging

from dume.config import Settings, get_settings
from dume.llm.base import LLMClient, LLMError
from dume.llm.openai_compat import OpenAICompatClient

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"google", "openai"})


def build_llm(settings: Settings | None = None) -> LLMClient:
    """Construct and return a configured LLM client.

    Reads ``LLM_PROVIDER`` from *settings* (or the global singleton when
    *settings* is ``None``) and instantiates the matching implementation.

    Args:
        settings: Application settings.  Uses ``get_settings()`` when ``None``.

    Returns:
        A ready-to-use ``LLMClient`` implementation.

    Raises:
        LLMError: If ``LLM_PROVIDER`` names an unsupported provider.
        LLMError: If ``LLM_API_KEY`` is empty (key not configured).
    """
    cfg: Settings = settings if settings is not None else get_settings()

    provider = cfg.LLM_PROVIDER.lower().strip()
    logger.info(
        "Building LLM client: provider=%s model=%s base_url=%s",
        provider,
        cfg.LLM_MODEL,
        cfg.LLM_BASE_URL,
    )

    if not cfg.LLM_API_KEY and not cfg.LLM_BACKUP_API_KEY:
        raise LLMError(
            "Neither LLM_API_KEY nor LLM_BACKUP_API_KEY is set. "
            "Add at least one key to your .env file before calling build_llm()."
        )

    if provider not in _SUPPORTED_PROVIDERS:
        raise LLMError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported providers: {sorted(_SUPPORTED_PROVIDERS)}"
        )

    # Both "google" and "openai" use the same OpenAI-compat client;
    # only the base_url/api_key differ (loaded from settings).
    client = OpenAICompatClient(settings=cfg)
    logger.debug("LLM client ready: %s", type(client).__name__)
    return client
