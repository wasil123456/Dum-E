"""OpenAI-compatible LLM client for DUM-E.

Implements LLMClient against any OpenAI-compatible REST endpoint (including
Google AI Studio's v1beta/openai path) using the installed openai SDK
(v1.x new API: ``openai.OpenAI`` client with ``client.chat.completions.create``).

Retry policy
------------
* Up to ``MAX_RETRIES`` attempts (default 3).
* Retries on HTTP 429 (``openai.RateLimitError``) and timeouts
  (``openai.APITimeoutError``).
* Exponential back-off: ``BASE_DELAY * 2^attempt`` seconds with ±10 % jitter.
* Any other API error is re-raised immediately as ``LLMError``.

Security
--------
The API key is **never** written to log output.  All log statements that
reference connection details use safe fields only (base_url, model, timeout).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import openai

from dume.config import Settings, get_settings
from dume.llm.base import LLMError

logger = logging.getLogger(__name__)

# Retry configuration constants
MAX_RETRIES: int = 3
BASE_DELAY: float = 1.0  # seconds before jitter


class OpenAICompatClient:
    """Synchronous LLM client wrapping the openai v1.x SDK.

    Uses ``openai.OpenAI(base_url=..., api_key=...)`` and calls
    ``client.chat.completions.create`` so it works against Google AI Studio
    and any other OpenAI-compatible endpoint.

    Parameters
    ----------
    settings:
        Application settings.  ``LLM_API_KEY``, ``LLM_BASE_URL``,
        ``LLM_MODEL``, ``LLM_TEMPERATURE``, ``LLM_MAX_TOKENS``, and
        ``LLM_TIMEOUT_SECONDS`` are read from here.
    _openai_client:
        Injected pre-built openai client for unit-testing without network
        access.  When provided, *settings* connection fields are ignored for
        client construction but still used for defaults.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        _openai_client: Any = None,
        _backup_openai_client: Any = None,
    ) -> None:
        """Initialise the client.

        Args:
            settings: Application settings.  Falls back to ``get_settings()``
                when *None*.
            _openai_client: Pre-built openai client for DI in tests (optional).
        """
        self._settings: Settings = settings if settings is not None else get_settings()

        if _openai_client is not None:
            self._client = _openai_client
        elif self._settings.LLM_API_KEY:
            self._client = self._build_client(self._settings.LLM_API_KEY)
        elif self._settings.LLM_BACKUP_API_KEY:
            self._client = self._build_client(self._settings.LLM_BACKUP_API_KEY)
        else:
            raise LLMError(
                "Neither LLM_API_KEY nor LLM_BACKUP_API_KEY is configured."
            )
        if _backup_openai_client is not None:
            self._backup_client = _backup_openai_client
        elif self._settings.LLM_API_KEY and self._settings.LLM_BACKUP_API_KEY:
            self._backup_client = self._build_client(
                self._settings.LLM_BACKUP_API_KEY
            )
        else:
            self._backup_client = None

        logger.debug(
            "OpenAICompatClient initialised: base_url=%s model=%s timeout=%.1fs",
            self._settings.LLM_BASE_URL,
            self._settings.LLM_MODEL,
            self._settings.LLM_TIMEOUT_SECONDS,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a chat-completion request; retry on rate-limit / timeout.

        Args:
            messages: Chat history in OpenAI format
                (list of ``{"role": ..., "content": ...}`` dicts).
            **kwargs: Override ``temperature``, ``max_tokens``, or other
                provider-specific parameters.

        Returns:
            Stripped assistant reply text.

        Raises:
            LLMError: After ``MAX_RETRIES`` failed attempts (rate-limit or
                timeout), or immediately on any other API error.
        """
        model: str = kwargs.pop("model", self._settings.LLM_MODEL)
        temperature: float = float(
            kwargs.pop("temperature", self._settings.LLM_TEMPERATURE)
        )
        max_tokens: int = int(
            kwargs.pop("max_tokens", self._settings.LLM_MAX_TOKENS)
        )

        try:
            return self._generate_with_retries(
                self._client,
                messages,
                model,
                temperature,
                max_tokens,
                kwargs,
            )
        except LLMError as primary_exc:
            if self._backup_client is None or not self._should_failover(primary_exc):
                raise

            logger.warning(
                "Primary LLM key failed with a failover-eligible error; "
                "retrying once with the backup key."
            )
            try:
                return self._generate_once(
                    self._backup_client,
                    messages,
                    model,
                    temperature,
                    max_tokens,
                    kwargs,
                )
            except Exception as backup_exc:
                raise LLMError(
                    "Both primary and backup LLM API keys failed. "
                    f"Backup error: {backup_exc}",
                    cause=backup_exc,
                ) from backup_exc

    def _generate_with_retries(
        self,
        client: Any,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        kwargs: dict[str, Any],
    ) -> str:
        """Call one configured key using the existing retry policy."""
        last_exc: BaseException | None = None

        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_once(
                    client,
                    messages,
                    model,
                    temperature,
                    max_tokens,
                    kwargs,
                )
            except openai.RateLimitError as exc:
                last_exc = exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.2fs.",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

            except openai.APITimeoutError as exc:
                last_exc = exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Request timeout (attempt %d/%d). Retrying in %.2fs.",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

            except openai.OpenAIError as exc:
                # Non-retriable API error — fail immediately.
                raise LLMError(
                    f"LLM API error on attempt {attempt + 1}: {exc}",
                    cause=exc,
                ) from exc

            except Exception as exc:  # noqa: BLE001
                # Unexpected error (network, JSON parse, etc.) — fail immediately.
                raise LLMError(
                    f"Unexpected error calling LLM on attempt {attempt + 1}: {exc}",
                    cause=exc,
                ) from exc

        raise LLMError(
            f"LLM call failed after {MAX_RETRIES} retries. "
            f"Last error: {last_exc}",
            cause=last_exc,
        )

    def _generate_once(
        self,
        client: Any,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        kwargs: dict[str, Any],
    ) -> str:
        """Execute one request against a specific key."""
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        reply: str = response.choices[0].message.content.strip()
        logger.debug("LLM response received: reply_len=%d", len(reply))
        return reply

    def _build_client(self, api_key: str) -> Any:
        """Build a client without exposing the key in logs."""
        return openai.OpenAI(
            api_key=api_key,
            base_url=self._settings.LLM_BASE_URL,
            timeout=self._settings.LLM_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _should_failover(error: LLMError) -> bool:
        """Return whether an LLM failure likely represents a key-specific outage."""
        cause: BaseException | None = error.__cause__ or error
        status_code = getattr(cause, "status_code", None)
        if status_code in {401, 403, 408, 429}:
            return True
        text = str(cause).casefold()
        return any(
            marker in text
            for marker in (
                "quota",
                "rate limit",
                "too many requests",
                "authentication",
                "unauthorized",
                "permission",
                "timeout",
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Compute exponential back-off with ±10 % jitter.

        Args:
            attempt: Zero-based attempt index.

        Returns:
            Sleep duration in seconds.
        """
        base = BASE_DELAY * (2 ** attempt)
        jitter = base * 0.1 * (2 * random.random() - 1)
        return max(0.0, base + jitter)
