"""LLM client interfaces for DUM-E.

Defines the LLMClient Protocol and LLMError exception so every component
that generates text talks to the same abstract surface.  Concrete
implementations live in sibling modules; nothing here imports a networking
library so the interface stays testable without network access.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Raised when an LLM call fails after all retries are exhausted.

    Wraps the underlying cause so callers can inspect it via __cause__.

    Attributes:
        message: Human-readable description of the failure.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        """Initialise LLMError.

        Args:
            message: Description of what went wrong.
            cause: Original exception that triggered this error (optional).
        """
        super().__init__(message)
        self.message = message
        if cause is not None:
            self.__cause__ = cause


@runtime_checkable
class LLMClient(Protocol):
    """Protocol satisfied by any synchronous LLM client used in DUM-E.

    All concrete clients (OpenAI-compat, stubs, fakes) must implement
    ``generate``.  The Protocol is decorated with ``@runtime_checkable``
    so ``isinstance`` checks work in tests.
    """

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a chat-completion request and return the assistant reply.

        Args:
            messages: List of role/content dicts in OpenAI chat format,
                e.g. ``[{"role": "system", "content": "..."}, ...]``.
            **kwargs: Provider-specific overrides (e.g. temperature, max_tokens).

        Returns:
            The assistant's reply text with leading/trailing whitespace stripped.

        Raises:
            LLMError: If the call fails after retries are exhausted.
        """
        ...
