"""Prompt construction for DUM-E.

Builds the ``messages`` list (OpenAI chat format) that is passed verbatim
to any ``LLMClient.generate()`` call.  The logic is deterministic and
pure: no I/O, no model calls — fully testable via dependency injection.

Message layout
--------------
1. **System** – grounding rules, long-term summary (if any).
2. **Active window turns** – prior ``Turn`` objects as user/assistant pairs.
3. **Context** – numbered source blocks formatted within a character budget.
4. **User** – the current query.
"""

from __future__ import annotations

import logging
from typing import Sequence

from dume.models import RetrievedChunk, Turn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION: str = (
    "You are DUM-E, a document-grounded assistant.  "
    "Answer ONLY from the provided context blocks below.  "
    "Do NOT use any outside knowledge or make up information.  "
    "If the answer cannot be found in the context, say "
    "\"I could not find this information in the provided documents.\"  "
    "Always cite the source number (e.g. [1], [2]) when you use information "
    "from a context block."
)

_DEFAULT_MAX_CONTEXT_CHARS: int = 4_000
_CONTEXT_HEADER: str = "\n\n--- CONTEXT ---\n"
_CONTEXT_FOOTER: str = "\n--- END CONTEXT ---"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_messages(
    query: str,
    context_chunks: Sequence[RetrievedChunk],
    active_turns: Sequence[Turn],
    long_term_summary: str | None = None,
    *,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> list[dict[str, str]]:
    """Assemble the OpenAI-format message list for a single RAG turn.

    Args:
        query: The current user question.
        context_chunks: Retrieved and reranked chunks to use as grounding.
            Chunks are numbered from 1 in the order supplied.
        active_turns: Prior conversation turns in chronological order.
            Each becomes a ``user`` / ``assistant`` message pair.
        long_term_summary: Optional abstractive summary of older history
            beyond the active window.  Included in the system message when
            provided.
        max_context_chars: Character budget for the concatenated context
            block (default 4 000).  Chunks that would exceed the budget
            are truncated (not dropped) so the budget is always respected.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts ready for
        ``LLMClient.generate()``.
    """
    messages: list[dict[str, str]] = []

    # 1. System message ------------------------------------------------
    system_content = _build_system_content(long_term_summary)
    messages.append({"role": "system", "content": system_content})
    logger.debug("System message length: %d chars", len(system_content))

    # 2. Active window turns -------------------------------------------
    for turn in active_turns:
        messages.append({"role": "user", "content": turn.query})
        messages.append({"role": "assistant", "content": turn.answer})
    logger.debug("Injected %d active turns (%d messages)", len(active_turns), len(active_turns) * 2)

    # 3. Context block -------------------------------------------------
    context_text = _build_context_block(context_chunks, max_context_chars)
    if context_text:
        # Attach to the user turn that contains the query so the model
        # sees query + context together.
        final_user_content = f"{context_text}\n\n{query}"
    else:
        final_user_content = query
        logger.debug("No context chunks; query sent without context block.")

    # 4. Final user message --------------------------------------------
    messages.append({"role": "user", "content": final_user_content})
    logger.debug(
        "Final user message length: %d chars", len(final_user_content)
    )

    return messages


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_system_content(long_term_summary: str | None) -> str:
    """Compose the system message text.

    Args:
        long_term_summary: Abstractive summary of older turns (may be None).

    Returns:
        Full system message string.
    """
    parts: list[str] = [_SYSTEM_INSTRUCTION]

    if long_term_summary and long_term_summary.strip():
        parts.append(
            "\n\nBackground (summary of earlier conversation):\n"
            + long_term_summary.strip()
        )

    return "".join(parts)


def _build_context_block(
    chunks: Sequence[RetrievedChunk],
    max_chars: int,
) -> str:
    """Format retrieved chunks into a numbered context block within a budget.

    Chunks are iterated in supplied order.  Each chunk is formatted as::

        [N] Source: <source>
        <text>

    If appending the full text of a chunk would exceed *max_chars*, the
    chunk text is **truncated** to fit rather than dropped entirely —
    ensuring the budget is always respected and the LLM still sees as
    much context as possible.

    Args:
        chunks: Sequence of retrieved chunks (pre-ranked).
        max_chars: Maximum total character count for the joined context.

    Returns:
        Formatted context string, or empty string when *chunks* is empty.
    """
    if not chunks:
        return ""

    sections: list[str] = []
    used: int = len(_CONTEXT_HEADER) + len(_CONTEXT_FOOTER)

    for idx, rc in enumerate(chunks, start=1):
        header = f"[{idx}] Source: {rc.source}\n"
        separator = "\n\n" if sections else ""
        prefix = separator + header

        budget_remaining = max_chars - used - len(prefix)
        if budget_remaining <= 0:
            logger.debug(
                "Context budget exhausted after %d/%d chunks.", idx - 1, len(chunks)
            )
            break

        text = rc.text
        if len(text) > budget_remaining:
            text = text[:budget_remaining].rstrip()
            logger.debug(
                "Chunk [%d] truncated to %d chars to stay within budget.",
                idx,
                len(text),
            )

        section = prefix + text
        sections.append(section)
        used += len(section)

    if not sections:
        return ""

    return _CONTEXT_HEADER + "".join(sections) + _CONTEXT_FOOTER
