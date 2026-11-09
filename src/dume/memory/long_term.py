"""Bounded long-term conversation memory for DUM-E."""

from __future__ import annotations

from collections import deque


class LongTermBuffer:
    """Store unique summaries while retaining only the newest entries."""

    def __init__(self, max_summaries: int = 24) -> None:
        """Create a buffer with a positive summary capacity."""
        if max_summaries <= 0:
            raise ValueError(
                f"max_summaries must be positive, got {max_summaries}"
            )
        self._summaries: deque[str] = deque(maxlen=max_summaries)
        self._max_summaries = max_summaries
        self._keys: set[str] = set()

    @property
    def max_summaries(self) -> int:
        """Return the maximum number of stored summaries."""
        return self._max_summaries

    def add(self, summary: str) -> bool:
        """Add a non-empty, non-duplicate summary and return whether it was added."""
        cleaned = " ".join(summary.split())
        if not cleaned:
            return False

        key = cleaned.casefold()
        if key in self._keys:
            return False

        if len(self._summaries) >= self._max_summaries:
            evicted = self._summaries.popleft()
            self._keys.remove(" ".join(evicted.split()).casefold())
        self._summaries.append(cleaned)
        self._keys.add(key)
        return True

    def summaries(self) -> list[str]:
        """Return a snapshot of summaries from oldest to newest."""
        return list(self._summaries)

    def combined(self, separator: str = "\n") -> str:
        """Return summaries joined for inclusion in a prompt."""
        return separator.join(self._summaries)

    def clear(self) -> None:
        """Remove all stored summaries."""
        self._summaries.clear()
        self._keys.clear()

    def __len__(self) -> int:
        """Return the number of stored summaries."""
        return len(self._summaries)
