"""Bounded active conversation window for DUM-E."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from dume.models import Turn


class ActiveWindow:
    """Keep the most recent conversation turns in insertion order."""

    def __init__(self, max_turns: int = 6) -> None:
        """Create an active window with a positive turn capacity."""
        if max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns}")
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._max_turns = max_turns

    @property
    def max_turns(self) -> int:
        """Return the maximum number of active turns."""
        return self._max_turns

    def add(self, turn: Turn) -> Turn | None:
        """Append a turn and return the evicted oldest turn, if any."""
        evicted = self._turns.popleft() if len(self._turns) >= self._max_turns else None
        self._turns.append(turn)
        return evicted

    def turns(self) -> list[Turn]:
        """Return a snapshot of active turns from oldest to newest."""
        return list(self._turns)

    def clear(self) -> None:
        """Remove all active turns."""
        self._turns.clear()

    def __len__(self) -> int:
        """Return the number of active turns."""
        return len(self._turns)

    def __iter__(self) -> Iterator[Turn]:
        """Iterate over active turns from oldest to newest."""
        return iter(self._turns)
