"""Coordinate active and long-term conversation memory."""

from __future__ import annotations

from dume.config import Settings, get_settings
from dume.memory.active_window import ActiveWindow
from dume.memory.long_term import LongTermBuffer
from dume.memory.summarizer import SumySummarizer
from dume.models import Turn


class MemoryManager:
    """Move evicted active turns into bounded local long-term summaries."""

    def __init__(
        self,
        active_window: ActiveWindow | None = None,
        summarizer: SumySummarizer | None = None,
        long_term_buffer: LongTermBuffer | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Create memory components, allowing fakes for isolated tests."""
        config = settings or get_settings()
        self.active_window = (
            active_window
            if active_window is not None
            else ActiveWindow(config.MAX_ACTIVE_TURNS)
        )
        self.summarizer = (
            summarizer
            if summarizer is not None
            else SumySummarizer(config.SUMMARY_SENTENCES)
        )
        self.long_term_buffer = (
            long_term_buffer
            if long_term_buffer is not None
            else LongTermBuffer()
        )

    def add_turn(self, turn: Turn) -> Turn | None:
        """Add a turn and summarize the evicted turn when the window overflows."""
        evicted = self.active_window.add(turn)
        if evicted is not None:
            summary = self.summarizer.summarize(evicted)
            if summary:
                evicted.summary = summary
                self.long_term_buffer.add(summary)
        return evicted

    def add(self, query: str, answer: str) -> Turn | None:
        """Create and add a turn from query and answer text."""
        return self.add_turn(Turn(query=query, answer=answer))

    def active_turns(self) -> list[Turn]:
        """Return the current active turns."""
        return self.active_window.turns()

    def long_term_summary(self) -> str:
        """Return the combined long-term summary text."""
        return self.long_term_buffer.combined()

    def clear(self) -> None:
        """Clear both active and long-term memory."""
        self.active_window.clear()
        self.long_term_buffer.clear()
