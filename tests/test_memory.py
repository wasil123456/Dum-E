"""Unit tests for DUM-E conversation memory."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dume.memory import ActiveWindow, LongTermBuffer, MemoryManager
from dume.models import Turn


def _turn(index: int) -> Turn:
    return Turn(query=f"question {index}", answer=f"answer {index}")


def test_active_window_evicts_oldest_turn() -> None:
    window = ActiveWindow(max_turns=2)

    assert window.add(_turn(1)) is None
    assert window.add(_turn(2)) is None
    evicted = window.add(_turn(3))

    assert evicted == _turn(1)
    assert [turn.query for turn in window.turns()] == ["question 2", "question 3"]


def test_active_window_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="max_turns must be positive"):
        ActiveWindow(max_turns=0)


def test_long_term_buffer_filters_duplicates_and_caps_size() -> None:
    buffer = LongTermBuffer(max_summaries=2)

    assert buffer.add(" First summary. ") is True
    assert buffer.add("first   summary.") is False
    assert buffer.add("Second summary.") is True
    assert buffer.add("Third summary.") is True

    assert buffer.summaries() == ["Second summary.", "Third summary."]


@dataclass
class FakeSummarizer:
    calls: list[Turn]

    def summarize(self, turn: Turn) -> str:
        self.calls.append(turn)
        return f"summary of {turn.query}"


def test_memory_manager_summarizes_evicted_turns() -> None:
    summarizer = FakeSummarizer(calls=[])
    manager = MemoryManager(
        active_window=ActiveWindow(max_turns=2),
        summarizer=summarizer,  # type: ignore[arg-type]
        long_term_buffer=LongTermBuffer(max_summaries=2),
    )

    manager.add_turn(_turn(1))
    manager.add_turn(_turn(2))
    evicted = manager.add_turn(_turn(3))

    assert evicted is not None
    assert [turn.query for turn in manager.active_turns()] == [
        "question 2",
        "question 3",
    ]
    assert [turn.query for turn in summarizer.calls] == ["question 1"]
    assert manager.long_term_summary() == "summary of question 1"
    assert evicted.summary == "summary of question 1"


def test_memory_manager_add_creates_turn_and_deduplicates_summaries() -> None:
    summarizer = FakeSummarizer(calls=[])
    manager = MemoryManager(
        active_window=ActiveWindow(max_turns=1),
        summarizer=summarizer,  # type: ignore[arg-type]
    )

    manager.add("same question", "first answer")
    manager.add("same question", "second answer")
    manager.add("same question", "third answer")

    assert len(manager.active_turns()) == 1
    assert len(manager.long_term_buffer) == 1
