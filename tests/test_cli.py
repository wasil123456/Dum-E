"""Unit tests for the Rich DUM-E CLI loop."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from dume.cli import run_chat
from dume.models import Answer


class FakeEngine:
    def __init__(self) -> None:
        self.questions: list[tuple[str, str]] = []
        self.reset_calls: list[str] = []
        self.memory = {"active_turns": [], "long_term_summary": ""}

    def ask(self, session_id: str, question: str) -> Answer:
        self.questions.append((session_id, question))
        return Answer(text="Grounded **answer**.", sources=[])

    def reset_session(self, session_id: str) -> None:
        self.reset_calls.append(session_id)

    def get_memory_view(self, session_id: str) -> dict[str, object]:
        return self.memory

    def stats(self) -> dict[str, int]:
        return {"asks": len(self.questions)}


def test_cli_commands_and_question_flow() -> None:
    engine = FakeEngine()
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    inputs = iter(["/help", "What is DUM-E?", "/memory", "/stats", "/reset", "/exit"])
    console.input = lambda *_args, **_kwargs: next(inputs)  # type: ignore[method-assign]

    run_chat(engine, session_id="test-session", console=console)

    rendered = output.getvalue()
    assert "available commands" in rendered
    assert "Grounded" in rendered
    assert engine.questions == [("test-session", "What is DUM-E?")]
    assert engine.reset_calls == ["test-session"]
