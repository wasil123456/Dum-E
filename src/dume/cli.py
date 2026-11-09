"""Rich terminal interface for DUM-E."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from dume.engine import DumeEngine
from dume.logging_setup import setup_logging
from dume.models import Answer

logger = logging.getLogger(__name__)

_HELP_TEXT = (
    "[bold]/help[/]  Show available commands\n"
    "[bold]/memory[/] Show the active window and long-term summary\n"
    "[bold]/reset[/]  Clear this conversation session\n"
    "[bold]/stats[/]  Show engine counters\n"
    "[bold]/exit[/]   Exit DUM-E\n"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Chat with the DUM-E RAG assistant.")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier to use (defaults to a generated local session).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _render_welcome(console: Console, session_id: str) -> None:
    console.print(
        Panel(
            "[bold cyan]DUM-E[/bold cyan] [dim]document-grounded assistant[/dim]\n"
            "Ask a question about your indexed documents, or type [bold]/help[/].",
            title="Welcome",
            border_style="cyan",
        )
    )
    console.print(f"[dim]Session: {session_id}[/dim]")


def _render_answer(console: Console, answer: Answer, elapsed_ms: float) -> None:
    cache_label = "[green]cache hit[/green]" if answer.cached else "[blue]generated[/blue]"
    source_count = len(answer.sources)
    metadata = (
        f"{cache_label}  ·  [dim]{elapsed_ms:.0f} ms  ·  "
        f"{source_count} source{'s' if source_count != 1 else ''}[/dim]"
    )
    console.print(
        Panel(
            Markdown(answer.text),
            title="DUM-E",
            subtitle=metadata,
            border_style="green" if answer.cached else "blue",
            padding=(1, 2),
        )
    )


def _render_memory(console: Console, view: dict[str, Any]) -> None:
    turns = view.get("active_turns", [])
    summary = str(view.get("long_term_summary", "")).strip()
    table = Table(title="Conversation memory", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Question")
    table.add_column("Answer")
    for index, turn in enumerate(turns, start=1):
        table.add_row(str(index), turn.query, turn.answer)
    console.print(table)
    console.print(
        Panel(
            summary or "[dim]No long-term summaries yet.[/dim]",
            title="Long-term summary",
            border_style="magenta",
        )
    )


def _render_stats(console: Console, stats: dict[str, int]) -> None:
    table = Table(title="Engine statistics")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for name, value in sorted(stats.items()):
        table.add_row(name.replace("_", " ").title(), str(value))
    console.print(table)


def run_chat(
    engine: DumeEngine,
    *,
    session_id: str,
    console: Console | None = None,
) -> None:
    """Run the interactive chat loop using an already-created engine."""
    active_console = console or Console()
    _render_welcome(active_console, session_id)

    while True:
        try:
            question = active_console.input("[bold cyan]You[/bold cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            active_console.print("\n[dim]Goodbye.[/dim]")
            return

        if not question:
            continue
        command = question.casefold()
        if command in {"/exit", "/quit"}:
            active_console.print("[dim]Goodbye.[/dim]")
            return
        if command == "/help":
            active_console.print(_HELP_TEXT)
            continue
        if command == "/reset":
            engine.reset_session(session_id)
            active_console.print("[green]Conversation memory reset.[/green]")
            continue
        if command == "/memory":
            _render_memory(active_console, engine.get_memory_view(session_id))
            continue
        if command == "/stats":
            _render_stats(active_console, engine.stats())
            continue
        if question.startswith("/"):
            active_console.print(
                "[yellow]Unknown command. Type /help to see available commands.[/yellow]"
            )
            continue

        started = time.perf_counter()
        try:
            answer = engine.ask(session_id, question)
        except Exception as exc:
            logger.exception("Question handling failed for session %s", session_id)
            active_console.print(
                Panel(
                    f"[red]Unable to answer this question:[/red] {exc}",
                    title="DUM-E error",
                    border_style="red",
                )
            )
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        _render_answer(active_console, answer, elapsed_ms)


def main(argv: list[str] | None = None) -> int:
    """Initialize logging and the lazy-loaded real engine, then start chat."""
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)
    session_id = args.session_id or f"cli-{uuid4().hex[:8]}"
    run_chat(DumeEngine(), session_id=session_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
