"""Unit tests for DUM-E logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from dume.logging_setup import setup_logging


def test_setup_logging_console_and_file(tmp_path: Path) -> None:
    """Verify setup_logging configures root logger with console and rotating file handlers."""
    log_file = tmp_path / "logs" / "test_dume.log"

    logger = setup_logging(level="DEBUG", log_file=log_file)

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2

    test_message = "Test log message for DUM-E verification."
    logger.info(test_message)

    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_message in content

