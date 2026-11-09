"""Logging configuration for DUM-E.

Provides console and rotating file handlers with structured formatting.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

from dume.config import PROJECT_ROOT


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | str | None = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure application-wide logging with console and rotating file outputs.

    Args:
        level: Logging level (e.g. 'DEBUG', 'INFO', 'WARNING', logging.INFO).
        log_file: Destination path for rotating log file. Defaults to PROJECT_ROOT / 'logs' / 'dume.log'.
        max_bytes: Maximum size in bytes before rotating log file (default: 5MB).
        backup_count: Number of rotated backup files to retain.

    Returns:
        logging.Logger: The configured root logger.
    """
    resolved_path: Path
    if log_file is not None:
        resolved_path = Path(log_file)
    else:
        resolved_path = PROJECT_ROOT / "logs" / "dume.log"

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level: int
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplication on repeated invocations
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=str(resolved_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return root_logger
