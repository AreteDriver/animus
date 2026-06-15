"""
Animus Logging Configuration

Provides structured logging with file rotation and console output.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_file: Path | None = None,
    level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure logging for Animus.

    Args:
        log_file: Path to log file. If None, file logging disabled.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to write logs to file

    Returns:
        Root logger for animus
    """
    logger = logging.getLogger("animus")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Format for log messages
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler (rotating, 10MB max, keep 3 backups)
    if log_to_file and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Capture everything to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler (warnings and above only - don't pollute CLI)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"animus.{name}")
