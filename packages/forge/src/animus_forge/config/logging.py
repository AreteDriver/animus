"""Logging configuration with JSON format support."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from animus_forge.utils.validation import sanitize_log_message


class SanitizingFilter(logging.Filter):
    """Filter that sanitizes sensitive data from log messages.

    Removes API keys, tokens, and other secrets from log output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize the log message."""
        if record.msg:
            record.msg = sanitize_log_message(str(record.msg))
        if record.args:
            # Sanitize any string arguments
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    sanitized_args.append(sanitize_log_message(arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    # Optional fields to extract from log records
    _OPTIONAL_FIELDS = (
        "trace_id",
        "span_id",
        "parent_span_id",
        "request_id",
        "method",
        "path",
        "http.method",
        "http.path",
        "http.status_code",
        "status_code",
        "duration_ms",
        "client_ip",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add optional fields if present
        for field in self._OPTIONAL_FIELDS:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Standard text formatter with consistent format."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(
    level: str = "INFO", format: str = "text", sanitize_logs: bool = True
) -> None:
    """Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format ('text' or 'json')
        sanitize_logs: If True, redact sensitive data (API keys, tokens) from logs
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Set formatter based on format
    if format.lower() == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())

    # Add sanitization filter if enabled
    if sanitize_logs:
        console_handler.addFilter(SanitizingFilter())

    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    # Configure tool audit logger — writes JSONL to logs/forge_audit.jsonl
    _configure_tool_audit_logger()


def _configure_tool_audit_logger() -> None:
    """Set up the forge.tool_audit logger with a JSONL file handler.

    Writes one JSON object per line to logs/forge_audit.jsonl.
    The logger is independent from the root logger — audit entries
    do not appear in console output.
    """
    from pathlib import Path

    audit_logger = logging.getLogger("forge.tool_audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't echo to console

    # Skip if already configured (e.g. multiple configure_logging calls)
    if audit_logger.handlers:
        return

    try:
        from animus_forge.config.settings import get_settings

        logs_dir = get_settings().logs_dir
    except Exception:
        logs_dir = Path("logs")

    logs_dir.mkdir(parents=True, exist_ok=True)
    audit_path = logs_dir / "forge_audit.jsonl"

    file_handler = logging.FileHandler(str(audit_path), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    # Raw passthrough — the audit logger already gets pre-formatted JSON
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(file_handler)
