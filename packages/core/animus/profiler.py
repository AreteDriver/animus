"""Lightweight performance profiler for Animus.

Zero external dependencies. Logs structured JSON to a rotating log file.

Usage:
    from animus.profiler import perf_log

    with perf_log("model_generate", provider="ollama"):
        response = model.generate(prompt, system)

    with perf_log("tool_execute", tool_name="read_file") as ctx:
        result = tools.execute("read_file", {"path": "/etc/hostname"})
        ctx["success"] = result.success
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("profiler")

# Rotating log configuration
LOG_DIR = Path.home() / ".animus" / "logs"
LOG_FILE = LOG_DIR / "performance.log"
MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_LOG_AGE_DAYS = 7

# In-memory buffer for real-time stats (last N entries)
_MAX_BUFFER = 1000
_perf_buffer: list[dict[str, Any]] = []
_perf_lock = threading.Lock()


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_if_needed() -> None:
    """Rotate log file if it exceeds max size."""
    if not LOG_FILE.exists():
        return
    try:
        if LOG_FILE.stat().st_size < MAX_LOG_SIZE_BYTES:
            return
        # Simple rotation: rename existing to .1
        rotated = LOG_FILE.with_suffix(".log.1")
        if rotated.exists():
            rotated.unlink()
        LOG_FILE.rename(rotated)
    except OSError as e:
        logger.warning(f"Failed to rotate performance log: {e}")


def _cleanup_old_logs() -> None:
    """Remove log files older than MAX_LOG_AGE_DAYS."""
    if not LOG_DIR.exists():
        return
    cutoff = time.time() - (MAX_LOG_AGE_DAYS * 86400)
    for path in LOG_DIR.glob("performance.log*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError as e:
            logger.debug(f"Failed to cleanup old performance log '{path}': {e}")


def log_event(
    phase: str,
    duration_ms: float,
    tool_name: str | None = None,
    model_provider: str | None = None,
    success: bool | None = None,
    context_tokens: int | None = None,
    response_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a single performance event.

    Args:
        phase: What was measured (e.g., "model_generate", "tool_execute")
        duration_ms: Elapsed time in milliseconds
        tool_name: Name of the tool (for tool phases)
        model_provider: Provider identifier (e.g., "ollama", "anthropic")
        success: Whether the operation succeeded
        context_tokens: Number of tokens in the prompt context
        response_tokens: Number of tokens in the response
        extra: Additional fields to include in the log entry
    """
    _ensure_log_dir()
    _rotate_if_needed()
    _cleanup_old_logs()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "duration_ms": round(duration_ms, 3),
    }
    if tool_name is not None:
        entry["tool_name"] = tool_name
    if model_provider is not None:
        entry["model_provider"] = model_provider
    if success is not None:
        entry["success"] = success
    if context_tokens is not None:
        entry["context_tokens"] = context_tokens
    if response_tokens is not None:
        entry["response_tokens"] = response_tokens
    if extra:
        entry.update(extra)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError as e:
        logger.warning(f"Failed to write performance log: {e}")

    # Buffer for real-time queries
    with _perf_lock:
        _perf_buffer.append(entry)
        if len(_perf_buffer) > _MAX_BUFFER:
            _perf_buffer.pop(0)


@contextmanager
def perf_log(
    phase: str,
    tool_name: str | None = None,
    model_provider: str | None = None,
    context_tokens: int | None = None,
):
    """Context manager that times a block and logs a performance event.

    Yields a dict that can be updated with additional fields (e.g., success).

    Example:
        with perf_log("tool_execute", tool_name="read_file") as ctx:
            result = tools.execute("read_file", params)
            ctx["success"] = result.success
            ctx["response_tokens"] = len(result.output)
    """
    start = time.perf_counter()
    ctx: dict[str, Any] = {}
    try:
        yield ctx
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log_event(
            phase=phase,
            duration_ms=duration_ms,
            tool_name=tool_name or ctx.get("tool_name"),
            model_provider=model_provider or ctx.get("model_provider"),
            success=ctx.get("success"),
            context_tokens=context_tokens or ctx.get("context_tokens"),
            response_tokens=ctx.get("response_tokens"),
            extra={
                k: v
                for k, v in ctx.items()
                if k
                not in {
                    "tool_name",
                    "model_provider",
                    "success",
                    "context_tokens",
                    "response_tokens",
                }
            },
        )


def get_recent_events(
    phase: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent performance events from the in-memory buffer."""
    events = _perf_buffer
    if phase:
        events = [e for e in events if e.get("phase") == phase]
    return events[-limit:]


def get_summary(
    phase: str | None = None,
    window: int = 100,
) -> dict[str, Any]:
    """Compute summary statistics for recent events.

    Returns:
        dict with count, mean_ms, min_ms, max_ms, p95_ms
    """
    events = get_recent_events(phase=phase, limit=window)
    if not events:
        return {"count": 0, "mean_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}

    durations = [e["duration_ms"] for e in events]
    durations_sorted = sorted(durations)
    p95_idx = int(len(durations_sorted) * 0.95)
    p95_idx = min(p95_idx, len(durations_sorted) - 1)

    return {
        "count": len(durations),
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "p95_ms": round(durations_sorted[p95_idx], 3),
    }
