"""MCP tool-usage recorder for cost optimization.

Lifted from the GitHub Agentic Workflows token-efficiency pattern
(see `docs/patterns/token-optimization-from-github-2026-05.md`):
every MCP tool registered with a workflow ships its schema in the
system prompt; unused tools are pure context overhead. This module
records *which MCP tools are actually called* per workflow, so a
later analyzer can propose a pruned allowlist.

The recorder writes to a JSONL append-log under the configured forge
data directory. It is deliberately lightweight — no SQL migration,
no schema change, no hot-path latency. Records are stitched together
by `mcp_tool_pruner` when an operator asks for an analysis.

Disable in production hot paths by setting
``FORGE_MCP_USAGE_RECORD=0`` (the default is on; the recorder is
designed to be cheap enough to leave on always).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_LOG_PATH = Path.home() / ".animus" / "forge" / "mcp_tool_usage.jsonl"
ENV_PATH = "FORGE_MCP_USAGE_LOG"
ENV_ENABLED = "FORGE_MCP_USAGE_RECORD"


@dataclass
class MCPToolUsageRecord:
    """One MCP tool invocation, written to the usage log."""

    workflow_id: str
    step_id: str
    server: str
    tool: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    duration_ms: int | None = None
    is_error: bool = False


def _resolve_log_path() -> Path:
    """Resolve the usage-log path, honoring the env override."""
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override)
    return DEFAULT_LOG_PATH


def is_recording_enabled() -> bool:
    """Recording defaults to on; ``FORGE_MCP_USAGE_RECORD=0`` disables."""
    return os.environ.get(ENV_ENABLED, "1") != "0"


def record_mcp_tool_usage(
    workflow_id: str,
    step_id: str,
    server: str,
    tool: str,
    duration_ms: int | None = None,
    is_error: bool = False,
    log_path: Path | None = None,
) -> None:
    """Append one usage record to the JSONL log.

    Failures are logged at debug and swallowed — the recorder must
    never break a workflow run. The disable env var is checked here
    so callers can record unconditionally.
    """
    if not is_recording_enabled():
        return
    record = MCPToolUsageRecord(
        workflow_id=workflow_id,
        step_id=step_id,
        server=server,
        tool=tool,
        duration_ms=duration_ms,
        is_error=is_error,
    )
    target = log_path or _resolve_log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
    except OSError as exc:
        logger.debug("MCP tool-usage record failed to write: %s", exc)


def read_usage_records(
    log_path: Path | None = None,
    workflow_id: str | None = None,
) -> list[MCPToolUsageRecord]:
    """Read records from the JSONL log, optionally filtered by workflow."""
    target = log_path or _resolve_log_path()
    if not target.exists():
        return []

    out: list[MCPToolUsageRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed mcp-tool-usage line: %r", line[:120])
                continue
            if workflow_id and data.get("workflow_id") != workflow_id:
                continue
            out.append(
                MCPToolUsageRecord(
                    workflow_id=data.get("workflow_id", ""),
                    step_id=data.get("step_id", ""),
                    server=data.get("server", ""),
                    tool=data.get("tool", ""),
                    timestamp=data.get("timestamp", ""),
                    duration_ms=data.get("duration_ms"),
                    is_error=bool(data.get("is_error", False)),
                )
            )
    return out


def aggregate_tool_calls(
    records: Iterable[MCPToolUsageRecord],
) -> dict[str, dict[str, int]]:
    """Aggregate to ``{server: {tool: call_count}}`` for the analyzer."""
    out: dict[str, dict[str, int]] = {}
    for r in records:
        out.setdefault(r.server, {})
        out[r.server][r.tool] = out[r.server].get(r.tool, 0) + 1
    return out
