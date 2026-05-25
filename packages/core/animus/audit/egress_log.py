"""MCP-egress audit log (Stage 3.B).

Append-only JSONL at ``<data_dir>/audit/mcp-egress-YYYY-MM-DD.jsonl``. One
file per UTC day for trivial rotation. 30-day retention per ARETE's locked
answer; older files are deleted by :func:`prune_old_logs`.

**No raw content is ever logged.** Only metadata: tool name, tier scope
permitted by the caller, result count, redaction count, response byte
length. This is by design — the audit log itself must not become a
secondary exfil vector.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus.memory.types import Sensitivity

logger = logging.getLogger("audit.egress")


_FILENAME_RE = re.compile(r"^mcp-egress-(\d{4}-\d{2}-\d{2})\.jsonl$")


class EgressAuditLog:
    """Daily-rotated append-only audit log for MCP tool egress events.

    Each entry is a single JSON line. Schema:

    ``{
        "ts": ISO-8601 UTC timestamp,
        "tool": tool name,
        "tier_scope": sorted list of permitted tier names,
        "result_count": number of memories returned,
        "redaction_count": number of DLP hits caught at egress,
        "response_bytes": byte length of the rendered response string,
    }``
    """

    def __init__(self, data_dir: Path | str):
        self.audit_dir = Path(data_dir) / "audit"

    def _file_for_today(self) -> Path:
        today = datetime.now(timezone.utc).date().isoformat()
        return self.audit_dir / f"mcp-egress-{today}.jsonl"

    def record(
        self,
        tool: str,
        tier_scope: set[Sensitivity] | None,
        result_count: int,
        redaction_count: int,
        response_bytes: int,
    ) -> None:
        """Append a single audit entry. Never raises — audit failures must
        not break tool execution."""
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "tier_scope": (
                    sorted(t.value for t in tier_scope) if tier_scope is not None else None
                ),
                "result_count": result_count,
                "redaction_count": redaction_count,
                "response_bytes": response_bytes,
            }
            with self._file_for_today().open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to write egress audit entry: %s", e)


def prune_old_logs(data_dir: Path | str, retention_days: int = 30) -> list[Path]:
    """Delete egress audit files older than ``retention_days``.

    Returns the list of deleted file paths. Safe to call repeatedly.
    """
    audit_dir = Path(data_dir) / "audit"
    if not audit_dir.is_dir():
        return []

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
    deleted: list[Path] = []

    for path in audit_dir.iterdir():
        if not path.is_file():
            continue
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            try:
                path.unlink()
                deleted.append(path)
            except OSError as e:
                logger.warning("Failed to prune audit file %s: %s", path, e)
    return deleted
