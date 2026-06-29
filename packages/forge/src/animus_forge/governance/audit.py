"""Audit trail for governance decisions.

Every decision made by the Policy Decision Point is recorded in the AuditTrail.
This provides non-repudiation, post-hoc analysis, and compliance reporting.

The audit log is append-only. Entries are immutable once written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """A single audited governance decision.

    Attributes:
        timestamp: When the decision was recorded
        decision_type: "allow", "deny", or "require_approval"
        action: The action that was evaluated
        policy: Policy name that was applied
        rule: Rule name that matched (if any)
        reason: Human-readable rationale
        context: Snapshot of the evaluated context
        request_id: Correlation ID for distributed tracing
    """

    timestamp: datetime
    decision_type: str
    action: str
    policy: str
    rule: str | None
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type,
            "action": self.action,
            "policy": self.policy,
            "rule": self.rule,
            "reason": self.reason,
            "context": self.context,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            decision_type=data["decision_type"],
            action=data["action"],
            policy=data["policy"],
            rule=data.get("rule"),
            reason=data["reason"],
            context=data.get("context", {}),
            request_id=data.get("request_id", ""),
        )


class AuditTrail:
    """Append-only audit log for governance decisions.

    Supports both in-memory and file-backed persistence.
    The file backend writes JSONL (one JSON object per line) for
    easy streaming and append-only semantics.
    """

    def __init__(self, file_path: Path | str | None = None):
        self._entries: list[AuditEntry] = []
        self._file_path = Path(file_path) if file_path else None

    def record(self, decision: Any) -> None:
        """Record a decision in the audit trail.

        Accepts either a Decision object or any object with a `to_dict()` method.
        """
        entry = AuditEntry(
            timestamp=datetime.now(),
            decision_type=decision.effect,
            action=decision.action,
            policy=decision.policy,
            rule=decision.rule,
            reason=decision.reason,
            context=dict(decision.context),
            request_id=decision.request_id,
        )
        self._entries.append(entry)

        # Append to file if configured
        if self._file_path:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def entries(
        self,
        action: str | None = None,
        policy: str | None = None,
        since: datetime | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        results = self._entries
        if action:
            results = [e for e in results if e.action == action]
        if policy:
            results = [e for e in results if e.policy == policy]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results

    def summary(self) -> dict[str, Any]:
        """Return summary statistics of the audit trail."""
        total = len(self._entries)
        by_type: dict[str, int] = {}
        by_policy: dict[str, int] = {}
        for e in self._entries:
            by_type[e.decision_type] = by_type.get(e.decision_type, 0) + 1
            by_policy[e.policy] = by_policy.get(e.policy, 0) + 1

        return {
            "total_entries": total,
            "by_decision_type": by_type,
            "by_policy": by_policy,
            "first_timestamp": self._entries[0].timestamp.isoformat() if self._entries else None,
            "last_timestamp": self._entries[-1].timestamp.isoformat() if self._entries else None,
        }

    def export(self, path: str, format: str = "jsonl") -> None:
        """Export audit trail to file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(out_path, "w") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        elif format == "json":
            data = [e.to_dict() for e in self._entries]
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, path: str) -> None:
        """Load audit entries from a JSONL file."""
        in_path = Path(path)
        if not in_path.exists():
            return

        with open(in_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self._entries.append(AuditEntry.from_dict(data))
