"""Stable checkpoint façade for external consumers.

Exposes a versioned, narrow API surface so Bootstrap (and other upper-layer
packages) do not need to import internal ``head.checkpoint`` types directly.

Version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus_kernel.head.checkpoint import HeadCheckpoint, HeadCheckpointStore


@dataclass(frozen=True)
class CheckpointData:
    """Stable snapshot of a session checkpoint.

    This dataclass is the *public* contract.  Internal ``HeadCheckpoint``
    may evolve; this façade maps to the stable fields consumers need.
    """

    session_id: str
    started_at: datetime
    last_active_at: datetime
    project_root: str | None = None
    model: str = ""
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    total_tokens: int = 0
    turns: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_internal(cls, cp: HeadCheckpoint) -> CheckpointData:
        return cls(
            session_id=cp.session_id,
            started_at=cp.started_at,
            last_active_at=cp.last_active_at,
            project_root=cp.project_root,
            model=cp.model,
            messages=cp.messages,
            summary=cp.summary,
            total_tokens=cp.total_tokens,
            turns=cp.turns,
            metadata=cp.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "project_root": self.project_root,
            "model": self.model,
            "messages": self.messages,
            "summary": self.summary,
            "total_tokens": self.total_tokens,
            "turns": self.turns,
            "metadata": self.metadata,
        }


class CheckpointFacade:
    """Stable façade over ``HeadCheckpointStore``.

    Usage::

        facade = CheckpointFacade()
        facade.save(CheckpointData(session_id="abc", ...))
        recent = facade.list_recent(limit=1)
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._store = HeadCheckpointStore(db_path=db_path)

    def save(self, data: CheckpointData) -> None:
        """Persist a checkpoint."""
        internal = HeadCheckpoint(
            session_id=data.session_id,
            started_at=data.started_at,
            last_active_at=data.last_active_at,
            project_root=data.project_root,
            model=data.model,
            messages=data.messages,
            summary=data.summary,
            total_tokens=data.total_tokens,
            turns=data.turns,
            metadata=data.metadata,
        )
        self._store.save(internal)

    def load(self, session_id: str) -> CheckpointData | None:
        """Load a checkpoint by session_id, or None if absent."""
        internal = self._store.load(session_id)
        if internal is None:
            return None
        return CheckpointData.from_internal(internal)

    def list_recent(self, limit: int = 10) -> list[CheckpointData]:
        """Return the most recent checkpoints, newest first."""
        internals = self._store.list_recent(limit=limit)
        return [CheckpointData.from_internal(ic) for ic in internals]

    def delete(self, session_id: str) -> bool:
        """Delete a checkpoint. Returns True if one was removed."""
        return self._store.delete(session_id)
