"""Proposal Queue — Approval lifecycle for ImprovementProposals.

Tracks proposals through states:
    DRAFT → SUBMITTED → PENDING_REVIEW → APPROVED → COMMISSIONED → COMPLETE
              ↓                ↓              ↓
           REJECTED     REJECTED       REJECTED

Every transition is logged with actor, timestamp, and reason.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.citizens.proposal import ImprovementProposal, ProposalStatus
from animus.logging import get_logger

logger = get_logger("citizens.proposal_queue")

_DEFAULT_SQLITE_PATH = Path.home() / ".config" / "animus" / "proposal_queue.db"


@dataclass
class Transition:
    """A single status transition in a proposal's lifecycle."""

    from_status: ProposalStatus
    to_status: ProposalStatus
    actor: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_status.value,
            "to": self.to_status.value,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }


@dataclass
class QueuedProposal:
    """A proposal with queue metadata."""

    proposal: ImprovementProposal
    transitions: list[Transition] = field(default_factory=list)
    submitted_at: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1=highest, 10=lowest
    tags: list[str] = field(default_factory=list)

    @property
    def current_status(self) -> ProposalStatus:
        if self.transitions:
            return self.transitions[-1].to_status
        return self.proposal.status

    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.submitted_at).total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "transitions": [t.to_dict() for t in self.transitions],
            "submitted_at": self.submitted_at.isoformat(),
            "priority": self.priority,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueuedProposal:
        proposal = ImprovementProposal.from_dict(data["proposal"])
        transitions = [
            Transition(
                from_status=ProposalStatus(t["from"]),
                to_status=ProposalStatus(t["to"]),
                actor=t["actor"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                reason=t.get("reason", ""),
            )
            for t in data.get("transitions", [])
        ]
        return cls(
            proposal=proposal,
            transitions=transitions,
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            priority=data.get("priority", 5),
            tags=data.get("tags", []),
        )


class ProposalQueue:
    """Queue for managing ImprovementProposal approval lifecycle.

    Usage:
        queue = ProposalQueue(memory_layer=memory)
        queue.submit(proposal, tags=["architect", "high-priority"])
        pending = queue.list_pending()
        queue.approve(pending[0].proposal.id, actor="human", reason="LGTM")
        queue.commission(proposal_id, actor="forge")
    """

    DEFAULT_FILENAME = "proposal_queue.json"

    def __init__(self, memory_layer: Any = None, storage_path: str | None = None):
        self._proposals: dict[str, QueuedProposal] = {}
        self.memory = memory_layer
        self._storage_path = Path(storage_path) if storage_path else None

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        proposal: ImprovementProposal,
        priority: int = 5,
        tags: list[str] | None = None,
    ) -> QueuedProposal:
        """Submit a proposal to the queue.

        Args:
            proposal: Proposal to submit.
            priority: Priority 1-10 (1=highest).
            tags: Optional tags for filtering.

        Returns:
            QueuedProposal with queue metadata.
        """
        qp = QueuedProposal(
            proposal=proposal,
            priority=priority,
            tags=tags or [],
        )
        qp.transitions.append(
            Transition(
                from_status=ProposalStatus.DRAFT,
                to_status=ProposalStatus.SUBMITTED,
                actor="citizen",
                reason="Auto-submitted after generation",
            )
        )
        proposal.status = ProposalStatus.SUBMITTED
        self._proposals[proposal.id] = qp

        logger.info(f"Proposal {proposal.id} submitted to queue (priority={priority})")
        self._persist()
        return qp

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def approve(self, proposal_id: str, actor: str = "human", reason: str = "") -> QueuedProposal | None:
        """Approve a proposal for commissioning.

        Args:
            proposal_id: ID of proposal to approve.
            actor: Who approved it.
            reason: Approval rationale.

        Returns:
            Updated QueuedProposal, or None if not found.
        """
        qp = self._proposals.get(proposal_id)
        if not qp:
            logger.warning(f"Proposal {proposal_id} not found in queue")
            return None

        if qp.current_status not in (ProposalStatus.SUBMITTED, ProposalStatus.PENDING_REVIEW):
            logger.warning(f"Cannot approve proposal in status {qp.current_status.value}")
            return qp

        qp.transitions.append(
            Transition(
                from_status=qp.current_status,
                to_status=ProposalStatus.APPROVED,
                actor=actor,
                reason=reason,
            )
        )
        qp.proposal.status = ProposalStatus.APPROVED
        qp.proposal.approved_by = actor
        qp.proposal.approved_at = datetime.now()

        logger.info(f"Proposal {proposal_id} approved by {actor}")
        self._persist()

        # Auto-commission to Forge if enabled
        if __import__("os").environ.get("ANIMUS_AUTO_COMMISSION") == "1":
            try:
                from animus.citizens.commissioner import ForgeCommissioner
                commissioner = ForgeCommissioner()
                result = commissioner.commission(qp.proposal)
                if result.success:
                    logger.info(f"Proposal {proposal_id} commissioned to Forge")
                    qp.transitions.append(
                        Transition(
                            from_status=ProposalStatus.APPROVED,
                            to_status=ProposalStatus.COMMISSIONED,
                            actor="forge",
                            reason="Auto-commissioned after approval",
                        )
                    )
                    self._persist()
                else:
                    logger.warning(f"Auto-commission failed: {result.error}")
            except Exception as e:
                logger.warning(f"Auto-commission error: {e}")

        return qp

    def reject(self, proposal_id: str, actor: str = "human", reason: str = "") -> QueuedProposal | None:
        """Reject a proposal.

        Args:
            proposal_id: ID of proposal to reject.
            actor: Who rejected it.
            reason: Rejection rationale.

        Returns:
            Updated QueuedProposal, or None if not found.
        """
        qp = self._proposals.get(proposal_id)
        if not qp:
            return None

        if qp.current_status in (ProposalStatus.COMPLETE, ProposalStatus.REJECTED):
            return qp

        qp.transitions.append(
            Transition(
                from_status=qp.current_status,
                to_status=ProposalStatus.REJECTED,
                actor=actor,
                reason=reason,
            )
        )
        qp.proposal.status = ProposalStatus.REJECTED

        logger.info(f"Proposal {proposal_id} rejected by {actor}: {reason}")
        self._persist()
        return qp

    def commission(self, proposal_id: str, actor: str = "forge", reason: str = "") -> QueuedProposal | None:
        """Mark a proposal as commissioned to Forge.

        Args:
            proposal_id: ID of proposal.
            actor: Who commissioned it.
            reason: Commission rationale.

        Returns:
            Updated QueuedProposal, or None if not found.
        """
        qp = self._proposals.get(proposal_id)
        if not qp:
            return None

        if qp.current_status != ProposalStatus.APPROVED:
            logger.warning(f"Cannot commission proposal in status {qp.current_status.value}")
            return qp

        qp.transitions.append(
            Transition(
                from_status=ProposalStatus.APPROVED,
                to_status=ProposalStatus.COMMISSIONED,
                actor=actor,
                reason=reason,
            )
        )
        qp.proposal.status = ProposalStatus.COMMISSIONED

        logger.info(f"Proposal {proposal_id} commissioned by {actor}")
        self._persist()
        return qp

    def complete(self, proposal_id: str, actor: str = "forge", reason: str = "") -> QueuedProposal | None:
        """Mark a proposal as complete.

        Args:
            proposal_id: ID of proposal.
            actor: Who completed it.
            reason: Completion rationale.

        Returns:
            Updated QueuedProposal, or None if not found.
        """
        qp = self._proposals.get(proposal_id)
        if not qp:
            return None

        qp.transitions.append(
            Transition(
                from_status=qp.current_status,
                to_status=ProposalStatus.COMPLETE,
                actor=actor,
                reason=reason,
            )
        )
        qp.proposal.status = ProposalStatus.COMPLETE

        logger.info(f"Proposal {proposal_id} marked complete by {actor}")
        self._persist()
        return qp

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_by_status(self, status: ProposalStatus) -> list[QueuedProposal]:
        """List proposals filtered by status."""
        return [qp for qp in self._proposals.values() if qp.current_status == status]

    def list_pending(self) -> list[QueuedProposal]:
        """List proposals awaiting review (SUBMITTED or PENDING_REVIEW)."""
        return [
            qp for qp in self._proposals.values()
            if qp.current_status in (ProposalStatus.SUBMITTED, ProposalStatus.PENDING_REVIEW)
        ]

    def list_approved(self) -> list[QueuedProposal]:
        """List approved but not yet commissioned proposals."""
        return [qp for qp in self._proposals.values() if qp.current_status == ProposalStatus.APPROVED]

    def list_commissioned(self) -> list[QueuedProposal]:
        """List commissioned but not yet complete proposals."""
        return [qp for qp in self._proposals.values() if qp.current_status == ProposalStatus.COMMISSIONED]

    def list_completed(self) -> list[QueuedProposal]:
        """List completed proposals."""
        return [qp for qp in self._proposals.values() if qp.current_status == ProposalStatus.COMPLETE]

    def list_rejected(self) -> list[QueuedProposal]:
        """List rejected proposals."""
        return [qp for qp in self._proposals.values() if qp.current_status == ProposalStatus.REJECTED]

    def get(self, proposal_id: str) -> QueuedProposal | None:
        """Get a specific queued proposal by ID."""
        return self._proposals.get(proposal_id)

    def get_backlog(self) -> list[QueuedProposal]:
        """Get all active (non-complete, non-rejected) proposals sorted by priority."""
        active = [
            qp for qp in self._proposals.values()
            if qp.current_status not in (ProposalStatus.COMPLETE, ProposalStatus.REJECTED)
        ]
        return sorted(active, key=lambda qp: (qp.priority, qp.age_hours))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _resolve_storage_path(self) -> Path | None:
        """Resolve the storage path for the queue JSON file.

        Priority:
        1. Explicitly provided storage_path
        2. Memory layer data_dir / proposal_queue.json
        3. None (in-memory only)
        """
        if self._storage_path:
            return self._storage_path
        if self.memory is not None:
            try:
                data_dir = getattr(self.memory, "data_dir", None)
                if data_dir:
                    return Path(data_dir) / self.DEFAULT_FILENAME
            except Exception:
                pass
        return None

    def _persist(self) -> None:
        """Persist queue state to a dedicated JSON file."""
        path = self._resolve_storage_path()
        if path is None:
            # Fallback to memory-only if no storage path available
            self._persist_to_memory()
            return

        try:
            data = [qp.to_dict() for qp in self._proposals.values()]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Queue file persistence failed: {e}")
            self._persist_to_memory()

    def _persist_to_sqlite(self) -> None:
        """Fallback: store queue state in SQLite when memory layer is unavailable."""
        try:
            _DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(_DEFAULT_SQLITE_PATH))
            conn.execute("CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, data TEXT)")
            conn.execute("DELETE FROM proposals")
            for qp in self._proposals.values():
                conn.execute(
                    "INSERT OR REPLACE INTO proposals (id, data) VALUES (?, ?)",
                    (qp.proposal.id, json.dumps(qp.to_dict(), default=str)),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Queue SQLite persistence failed: {e}")

    def _load_from_sqlite(self) -> int:
        """Load queue state from SQLite fallback."""
        if not _DEFAULT_SQLITE_PATH.exists():
            return 0
        try:
            conn = sqlite3.connect(str(_DEFAULT_SQLITE_PATH))
            rows = conn.execute("SELECT data FROM proposals").fetchall()
            conn.close()
            for (data,) in rows:
                qp = QueuedProposal.from_dict(json.loads(data))
                self._proposals[qp.proposal.id] = qp
            return len(self._proposals)
        except Exception as e:
            logger.debug(f"Queue SQLite load failed: {e}")
            return 0

    def _persist_to_memory(self) -> None:
        """Fallback: store queue state in memory layer."""
        if self.memory is None:
            self._persist_to_sqlite()
            return
        try:
            from animus.memory import MemoryType

            data = [qp.to_dict() for qp in self._proposals.values()]
            self.memory.remember(
                content=f"Proposal queue state: {len(data)} proposals",
                memory_type=MemoryType.PROCEDURAL,
                tags=["proposal_queue", "state"],
                metadata={"proposals": data, "count": len(data)},
            )
        except Exception as e:
            logger.debug(f"Queue memory persistence failed: {e}")
            self._persist_to_sqlite()

    def load_from_memory(self) -> int:
        """Load queue state from disk or memory.

        Returns:
            Number of proposals loaded.
        """
        # Try dedicated JSON file first
        path = self._resolve_storage_path()
        if path is not None and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for p_data in data:
                    try:
                        qp = QueuedProposal.from_dict(p_data)
                        self._proposals[qp.proposal.id] = qp
                    except Exception:
                        continue
                return len(self._proposals)
            except Exception as e:
                logger.warning(f"Queue file load failed: {e}")

        # Fallback to SQLite
        sqlite_count = self._load_from_sqlite()
        if sqlite_count > 0:
            return sqlite_count

        # Fallback to memory search
        if self.memory is not None:
            try:
                from animus.memory import MemoryType
                results = self.memory.search(
                    query="proposal_queue state",
                    memory_type=MemoryType.PROCEDURAL,
                    limit=5,
                )
                for mem in results:
                    if hasattr(mem, "to_dict"):
                        mem_dict = mem.to_dict()
                    elif isinstance(mem, dict):
                        mem_dict = mem
                    else:
                        continue
                    meta = mem_dict.get("metadata", {})
                    for p_data in meta.get("proposals", []):
                        try:
                            qp = QueuedProposal.from_dict(p_data)
                            self._proposals[qp.proposal.id] = qp
                        except Exception:
                            continue
                return len(self._proposals)
            except Exception as e:
                logger.debug(f"Queue memory load failed: {e}")

        return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Get queue statistics."""
        return {
            "total": len(self._proposals),
            "pending": len(self.list_pending()),
            "approved": len(self.list_approved()),
            "commissioned": len(self.list_commissioned()),
            "complete": len(self.list_completed()),
            "rejected": len(self.list_rejected()),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"ProposalQueue(total={s['total']}, pending={s['pending']}, approved={s['approved']}, complete={s['complete']})"
