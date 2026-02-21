"""Human approval gates for self-improvement operations."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalStage(str, Enum):
    """Stage requiring approval."""

    PLAN = "plan"  # Before starting work
    APPLY = "apply"  # Before applying changes
    MERGE = "merge"  # Before merging PR


@dataclass
class ApprovalRequest:
    """A request for human approval."""

    id: str
    stage: ApprovalStage
    title: str
    description: str
    details: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: datetime | None = None
    decided_by: str | None = None
    reason: str | None = None


class ApprovalGate:
    """Manages human approval for self-improvement operations.

    Supports two modes:
    - In-memory (default): approvals live only in process memory.
    - Persistent (with backend): approvals stored in SQLite so the
      orchestrator can resume after restarts.
    """

    def __init__(self, backend: DatabaseBackend | None = None):
        """Initialize the approval gate.

        Args:
            backend: Optional database backend for persistence.
                     If None, approvals are in-memory only.
        """
        self._backend = backend
        self._pending_approvals: dict[str, ApprovalRequest] = {}
        self._approval_history: list[ApprovalRequest] = []

        if self._backend:
            self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the approvals table if it doesn't exist."""
        if not self._backend:
            return
        self._backend.execute(
            """CREATE TABLE IF NOT EXISTS self_improve_approvals (
                id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                decided_at TEXT,
                decided_by TEXT,
                reason TEXT
            )"""
        )

    def request_approval(
        self,
        stage: ApprovalStage,
        title: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """Create an approval request.

        Args:
            stage: Which stage this approval is for.
            title: Short title for the approval.
            description: Detailed description of what needs approval.
            details: Additional details (files, changes, etc).

        Returns:
            Created approval request.
        """
        import uuid

        request = ApprovalRequest(
            id=str(uuid.uuid4())[:8],
            stage=stage,
            title=title,
            description=description,
            details=details or {},
        )

        self._pending_approvals[request.id] = request
        logger.info(f"Created approval request {request.id}: {title}")

        if self._backend:
            self._persist_request(request)

        return request

    def _persist_request(self, request: ApprovalRequest) -> None:
        """Write an approval request to the database."""
        if not self._backend:
            return
        with self._backend.transaction():
            self._backend.execute(
                """INSERT OR REPLACE INTO self_improve_approvals
                   (id, stage, title, description, details, status, created_at, decided_at, decided_by, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.id,
                    request.stage.value,
                    request.title,
                    request.description,
                    json.dumps(request.details),
                    request.status.value,
                    request.created_at.isoformat(),
                    request.decided_at.isoformat() if request.decided_at else None,
                    request.decided_by,
                    request.reason,
                ),
            )

    def _load_request(self, request_id: str) -> ApprovalRequest | None:
        """Load an approval request from the database."""
        if not self._backend:
            return None
        row = self._backend.fetchone(
            "SELECT * FROM self_improve_approvals WHERE id = ?",
            (request_id,),
        )
        if not row:
            return None
        return self._row_to_request(row)

    @staticmethod
    def _row_to_request(row: dict) -> ApprovalRequest:
        """Convert a database row to an ApprovalRequest."""
        return ApprovalRequest(
            id=row["id"],
            stage=ApprovalStage(row["stage"]),
            title=row["title"],
            description=row["description"],
            details=json.loads(row["details"]) if row["details"] else {},
            status=ApprovalStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            decided_at=datetime.fromisoformat(row["decided_at"]) if row.get("decided_at") else None,
            decided_by=row.get("decided_by"),
            reason=row.get("reason"),
        )

    def get_pending(self, stage: ApprovalStage | None = None) -> list[ApprovalRequest]:
        """Get pending approval requests.

        Args:
            stage: Optional filter by stage.

        Returns:
            List of pending requests.
        """
        if self._backend:
            if stage:
                rows = self._backend.fetchall(
                    "SELECT * FROM self_improve_approvals WHERE status = 'pending' AND stage = ?",
                    (stage.value,),
                )
            else:
                rows = self._backend.fetchall(
                    "SELECT * FROM self_improve_approvals WHERE status = 'pending'"
                )
            return [self._row_to_request(r) for r in rows]

        pending = list(self._pending_approvals.values())
        if stage:
            pending = [r for r in pending if r.stage == stage]
        return pending

    def approve(
        self,
        request_id: str,
        approved_by: str = "human",
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        """Approve a request.

        Args:
            request_id: ID of request to approve.
            approved_by: Who approved it.
            reason: Optional reason.

        Returns:
            Updated request, or None if not found.
        """
        request = self._pending_approvals.get(request_id)
        if not request and self._backend:
            request = self._load_request(request_id)
        if not request:
            return None

        request.status = ApprovalStatus.APPROVED
        request.decided_at = datetime.now()
        request.decided_by = approved_by
        request.reason = reason

        self._pending_approvals.pop(request_id, None)
        self._approval_history.append(request)

        if self._backend:
            self._persist_request(request)

        logger.info(f"Approved request {request_id} by {approved_by}")
        return request

    def reject(
        self,
        request_id: str,
        rejected_by: str = "human",
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        """Reject a request.

        Args:
            request_id: ID of request to reject.
            rejected_by: Who rejected it.
            reason: Optional reason.

        Returns:
            Updated request, or None if not found.
        """
        request = self._pending_approvals.get(request_id)
        if not request and self._backend:
            request = self._load_request(request_id)
        if not request:
            return None

        request.status = ApprovalStatus.REJECTED
        request.decided_at = datetime.now()
        request.decided_by = rejected_by
        request.reason = reason

        self._pending_approvals.pop(request_id, None)
        self._approval_history.append(request)

        if self._backend:
            self._persist_request(request)

        logger.info(f"Rejected request {request_id} by {rejected_by}: {reason}")
        return request

    def is_approved(self, request_id: str) -> bool:
        """Check if a request was approved.

        Args:
            request_id: ID to check.

        Returns:
            True if approved.
        """
        # Check in-memory history first
        for request in self._approval_history:
            if request.id == request_id:
                return request.status == ApprovalStatus.APPROVED

        # Check database if available
        if self._backend:
            row = self._backend.fetchone(
                "SELECT status FROM self_improve_approvals WHERE id = ?",
                (request_id,),
            )
            if row:
                return row["status"] == ApprovalStatus.APPROVED.value

        return False

    async def wait_for_approval(
        self,
        request: ApprovalRequest,
        timeout: float = 3600.0,
        poll_interval: float = 5.0,
    ) -> ApprovalStatus:
        """Wait for an approval decision with async polling.

        Polls the in-memory state (and database if available) until
        the request is decided or the timeout expires.

        Args:
            request: Request to wait for.
            timeout: Maximum seconds to wait (default 1 hour).
            poll_interval: Seconds between polls (default 5s).

        Returns:
            Final status (APPROVED, REJECTED, or EXPIRED).
        """
        elapsed = 0.0
        while elapsed < timeout:
            # Check in-memory first
            if request.id not in self._pending_approvals:
                return request.status

            # Check database for external decisions
            if self._backend:
                db_request = self._load_request(request.id)
                if db_request and db_request.status != ApprovalStatus.PENDING:
                    # Sync in-memory state
                    request.status = db_request.status
                    request.decided_at = db_request.decided_at
                    request.decided_by = db_request.decided_by
                    request.reason = db_request.reason
                    self._pending_approvals.pop(request.id, None)
                    self._approval_history.append(request)
                    return request.status

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout — mark as expired
        request.status = ApprovalStatus.EXPIRED
        request.decided_at = datetime.now()
        self._pending_approvals.pop(request.id, None)
        self._approval_history.append(request)

        if self._backend:
            self._persist_request(request)

        logger.warning(f"Approval request {request.id} expired after {timeout}s")
        return ApprovalStatus.EXPIRED

    def auto_approve_for_testing(self, request_id: str) -> None:
        """Auto-approve a request (for testing only).

        Args:
            request_id: Request to auto-approve.
        """
        self.approve(request_id, approved_by="auto_test", reason="Auto-approved for testing")

    def get_history(
        self,
        stage: ApprovalStage | None = None,
        limit: int = 50,
    ) -> list[ApprovalRequest]:
        """Get approval history.

        Args:
            stage: Optional filter by stage.
            limit: Max items to return.

        Returns:
            List of historical requests.
        """
        if self._backend:
            if stage:
                rows = self._backend.fetchall(
                    "SELECT * FROM self_improve_approvals WHERE status != 'pending' AND stage = ? ORDER BY decided_at DESC LIMIT ?",
                    (stage.value, limit),
                )
            else:
                rows = self._backend.fetchall(
                    "SELECT * FROM self_improve_approvals WHERE status != 'pending' ORDER BY decided_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_request(r) for r in rows]

        history = self._approval_history.copy()
        if stage:
            history = [r for r in history if r.stage == stage]
        return history[-limit:]
