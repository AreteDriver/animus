"""Human approval gates for self-improvement operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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

    In a real implementation, this would integrate with:
    - A web UI for approval
    - Slack/Discord notifications
    - GitHub PR reviews
    """

    def __init__(self):
        """Initialize the approval gate."""
        self._pending_approvals: dict[str, ApprovalRequest] = {}
        self._approval_history: list[ApprovalRequest] = []

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

        return request

    def get_pending(self, stage: ApprovalStage | None = None) -> list[ApprovalRequest]:
        """Get pending approval requests.

        Args:
            stage: Optional filter by stage.

        Returns:
            List of pending requests.
        """
        pending = [r for r in self._pending_approvals.values()]
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
        if not request:
            return None

        request.status = ApprovalStatus.APPROVED
        request.decided_at = datetime.now()
        request.decided_by = approved_by
        request.reason = reason

        del self._pending_approvals[request_id]
        self._approval_history.append(request)

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
        if not request:
            return None

        request.status = ApprovalStatus.REJECTED
        request.decided_at = datetime.now()
        request.decided_by = rejected_by
        request.reason = reason

        del self._pending_approvals[request_id]
        self._approval_history.append(request)

        logger.info(f"Rejected request {request_id} by {rejected_by}: {reason}")
        return request

    def is_approved(self, request_id: str) -> bool:
        """Check if a request was approved.

        Args:
            request_id: ID to check.

        Returns:
            True if approved.
        """
        # Check history
        for request in self._approval_history:
            if request.id == request_id:
                return request.status == ApprovalStatus.APPROVED
        return False

    def wait_for_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Wait for approval decision.

        In a real implementation, this would:
        - Send notifications
        - Poll for decision
        - Handle timeouts

        For now, this is synchronous and expects approval to be
        set externally before calling.

        Args:
            request: Request to wait for.

        Returns:
            Final status.
        """
        # In a real implementation, this would be async and poll
        # For now, just return current status
        if request.id in self._pending_approvals:
            return ApprovalStatus.PENDING
        return request.status

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
        history = self._approval_history.copy()
        if stage:
            history = [r for r in history if r.stage == stage]
        return history[-limit:]
