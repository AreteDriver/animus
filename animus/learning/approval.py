"""
Approval Workflow Management

Handles approval requests for learnings that require user consent.
"""

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.learning.categories import (
    CATEGORY_APPROVAL,
    ApprovalRequirement,
    LearningCategory,
)
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.learning.categories import LearnedItem

logger = get_logger("learning.approval")


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """A request for user approval of a learning."""

    id: str
    learned_item_id: str
    category: LearningCategory
    description: str
    evidence_summary: str
    created_at: datetime
    expires_at: datetime | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    user_response: str | None = None
    responded_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        learned_item: "LearnedItem",
        evidence_summary: str,
        expires_in_days: int = 7,
    ) -> "ApprovalRequest":
        """Create a new approval request for a learned item."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            learned_item_id=learned_item.id,
            category=learned_item.category,
            description=learned_item.content,
            evidence_summary=evidence_summary,
            created_at=now,
            expires_at=now + timedelta(days=expires_in_days),
        )

    def is_expired(self) -> bool:
        """Check if this request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def approve(self, comment: str | None = None) -> None:
        """Mark as approved."""
        self.status = ApprovalStatus.APPROVED
        self.user_response = comment
        self.responded_at = datetime.now()

    def reject(self, reason: str | None = None) -> None:
        """Mark as rejected."""
        self.status = ApprovalStatus.REJECTED
        self.user_response = reason
        self.responded_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "learned_item_id": self.learned_item_id,
            "category": self.category.value,
            "description": self.description,
            "evidence_summary": self.evidence_summary,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "user_response": self.user_response,
            "responded_at": (self.responded_at.isoformat() if self.responded_at else None),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequest":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            learned_item_id=data["learned_item_id"],
            category=LearningCategory(data["category"]),
            description=data["description"],
            evidence_summary=data["evidence_summary"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            status=ApprovalStatus(data.get("status", "pending")),
            user_response=data.get("user_response"),
            responded_at=(
                datetime.fromisoformat(data["responded_at"]) if data.get("responded_at") else None
            ),
            metadata=data.get("metadata", {}),
        )


class ApprovalManager:
    """
    Manages approval workflows for learning.

    Different categories have different approval requirements:
    - AUTO: Applied immediately
    - NOTIFY: Applied, user informed
    - CONFIRM: User must confirm
    - APPROVE: User must explicitly approve
    """

    def __init__(
        self,
        data_dir: Path,
        notification_callback: Callable[[str], None] | None = None,
    ):
        self.data_dir = data_dir
        self._notification_callback = notification_callback
        self._pending_requests: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self._load_requests()

    def _load_requests(self) -> None:
        """Load pending requests from disk."""
        requests_file = self.data_dir / "approval_requests.json"
        if requests_file.exists():
            try:
                with open(requests_file) as f:
                    data = json.load(f)
                for item in data.get("pending", []):
                    req = ApprovalRequest.from_dict(item)
                    if not req.is_expired():
                        self._pending_requests[req.id] = req
                    else:
                        req.status = ApprovalStatus.EXPIRED
                        self._history.append(req)
                for item in data.get("history", []):
                    self._history.append(ApprovalRequest.from_dict(item))
                logger.info(f"Loaded {len(self._pending_requests)} pending requests")
            except Exception as e:
                logger.error(f"Failed to load approval requests: {e}")

    def _save_requests(self) -> None:
        """Save requests to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        requests_file = self.data_dir / "approval_requests.json"
        data = {
            "pending": [r.to_dict() for r in self._pending_requests.values()],
            "history": [r.to_dict() for r in self._history[-100:]],  # Keep last 100
        }
        with open(requests_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_approval_requirement(self, category: LearningCategory) -> ApprovalRequirement:
        """Get the approval requirement for a category."""
        return CATEGORY_APPROVAL.get(category, ApprovalRequirement.APPROVE)

    def needs_approval(self, learned_item: "LearnedItem") -> bool:
        """Check if a learned item needs explicit approval."""
        requirement = self.get_approval_requirement(learned_item.category)
        return requirement in (ApprovalRequirement.CONFIRM, ApprovalRequirement.APPROVE)

    def should_notify(self, learned_item: "LearnedItem") -> bool:
        """Check if user should be notified about a learning."""
        requirement = self.get_approval_requirement(learned_item.category)
        return requirement == ApprovalRequirement.NOTIFY

    def request_approval(
        self,
        learned_item: "LearnedItem",
        evidence_summary: str = "",
    ) -> ApprovalRequest:
        """
        Create an approval request for a learning.

        Args:
            learned_item: The item needing approval
            evidence_summary: Summary of evidence supporting this learning

        Returns:
            The created approval request
        """
        request = ApprovalRequest.create(
            learned_item=learned_item,
            evidence_summary=evidence_summary
            or f"Based on {len(learned_item.evidence)} observations",
        )

        self._pending_requests[request.id] = request
        self._save_requests()

        logger.info(f"Created approval request: {request.id}")
        return request

    def approve(self, request_id: str, user_comment: str | None = None) -> bool:
        """
        Approve a learning request.

        Args:
            request_id: ID of request to approve
            user_comment: Optional comment from user

        Returns:
            True if approved, False if not found
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.approve(user_comment)
        del self._pending_requests[request_id]
        self._history.append(request)
        self._save_requests()

        logger.info(f"Approved request: {request_id}")
        return True

    def reject(self, request_id: str, reason: str | None = None) -> bool:
        """
        Reject a learning request.

        Args:
            request_id: ID of request to reject
            reason: Optional reason for rejection

        Returns:
            True if rejected, False if not found
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.reject(reason)
        del self._pending_requests[request_id]
        self._history.append(request)
        self._save_requests()

        logger.info(f"Rejected request: {request_id}")
        return True

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        # Check for expired requests
        expired = []
        for req_id, req in self._pending_requests.items():
            if req.is_expired():
                req.status = ApprovalStatus.EXPIRED
                expired.append(req_id)
                self._history.append(req)

        for req_id in expired:
            del self._pending_requests[req_id]

        if expired:
            self._save_requests()

        return list(self._pending_requests.values())

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a specific request by ID."""
        return self._pending_requests.get(request_id)

    def get_history(self, limit: int = 50) -> list[ApprovalRequest]:
        """Get approval history."""
        return self._history[-limit:]

    def notify_user(self, message: str) -> None:
        """
        Send notification to user.

        Args:
            message: Message to send
        """
        if self._notification_callback:
            self._notification_callback(message)
        logger.info(f"User notification: {message}")

    def get_statistics(self) -> dict[str, Any]:
        """Get approval statistics."""
        pending = len(self._pending_requests)
        approved = sum(1 for r in self._history if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in self._history if r.status == ApprovalStatus.REJECTED)
        expired = sum(1 for r in self._history if r.status == ApprovalStatus.EXPIRED)

        return {
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "expired": expired,
            "total_processed": approved + rejected + expired,
        }
