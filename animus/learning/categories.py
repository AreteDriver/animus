"""
Learning Categories and Data Structures

Defines learning categories, approval requirements, and the LearnedItem dataclass.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class LearningCategory(Enum):
    """Categories of learned information with different approval requirements."""

    STYLE = "style"  # Communication style preferences (auto)
    PREFERENCE = "preference"  # User preferences/dislikes (auto)
    WORKFLOW = "workflow"  # Repeated patterns/processes (notify)
    FACT = "fact"  # Factual information about user/world (confirm)
    CAPABILITY = "capability"  # New tool/integration permissions (approve)
    BOUNDARY = "boundary"  # Access/permission boundaries (approve)


class ApprovalRequirement(Enum):
    """What approval is needed for learning."""

    AUTO = "auto"  # Applied automatically
    NOTIFY = "notify"  # Applied, user notified
    CONFIRM = "confirm"  # User must confirm before applying
    APPROVE = "approve"  # User must explicitly approve


# Category to approval mapping
CATEGORY_APPROVAL: dict[LearningCategory, ApprovalRequirement] = {
    LearningCategory.STYLE: ApprovalRequirement.AUTO,
    LearningCategory.PREFERENCE: ApprovalRequirement.AUTO,
    LearningCategory.WORKFLOW: ApprovalRequirement.NOTIFY,
    LearningCategory.FACT: ApprovalRequirement.CONFIRM,
    LearningCategory.CAPABILITY: ApprovalRequirement.APPROVE,
    LearningCategory.BOUNDARY: ApprovalRequirement.APPROVE,
}


@dataclass
class LearnedItem:
    """A single learned piece of information."""

    id: str
    category: LearningCategory
    content: str
    confidence: float  # 0.0-1.0
    evidence: list[str]  # Memory IDs that support this learning
    created_at: datetime
    updated_at: datetime
    applied: bool = False
    approved_at: datetime | None = None
    approved_by: str | None = None
    source_pattern_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Rollback support
    version: int = 1
    previous_version_id: str | None = None

    @classmethod
    def create(
        cls,
        category: LearningCategory,
        content: str,
        confidence: float,
        evidence: list[str],
        source_pattern_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "LearnedItem":
        """Create a new LearnedItem with generated ID and timestamps."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            category=category,
            content=content,
            confidence=confidence,
            evidence=evidence,
            created_at=now,
            updated_at=now,
            source_pattern_id=source_pattern_id,
            metadata=metadata or {},
        )

    def apply(self, approved_by: str = "system") -> None:
        """Mark this learning as applied."""
        self.applied = True
        self.approved_at = datetime.now()
        self.approved_by = approved_by
        self.updated_at = datetime.now()

    def update_confidence(self, delta: float) -> None:
        """Adjust confidence by delta, clamping to [0.0, 1.0]."""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.updated_at = datetime.now()

    def create_new_version(self, new_content: str) -> "LearnedItem":
        """Create a new version of this item for rollback support."""
        now = datetime.now()
        return LearnedItem(
            id=str(uuid.uuid4()),
            category=self.category,
            content=new_content,
            confidence=self.confidence,
            evidence=self.evidence.copy(),
            created_at=self.created_at,
            updated_at=now,
            applied=self.applied,
            approved_at=self.approved_at,
            approved_by=self.approved_by,
            source_pattern_id=self.source_pattern_id,
            metadata=self.metadata.copy(),
            version=self.version + 1,
            previous_version_id=self.id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "applied": self.applied,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "source_pattern_id": self.source_pattern_id,
            "metadata": self.metadata,
            "version": self.version,
            "previous_version_id": self.previous_version_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=LearningCategory(data["category"]),
            content=data["content"],
            confidence=data["confidence"],
            evidence=data["evidence"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            applied=data.get("applied", False),
            approved_at=(
                datetime.fromisoformat(data["approved_at"])
                if data.get("approved_at")
                else None
            ),
            approved_by=data.get("approved_by"),
            source_pattern_id=data.get("source_pattern_id"),
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
            previous_version_id=data.get("previous_version_id"),
        )
