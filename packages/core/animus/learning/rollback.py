"""
Unlearn and Rollback Functionality

Manages unlearning specific items and rolling back to previous states.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.learning.categories import LearnedItem

logger = get_logger("learning.rollback")


@dataclass
class RollbackPoint:
    """A point in time to which learning can be rolled back."""

    id: str
    timestamp: datetime
    description: str
    learned_item_ids: list[str]  # Items active at this point
    checksum: str  # For integrity verification
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        description: str,
        learned_items: list["LearnedItem"],
        metadata: dict[str, Any] | None = None,
    ) -> "RollbackPoint":
        """Create a new rollback point."""
        item_ids = [item.id for item in learned_items if item.applied]
        # Create checksum from sorted item IDs
        checksum_str = ",".join(sorted(item_ids))
        checksum = hashlib.sha256(checksum_str.encode()).hexdigest()[:16]

        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            description=description,
            learned_item_ids=item_ids,
            checksum=checksum,
            metadata=metadata or {},
        )

    def verify_integrity(self, current_item_ids: list[str]) -> bool:
        """
        Verify this rollback point's integrity.

        Args:
            current_item_ids: Current list of learned item IDs

        Returns:
            True if integrity check passes
        """
        # Check that all expected items still exist
        current_set = set(current_item_ids)
        for item_id in self.learned_item_ids:
            if item_id not in current_set:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "learned_item_ids": self.learned_item_ids,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RollbackPoint":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data["description"],
            learned_item_ids=data["learned_item_ids"],
            checksum=data["checksum"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class UnlearnRecord:
    """Record of an unlearned item."""

    id: str
    learned_item_id: str
    learned_item_content: str
    unlearned_at: datetime
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "learned_item_id": self.learned_item_id,
            "learned_item_content": self.learned_item_content,
            "unlearned_at": self.unlearned_at.isoformat(),
            "reason": self.reason,
        }


class RollbackManager:
    """
    Manages unlearning and rollback of learned behaviors.

    Features:
    - Unlearn specific items
    - Rollback to a point in time
    - Create named checkpoints
    - Verify rollback integrity
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._rollback_points: list[RollbackPoint] = []
        self._unlearn_history: list[UnlearnRecord] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load rollback data from disk."""
        data_file = self.data_dir / "rollback_data.json"
        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                for item in data.get("rollback_points", []):
                    self._rollback_points.append(RollbackPoint.from_dict(item))
                for item in data.get("unlearn_history", []):
                    self._unlearn_history.append(
                        UnlearnRecord(
                            id=item["id"],
                            learned_item_id=item["learned_item_id"],
                            learned_item_content=item["learned_item_content"],
                            unlearned_at=datetime.fromisoformat(item["unlearned_at"]),
                            reason=item.get("reason"),
                        )
                    )
                logger.info(f"Loaded {len(self._rollback_points)} rollback points")
            except Exception as e:
                logger.error(f"Failed to load rollback data: {e}")

    def _save_data(self) -> None:
        """Save rollback data to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data_file = self.data_dir / "rollback_data.json"
        data = {
            "rollback_points": [p.to_dict() for p in self._rollback_points[-50:]],
            "unlearn_history": [r.to_dict() for r in self._unlearn_history[-100:]],
        }
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_checkpoint(
        self,
        description: str,
        learned_items: list["LearnedItem"],
        metadata: dict[str, Any] | None = None,
    ) -> RollbackPoint:
        """
        Create a named rollback checkpoint.

        Args:
            description: Human-readable description
            learned_items: Current list of learned items
            metadata: Optional additional metadata

        Returns:
            The created rollback point
        """
        point = RollbackPoint.create(
            description=description,
            learned_items=learned_items,
            metadata=metadata,
        )
        self._rollback_points.append(point)
        self._save_data()

        logger.info(f"Created checkpoint: {point.id} - {description}")
        return point

    def record_unlearn(
        self,
        learned_item: "LearnedItem",
        reason: str | None = None,
    ) -> UnlearnRecord:
        """
        Record that an item was unlearned.

        Args:
            learned_item: The item that was unlearned
            reason: Optional reason for unlearning

        Returns:
            The unlearn record
        """
        record = UnlearnRecord(
            id=str(uuid.uuid4()),
            learned_item_id=learned_item.id,
            learned_item_content=learned_item.content,
            unlearned_at=datetime.now(),
            reason=reason,
        )
        self._unlearn_history.append(record)
        self._save_data()

        logger.info(f"Recorded unlearn: {learned_item.id}")
        return record

    def get_items_to_unlearn(
        self,
        rollback_point_id: str,
        current_items: list["LearnedItem"],
    ) -> list[str]:
        """
        Get list of item IDs that should be unlearned to reach a rollback point.

        Args:
            rollback_point_id: Target rollback point
            current_items: Current list of learned items

        Returns:
            List of item IDs to unlearn
        """
        point = self._get_point(rollback_point_id)
        if not point:
            return []

        target_set = set(point.learned_item_ids)
        current_applied = [item.id for item in current_items if item.applied]

        # Items that are currently applied but weren't at the rollback point
        to_unlearn = [item_id for item_id in current_applied if item_id not in target_set]

        return to_unlearn

    def _get_point(self, point_id: str) -> RollbackPoint | None:
        """Get a rollback point by ID."""
        for point in self._rollback_points:
            if point.id == point_id:
                return point
        return None

    def get_rollback_points(self) -> list[RollbackPoint]:
        """Get available rollback points."""
        return list(self._rollback_points)

    def get_point_by_time(self, target_time: datetime) -> RollbackPoint | None:
        """
        Find the rollback point closest to but not after target_time.

        Args:
            target_time: Target timestamp

        Returns:
            Closest rollback point, or None if none before target_time
        """
        candidates = [p for p in self._rollback_points if p.timestamp <= target_time]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.timestamp)

    def get_unlearn_history(self, limit: int = 50) -> list[UnlearnRecord]:
        """Get unlearn history."""
        return self._unlearn_history[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get rollback statistics."""
        return {
            "checkpoint_count": len(self._rollback_points),
            "unlearn_count": len(self._unlearn_history),
            "oldest_checkpoint": (
                self._rollback_points[0].timestamp.isoformat() if self._rollback_points else None
            ),
            "newest_checkpoint": (
                self._rollback_points[-1].timestamp.isoformat() if self._rollback_points else None
            ),
        }
