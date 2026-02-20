"""
Learning Transparency and Logging System

Provides full transparency into what Animus has learned.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.learning.categories import LearnedItem

logger = get_logger("learning.transparency")


@dataclass
class LearningEvent:
    """A logged learning event."""

    id: str
    event_type: str  # detected, proposed, approved, applied, rejected, rolled_back
    learned_item_id: str
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    user_action: str | None = None  # approve, reject, modify, rollback

    @classmethod
    def create(
        cls,
        event_type: str,
        learned_item_id: str,
        details: dict[str, Any] | None = None,
        user_action: str | None = None,
    ) -> "LearningEvent":
        """Create a new learning event."""
        return cls(
            id=str(uuid.uuid4()),
            event_type=event_type,
            learned_item_id=learned_item_id,
            timestamp=datetime.now(),
            details=details or {},
            user_action=user_action,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "learned_item_id": self.learned_item_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "user_action": self.user_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningEvent":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            learned_item_id=data["learned_item_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data.get("details", {}),
            user_action=data.get("user_action"),
        )


@dataclass
class LearningDashboardData:
    """Data for the learning transparency dashboard."""

    total_learned: int
    pending_approval: int
    recently_applied: list["LearnedItem"]
    recently_rejected: list[str]  # Item IDs
    by_category: dict[str, int]
    confidence_distribution: dict[str, int]  # low/medium/high buckets
    events_today: int
    guardrail_violations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_learned": self.total_learned,
            "pending_approval": self.pending_approval,
            "recently_applied": [item.to_dict() for item in self.recently_applied],
            "recently_rejected": self.recently_rejected,
            "by_category": self.by_category,
            "confidence_distribution": self.confidence_distribution,
            "events_today": self.events_today,
            "guardrail_violations": self.guardrail_violations,
        }


class LearningTransparency:
    """
    Provides full transparency into what Animus has learned.

    Features:
    - Complete learning history log
    - Dashboard data for UI
    - Export learning log
    - Audit trail for compliance
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._events: list[LearningEvent] = []
        self._load_events()

    def _load_events(self) -> None:
        """Load events from disk."""
        events_file = self.data_dir / "learning_events.json"
        if events_file.exists():
            try:
                with open(events_file) as f:
                    data = json.load(f)
                for item in data:
                    self._events.append(LearningEvent.from_dict(item))
                logger.info(f"Loaded {len(self._events)} learning events")
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.error(f"Failed to load learning events: {e}")

    def _save_events(self) -> None:
        """Save events to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        events_file = self.data_dir / "learning_events.json"
        # Keep last 1000 events
        recent_events = self._events[-1000:]
        data = [e.to_dict() for e in recent_events]
        with open(events_file, "w") as f:
            json.dump(data, f, indent=2)

    def log_event(
        self,
        event_type: str,
        learned_item_id: str,
        details: dict[str, Any] | None = None,
        user_action: str | None = None,
    ) -> LearningEvent:
        """
        Log a learning event.

        Args:
            event_type: Type of event
            learned_item_id: ID of related learned item
            details: Optional event details
            user_action: Optional user action that triggered event

        Returns:
            The created event
        """
        event = LearningEvent.create(
            event_type=event_type,
            learned_item_id=learned_item_id,
            details=details,
            user_action=user_action,
        )
        self._events.append(event)
        self._save_events()

        logger.debug(f"Logged learning event: {event_type} for {learned_item_id}")
        return event

    def get_dashboard_data(
        self,
        learned_items: list["LearnedItem"],
        pending_count: int = 0,
        violation_count: int = 0,
    ) -> LearningDashboardData:
        """
        Generate dashboard data.

        Args:
            learned_items: All learned items
            pending_count: Number of pending approvals
            violation_count: Number of guardrail violations

        Returns:
            Dashboard data object
        """
        # Count by category
        by_category: dict[str, int] = {}
        for item in learned_items:
            cat = item.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Confidence distribution
        confidence_dist = {"low": 0, "medium": 0, "high": 0}
        for item in learned_items:
            if item.confidence < 0.4:
                confidence_dist["low"] += 1
            elif item.confidence < 0.7:
                confidence_dist["medium"] += 1
            else:
                confidence_dist["high"] += 1

        # Recently applied (last 10)
        applied = [item for item in learned_items if item.applied]
        applied.sort(key=lambda x: x.updated_at, reverse=True)
        recently_applied = applied[:10]

        # Recently rejected (from events)
        today = datetime.now().date()
        rejected_ids = [
            e.learned_item_id
            for e in self._events
            if e.event_type == "rejected" and e.timestamp.date() == today
        ]

        # Events today
        events_today = sum(1 for e in self._events if e.timestamp.date() == today)

        return LearningDashboardData(
            total_learned=len([i for i in learned_items if i.applied]),
            pending_approval=pending_count,
            recently_applied=recently_applied,
            recently_rejected=rejected_ids[:10],
            by_category=by_category,
            confidence_distribution=confidence_dist,
            events_today=events_today,
            guardrail_violations=violation_count,
        )

    def get_history(
        self,
        limit: int = 100,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> list[LearningEvent]:
        """
        Get learning event history.

        Args:
            limit: Maximum events to return
            event_type: Optional type filter
            since: Optional date filter

        Returns:
            List of events
        """
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def export_log(self, format: str = "json") -> str:
        """
        Export complete learning log.

        Args:
            format: Export format (json or jsonl)

        Returns:
            Exported log as string
        """
        if format == "jsonl":
            lines = [json.dumps(e.to_dict()) for e in self._events]
            return "\n".join(lines)
        else:
            return json.dumps([e.to_dict() for e in self._events], indent=2)

    def explain_learning(self, learned_item: "LearnedItem") -> str:
        """
        Generate human-readable explanation of a learning.

        Args:
            learned_item: The learned item to explain

        Returns:
            Human-readable explanation
        """
        # Get related events
        related_events = [e for e in self._events if e.learned_item_id == learned_item.id]
        related_events.sort(key=lambda x: x.timestamp)

        explanation_parts = [
            f"Learning: {learned_item.content}",
            f"Category: {learned_item.category.value}",
            f"Confidence: {learned_item.confidence:.0%}",
            f"Status: {'Applied' if learned_item.applied else 'Pending'}",
            f"Based on {len(learned_item.evidence)} observations",
            "",
            "Timeline:",
        ]

        for event in related_events:
            time_str = event.timestamp.strftime("%Y-%m-%d %H:%M")
            explanation_parts.append(f"  - {time_str}: {event.event_type}")
            if event.user_action:
                explanation_parts.append(f"    User action: {event.user_action}")

        return "\n".join(explanation_parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get transparency statistics."""
        by_type: dict[str, int] = {}
        for event in self._events:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1

        # Events over time (last 7 days)
        daily_counts: dict[str, int] = {}
        for i in range(7):
            day = (datetime.now() - timedelta(days=i)).date()
            day_str = day.isoformat()
            daily_counts[day_str] = sum(1 for e in self._events if e.timestamp.date() == day)

        return {
            "total_events": len(self._events),
            "by_type": by_type,
            "daily_counts": daily_counts,
        }
