"""Notification models and event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class EventType(Enum):
    """Types of workflow events."""

    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    SCHEDULE_TRIGGERED = "schedule_triggered"


@dataclass
class NotificationEvent:
    """A notification event to send."""

    event_type: EventType
    workflow_name: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, success

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "workflow_name": self.workflow_name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "severity": self.severity,
        }
