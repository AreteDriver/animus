"""MissionOrder — bounded deployment contract for citizen missions.

A MissionOrder is the only way a citizen may leave the Animus core.
It defines authority, objectives, safety limits, and return conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any


class MissionStatus(Enum):
    """Lifecycle states for a mission."""

    DRAFT = auto()
    ISSUED = auto()
    SPAWNED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMED_OUT = auto()
    DEBRIEFED = auto()
    REJECTED = auto()


@dataclass
class MissionConstraint:
    """A single safety or authority constraint on a mission.

    Examples:
        MissionConstraint(name="max_tokens", value=10000, kind="budget")
        MissionConstraint(name="readonly", value=True, kind="authority")
        MissionConstraint(name="no_network", value=True, kind="safety")
    """

    name: str
    value: Any
    kind: str = "general"  # "budget", "authority", "safety", "time", "general"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "kind": self.kind,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionConstraint:
        return cls(
            name=data["name"],
            value=data["value"],
            kind=data.get("kind", "general"),
            description=data.get("description", ""),
        )


@dataclass
class MissionResult:
    """Outcome of a completed mission.

    Results are reintegrated into Animus memory during debrief.
    """

    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)  # file paths, URIs
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "outputs": self.outputs,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "errors": self.errors,
            "logs": self.logs,
            "completed_at": self.completed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionResult:
        return cls(
            success=data["success"],
            outputs=data.get("outputs", {}),
            artifacts=data.get("artifacts", []),
            metrics=data.get("metrics", {}),
            errors=data.get("errors", []),
            logs=data.get("logs", []),
            completed_at=datetime.fromisoformat(data["completed_at"]),
        )


@dataclass
class MissionOrder:
    """Bounded deployment contract allowing a citizen to execute on a remote body.

    Core rule: Citizens may leave the Animus core only through issued mission orders.
    """

    citizen_id: str
    mission_type: str  # e.g., "code_review", "threat_scan", "data_analysis"
    objectives: dict[str, Any] = field(default_factory=dict)
    constraints: list[MissionConstraint] = field(default_factory=list)
    authority_limits: list[str] = field(default_factory=list)  # e.g., ["read", "analysis"], NOT ["write", "deploy"]
    return_conditions: list[str] = field(default_factory=list)  # e.g., ["on_complete", "on_timeout", "on_error"]
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    priority: int = 5  # 1=highest, 10=lowest
    tags: list[str] = field(default_factory=list)
    # Runtime selection
    preferred_runtime: str = "local"  # "local", "adk", "langgraph", "openai", "ssh"
    # Set by MissionSystem on issue
    id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:12])
    status: MissionStatus = MissionStatus.DRAFT
    issued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: MissionResult | None = None

    def __post_init__(self):
        # Ensure readonly is default if no authority limits specified
        if not self.authority_limits:
            self.authority_limits = ["read", "analysis", "report"]
        # Ensure safe return conditions if none specified
        if not self.return_conditions:
            self.return_conditions = ["on_complete", "on_timeout", "on_error"]

    @property
    def is_active(self) -> bool:
        return self.status in (
            MissionStatus.ISSUED,
            MissionStatus.SPAWNED,
            MissionStatus.RUNNING,
            MissionStatus.PAUSED,
        )

    @property
    def has_debriefed(self) -> bool:
        return self.status == MissionStatus.DEBRIEFED

    def add_constraint(self, name: str, value: Any, kind: str = "general") -> MissionOrder:
        """Fluent method to add a constraint."""
        self.constraints.append(MissionConstraint(name=name, value=value, kind=kind))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "citizen_id": self.citizen_id,
            "mission_type": self.mission_type,
            "objectives": self.objectives,
            "constraints": [c.to_dict() for c in self.constraints],
            "authority_limits": self.authority_limits,
            "return_conditions": self.return_conditions,
            "timeout_seconds": self.timeout.total_seconds(),
            "priority": self.priority,
            "tags": self.tags,
            "preferred_runtime": self.preferred_runtime,
            "status": self.status.name,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionOrder:
        order = cls(
            id=data.get("id", __import__("uuid").uuid4().hex[:12]),
            citizen_id=data["citizen_id"],
            mission_type=data["mission_type"],
            objectives=data.get("objectives", {}),
            constraints=[
                MissionConstraint.from_dict(c) for c in data.get("constraints", [])
            ],
            authority_limits=data.get("authority_limits", []),
            return_conditions=data.get("return_conditions", []),
            timeout=timedelta(seconds=data.get("timeout_seconds", 1800)),
            priority=data.get("priority", 5),
            tags=data.get("tags", []),
            preferred_runtime=data.get("preferred_runtime", "local"),
            status=MissionStatus[data.get("status", "DRAFT")],
            issued_at=datetime.fromisoformat(data["issued_at"]) if data.get("issued_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )
        if data.get("result"):
            order.result = MissionResult.from_dict(data["result"])
        return order
