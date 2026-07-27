"""Mission dataclasses for Research Citizen."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MissionState(str, Enum):
    """Lifecycle states of a research mission."""

    PENDING = "pending"
    RUNNING = "running"
    EVALUATING = "evaluating"
    EVIDENCE_COLLECTING = "evidence_collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"


@dataclass
class MissionConfig:
    """Configuration for commissioning a research mission."""

    objective: str
    eval_suite: str
    workflow_template: str
    max_iterations: int = 3
    min_pass_rate: float = 0.9
    max_variance: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "eval_suite": self.eval_suite,
            "workflow_template": self.workflow_template,
            "max_iterations": self.max_iterations,
            "min_pass_rate": self.min_pass_rate,
            "max_variance": self.max_variance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionConfig:
        return cls(
            objective=data["objective"],
            eval_suite=data["eval_suite"],
            workflow_template=data["workflow_template"],
            max_iterations=data.get("max_iterations", 3),
            min_pass_rate=data.get("min_pass_rate", 0.9),
            max_variance=data.get("max_variance", 0.1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MissionRecord:
    """Persistent record of a mission execution."""

    id: str
    config: MissionConfig
    state: MissionState = MissionState.PENDING
    current_iteration: int = 0
    last_eval_run_id: str | None = None
    last_pass_rate: float = 0.0
    last_score_variance: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "state": self.state.value,
            "current_iteration": self.current_iteration,
            "last_eval_run_id": self.last_eval_run_id,
            "last_pass_rate": self.last_pass_rate,
            "last_score_variance": self.last_score_variance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error,
            "metadata": self.metadata,
        }
