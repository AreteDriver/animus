"""Frozen dataclass models for the skill evolution system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class SkillMetrics:
    """Aggregated performance metrics for a skill+version over a time window."""

    skill_name: str
    skill_version: str
    period_start: str
    period_end: str
    total_invocations: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    trend: str = "stable"
    computed_at: str = ""

    def __post_init__(self) -> None:
        if not self.computed_at:
            object.__setattr__(self, "computed_at", datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class SkillChange:
    """A proposed modification to a skill definition."""

    skill_name: str
    old_version: str
    new_version: str
    change_type: str  # "tune", "generate", "deprecate"
    description: str = ""
    diff: str = ""
    modifications: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for an A/B test experiment."""

    experiment_id: str
    skill_name: str
    control_version: str
    variant_version: str
    traffic_split: float = 0.5
    min_invocations: int = 100
    start_date: str = ""

    def __post_init__(self) -> None:
        if not self.start_date:
            object.__setattr__(self, "start_date", datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class ExperimentResult:
    """Result of a concluded A/B test experiment."""

    experiment_id: str
    skill_name: str
    control_version: str
    variant_version: str
    control_metrics: SkillMetrics | None = None
    variant_metrics: SkillMetrics | None = None
    winner: str = ""  # version string of the winner
    statistical_significance: float = 0.0
    conclusion_reason: str = ""


@dataclass(frozen=True)
class DeprecationRecord:
    """Lifecycle record for a deprecated skill."""

    skill_name: str
    status: str = "flagged"  # flagged, deprecated, retired
    flagged_at: str = ""
    deprecated_at: str = ""
    retired_at: str = ""
    reason: str = ""
    success_rate_at_flag: float = 0.0
    invocations_at_flag: int = 0
    replacement_skill: str = ""
    approval_id: str = ""

    def __post_init__(self) -> None:
        if not self.flagged_at:
            object.__setattr__(self, "flagged_at", datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class CapabilityGap:
    """A detected gap in the skill library's coverage."""

    description: str
    failure_contexts: list[str] = field(default_factory=list)
    suggested_category: str = ""
    suggested_agent: str = ""
    confidence: float = 0.0
