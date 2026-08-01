"""Improvement Proposal Standard Schema.

Every proposal produced by the Architect Citizen follows this schema,
creating a searchable, auditable history of every architectural decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ProposalStatus(str, Enum):
    """Lifecycle stages of an improvement proposal."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    COMMISSIONED = "commissioned"
    IMPLEMENTING = "implementing"
    IMPLEMENTED = "implemented"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"


class ProposalConfidence(str, Enum):
    """Confidence levels for proposals."""

    VERY_HIGH = "very_high"  # ≥ 0.9
    HIGH = "high"  # ≥ 0.75
    MEDIUM = "medium"  # ≥ 0.5
    LOW = "low"  # ≥ 0.25
    VERY_LOW = "very_low"  # < 0.25


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting a proposal."""

    source: str  # e.g., "conversation_log", "code_analysis", "benchmark"
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAssessment:
    """Potential risk identified for a proposal."""

    description: str
    severity: str  # "critical", "high", "medium", "low"
    mitigation: str
    probability: float = 0.5  # 0.0–1.0


@dataclass
class ImprovementProposal:
    """Standard schema for all Architect improvement proposals.

    This is the single artifact that flows through the entire
    governance pipeline:

        Architect → Human Review → Forge → Evidence → Merge

    Every field should be populated with specific, verifiable information.
    Avoid vague claims like "improves performance" — instead:
    "Reduces p99 latency from 450ms to 200ms per benchmark X."
    """

    # Identity
    id: str
    title: str

    # Problem statement
    problem: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    root_cause: str = ""

    # Recommendation
    recommendation: str = ""
    alternatives_considered: list[str] = field(default_factory=list)
    expected_benefits: str = ""

    # Risk analysis
    potential_risks: list[RiskAssessment] = field(default_factory=list)

    # Estimation
    confidence_score: float = 0.5  # 0.0–1.0
    confidence_label: ProposalConfidence = field(default_factory=lambda: ProposalConfidence.MEDIUM)
    estimated_effort_hours: float = 0.0
    affected_components: list[str] = field(default_factory=list)

    # Validation plan
    evaluation_plan: str = ""
    rollback_plan: str = ""
    success_metrics: list[str] = field(default_factory=list)

    # Lifecycle
    status: ProposalStatus = field(default_factory=lambda: ProposalStatus.DRAFT)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    approved_by: str | None = None
    approved_at: datetime | None = None
    implemented_at: datetime | None = None

    # Evidence bundle (populated after implementation)
    evidence_bundle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage and transmission."""
        from dataclasses import asdict

        data = asdict(self)

        # Convert datetime objects to ISO strings for JSON serialization
        def _serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj

        return _serialize(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImprovementProposal:
        """Deserialize from dictionary."""
        from datetime import datetime as _dt

        # Handle nested dataclasses
        evidence = [EvidenceItem(**e) for e in data.get("evidence", [])]
        risks = [RiskAssessment(**r) for r in data.get("potential_risks", [])]

        # Handle enums
        status = ProposalStatus(data.get("status", "draft"))
        confidence_label = ProposalConfidence(data.get("confidence_label", "medium"))

        # Parse ISO datetime strings back to datetime objects
        datetime_fields = {"created_at", "updated_at", "approved_at", "implemented_at", "timestamp"}
        for field_name in datetime_fields:
            if field_name in data and isinstance(data[field_name], str):
                try:
                    data[field_name] = _dt.fromisoformat(data[field_name])
                except ValueError:
                    pass

        # Filter out handled keys
        filtered = {
            k: v
            for k, v in data.items()
            if k not in ("evidence", "potential_risks", "status", "confidence_label")
        }

        return cls(
            **filtered,
            evidence=evidence,
            potential_risks=risks,
            status=status,
            confidence_label=confidence_label,
        )

    @property
    def confidence(self) -> ProposalConfidence:
        """Map numeric score to label."""
        if self.confidence_score >= 0.9:
            return ProposalConfidence.VERY_HIGH
        elif self.confidence_score >= 0.75:
            return ProposalConfidence.HIGH
        elif self.confidence_score >= 0.5:
            return ProposalConfidence.MEDIUM
        elif self.confidence_score >= 0.25:
            return ProposalConfidence.LOW
        else:
            return ProposalConfidence.VERY_LOW

    def update_status(self, status: ProposalStatus, actor: str | None = None) -> None:
        """Update status with timestamp tracking."""
        self.status = status
        self.updated_at = datetime.now()
        if status == ProposalStatus.APPROVED:
            self.approved_by = actor
            self.approved_at = self.updated_at
        elif status == ProposalStatus.IMPLEMENTED:
            self.implemented_at = self.updated_at
