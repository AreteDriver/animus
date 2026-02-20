"""
Python-native intent data models.
These provide a clean API and can delegate to the Rust core when available.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from convergent.matching import (
    names_overlap,
    normalize_constraint_target,
    signatures_compatible,
)


class InterfaceKind(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODEL = "model"
    ENDPOINT = "endpoint"
    MIGRATION = "migration"
    CONFIG = "config"


class ConstraintSeverity(str, Enum):
    PREFERRED = "preferred"
    REQUIRED = "required"
    CRITICAL = "critical"


class EvidenceKind(str, Enum):
    TEST_PASS = "test_pass"
    TEST_FAIL = "test_fail"
    CODE_COMMITTED = "code_committed"
    CONSUMED_BY_OTHER = "consumed_by"
    CONFLICT = "conflict"
    MANUAL_APPROVAL = "manual_approval"


@dataclass
class InterfaceSpec:
    """A typed interface that an agent provides or requires."""

    name: str
    kind: InterfaceKind
    signature: str
    module_path: str = ""
    tags: list[str] = field(default_factory=list)

    def structurally_overlaps(self, other: InterfaceSpec) -> bool:
        """Check if two interface specs likely refer to the same concept."""
        if names_overlap(self.name, other.name):
            return True
        shared_tags = set(self.tags) & set(other.tags)
        return len(shared_tags) >= 2

    def signature_compatible(self, other: InterfaceSpec) -> bool:
        """Check if signatures are compatible (superset with type normalization)."""
        return signatures_compatible(self.signature, other.signature)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "signature": self.signature,
            "module_path": self.module_path,
            "tags": self.tags,
        }


@dataclass
class Constraint:
    """A constraint that an agent's decision imposes on other scopes."""

    target: str
    requirement: str
    severity: ConstraintSeverity = ConstraintSeverity.REQUIRED
    affects_tags: list[str] = field(default_factory=list)

    def applies_to(self, intent: Intent) -> bool:
        """Check if this constraint applies to a given intent."""
        all_tags = set()
        for spec in intent.provides + intent.requires:
            all_tags.update(spec.tags)
        return bool(set(self.affects_tags) & all_tags)

    def conflicts_with(self, other: Constraint) -> bool:
        """Check if two constraints conflict (normalized target comparison)."""
        return (
            normalize_constraint_target(self.target) == normalize_constraint_target(other.target)
            and self.requirement != other.requirement
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "requirement": self.requirement,
            "affects_tags": self.affects_tags,
        }


@dataclass
class Evidence:
    """Evidence supporting or undermining an intent's stability."""

    kind: EvidenceKind
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def test_pass(cls, description: str) -> Evidence:
        return cls(kind=EvidenceKind.TEST_PASS, description=description)

    @classmethod
    def code_committed(cls, description: str) -> Evidence:
        return cls(kind=EvidenceKind.CODE_COMMITTED, description=description)

    @classmethod
    def consumed_by(cls, agent_id: str) -> Evidence:
        return cls(
            kind=EvidenceKind.CONSUMED_BY_OTHER,
            description=f"Consumed by agent {agent_id}",
        )

    @classmethod
    def conflict(cls, description: str) -> Evidence:
        return cls(kind=EvidenceKind.CONFLICT, description=description)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "description": self.description,
        }


@dataclass
class Intent:
    """A single unit of semantic intent in the shared graph."""

    agent_id: str
    intent: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provides: list[InterfaceSpec] = field(default_factory=list)
    requires: list[InterfaceSpec] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    stability: float = 0.3
    evidence: list[Evidence] = field(default_factory=list)
    parent_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for passing to Rust core."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "intent": self.intent,
            "provides": [s.to_dict() for s in self.provides],
            "requires": [s.to_dict() for s in self.requires],
            "constraints": [c.to_dict() for c in self.constraints],
            "stability": self.stability,
            "evidence": [e.to_dict() for e in self.evidence],
            "parent_id": self.parent_id,
        }

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def compute_stability(self) -> float:
        """Compute stability from evidence. Mirrors Rust StabilityScorer."""
        score = 0.3  # base

        test_passes = sum(1 for e in self.evidence if e.kind == EvidenceKind.TEST_PASS)
        score += min(test_passes * 0.05, 0.3)

        if any(e.kind == EvidenceKind.CODE_COMMITTED for e in self.evidence):
            score += 0.2

        dependents = sum(1 for e in self.evidence if e.kind == EvidenceKind.CONSUMED_BY_OTHER)
        score += min(dependents * 0.1, 0.2)

        conflicts = sum(1 for e in self.evidence if e.kind == EvidenceKind.CONFLICT)
        score -= conflicts * 0.15

        test_fails = sum(1 for e in self.evidence if e.kind == EvidenceKind.TEST_FAIL)
        score -= test_fails * 0.15

        if any(e.kind == EvidenceKind.MANUAL_APPROVAL for e in self.evidence):
            score += 0.3

        return max(0.0, min(1.0, score))


@dataclass
class Adjustment:
    """An adjustment the resolver recommends."""

    kind: str  # "ConsumeInstead", "AdoptConstraint", "YieldTo", "AdaptSignature"
    description: str
    source_intent_id: str
    confidence: float = 1.0


@dataclass
class ConflictReport:
    """A conflict between two intents."""

    my_intent_id: str
    their_intent_id: str
    description: str
    their_stability: float
    resolution_suggestion: str
    confidence: float = 1.0


@dataclass
class ResolutionResult:
    """Result of resolving an intent against the graph."""

    original_intent_id: str
    adjustments: list[Adjustment] = field(default_factory=list)
    conflicts: list[ConflictReport] = field(default_factory=list)
    adopted_constraints: list[Constraint] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.conflicts) == 0

    @property
    def has_adjustments(self) -> bool:
        return len(self.adjustments) > 0

    @property
    def min_confidence(self) -> float:
        """Return the lowest confidence score across all adjustments and conflicts."""
        scores = [a.confidence for a in self.adjustments] + [c.confidence for c in self.conflicts]
        return min(scores) if scores else 1.0

    def adjustments_above(self, threshold: float) -> list[Adjustment]:
        """Return only adjustments with confidence >= threshold."""
        return [a for a in self.adjustments if a.confidence >= threshold]
