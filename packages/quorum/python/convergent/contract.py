"""
Formal coordination contract for the Intent Graph.

This module defines the machine-readable specification of the intent graph's
semantics. A second client can be implemented solely from this contract without
reading the rest of the codebase.

The contract specifies:
  - Node types, edge types, and their relationships
  - Graph invariants that must always hold
  - Allowed mutations and their preconditions
  - Conflict classification and resolution policies
  - Stability computation weights (deterministic)
  - Versioning rules (snapshots, branching, merges)

Contract version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    EvidenceKind,
    Intent,
    InterfaceKind,
)

# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


class EdgeType(str, Enum):
    """Directed edges in the intent graph.

    Each IntentNode participates in edges via its fields:
      - PROVIDES: node.provides[] → InterfaceSpec (outbound capability)
      - REQUIRES: node.requires[] → InterfaceSpec (inbound dependency)
      - CONSTRAINS: node.constraints[] → other nodes via tag matching
      - SUPERSEDES: node.parent_id → previous version of this intent
    """

    PROVIDES = "provides"
    REQUIRES = "requires"
    CONSTRAINS = "constrains"
    SUPERSEDES = "supersedes"


# ---------------------------------------------------------------------------
# Conflict classification
# ---------------------------------------------------------------------------


class ConflictClass(str, Enum):
    """Classification of conflicts by resolution strategy.

    HARD_FAIL: Critical invariant violation. Processing must stop.
               Examples: Critical constraint violated, type safety broken.

    AUTO_RESOLVE: Resolvable by deterministic policy (stability ordering).
                  Examples: Duplicate provisions, preferred constraint conflicts.

    HUMAN_ESCALATION: Requires human decision. Cannot be auto-resolved.
                      Examples: Equal-stability provision conflict,
                      conflicting Required constraints with no stability winner.
    """

    HARD_FAIL = "hard_fail"
    AUTO_RESOLVE = "auto_resolve"
    HUMAN_ESCALATION = "human_escalation"


# ---------------------------------------------------------------------------
# Graph invariants
# ---------------------------------------------------------------------------


class GraphInvariant(str, Enum):
    """Invariants that the intent graph must maintain at all times.

    APPEND_ONLY: Published intents are never deleted or modified in-place.
                 New versions are published with parent_id referencing the old.

    UNIQUE_IDS: Every intent has a globally unique ID (UUID v4).

    DETERMINISTIC_STABILITY: Given the same evidence set, stability computation
                             always produces the same score. No randomness.

    STABLE_ATTRACTORS: Higher-stability intents dominate lower-stability ones
                       in conflict resolution. This is monotonic: once an intent
                       reaches high stability, only new evidence can change it.

    CAUSAL_ORDERING: An intent with parent_id=X can only be published after
                     intent X exists in the graph. Timestamps are monotonically
                     increasing per agent.

    SELF_EXCLUSION: An agent's intents never conflict with its own other intents
                    during resolution. Self-overlap is evolution, not conflict.
    """

    APPEND_ONLY = "append_only"
    UNIQUE_IDS = "unique_ids"
    DETERMINISTIC_STABILITY = "deterministic_stability"
    STABLE_ATTRACTORS = "stable_attractors"
    CAUSAL_ORDERING = "causal_ordering"
    SELF_EXCLUSION = "self_exclusion"


# ---------------------------------------------------------------------------
# Allowed mutations
# ---------------------------------------------------------------------------


class MutationType(str, Enum):
    """The only mutations allowed on the intent graph.

    PUBLISH: Append a new IntentNode. The only write operation.
             Preconditions:
               - ID must be unique (not already in graph)
               - If parent_id is set, parent must exist in graph
               - agent_id must be non-empty
               - At least one of provides/requires/constraints must be non-empty

    ADD_EVIDENCE: Append evidence to an existing intent's evidence list.
                  This is the only in-place modification allowed, because
                  evidence is append-only within an intent. The intent itself
                  is not replaced — its evidence list grows.
                  Preconditions:
                    - Target intent must exist
                    - Evidence timestamp >= intent timestamp
    """

    PUBLISH = "publish"
    ADD_EVIDENCE = "add_evidence"


# ---------------------------------------------------------------------------
# Stability weights (deterministic computation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityWeights:
    """Weights for deterministic stability computation.

    The stability score is computed as:
        score  = base
        score += min(test_pass_count * test_pass, test_pass_cap)
        score += code_committed   (if any CODE_COMMITTED evidence)
        score += min(consumed_count * consumed_by_other, consumed_cap)
        score -= conflict_count * conflict_penalty
        score -= test_fail_count * test_fail_penalty
        score += manual_approval  (if any MANUAL_APPROVAL evidence)
        score  = clamp(score, 0.0, 1.0)

    This is a pure function of the evidence list. No randomness, no ordering
    dependence, no external state.
    """

    base: float = 0.3
    test_pass: float = 0.05
    test_pass_cap: float = 0.3
    code_committed: float = 0.2
    consumed_by_other: float = 0.1
    consumed_cap: float = 0.2
    conflict_penalty: float = 0.15
    test_fail_penalty: float = 0.15
    manual_approval: float = 0.3

    def compute(self, evidence: list[Evidence]) -> float:
        """Compute stability score from evidence. Deterministic."""
        score = self.base

        test_passes = sum(1 for e in evidence if e.kind == EvidenceKind.TEST_PASS)
        score += min(test_passes * self.test_pass, self.test_pass_cap)

        if any(e.kind == EvidenceKind.CODE_COMMITTED for e in evidence):
            score += self.code_committed

        dependents = sum(1 for e in evidence if e.kind == EvidenceKind.CONSUMED_BY_OTHER)
        score += min(dependents * self.consumed_by_other, self.consumed_cap)

        conflicts = sum(1 for e in evidence if e.kind == EvidenceKind.CONFLICT)
        score -= conflicts * self.conflict_penalty

        test_fails = sum(1 for e in evidence if e.kind == EvidenceKind.TEST_FAIL)
        score -= test_fails * self.test_fail_penalty

        if any(e.kind == EvidenceKind.MANUAL_APPROVAL for e in evidence):
            score += self.manual_approval

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Resolution policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionPolicy:
    """Policy governing how conflicts are classified and resolved.

    Conflict classification rules (applied in order):

    1. Critical constraint violation → HARD_FAIL
       If a Critical-severity constraint is violated, the system must halt.

    2. Required constraint conflict with stability gap → AUTO_RESOLVE
       If two Required constraints conflict but one source has strictly
       higher stability, the higher-stability constraint wins.

    3. Duplicate provision with stability gap → AUTO_RESOLVE
       When two agents provide the same interface, the higher-stability
       agent keeps the provision; the lower-stability agent receives a
       ConsumeInstead adjustment.

    4. Equal stability conflicts → HUMAN_ESCALATION
       When stability difference is within `stability_tie_epsilon`,
       the conflict cannot be auto-resolved.

    5. Preferred constraint conflict → AUTO_RESOLVE
       Preferred-severity constraints always auto-resolve (ignored if
       conflicting).
    """

    stability_tie_epsilon: float = 0.01
    severity_to_class: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.severity_to_class:
            # Use object.__setattr__ since this is a frozen dataclass
            object.__setattr__(
                self,
                "severity_to_class",
                {
                    ConstraintSeverity.CRITICAL.value: ConflictClass.HARD_FAIL.value,
                    ConstraintSeverity.REQUIRED.value: ConflictClass.AUTO_RESOLVE.value,
                    ConstraintSeverity.PREFERRED.value: ConflictClass.AUTO_RESOLVE.value,
                },
            )

    def classify_constraint_conflict(
        self,
        constraint: Constraint,
        my_stability: float,
        their_stability: float,
    ) -> ConflictClass:
        """Classify a constraint conflict.

        Args:
            constraint: The conflicting constraint.
            my_stability: Stability of the intent being resolved.
            their_stability: Stability of the intent that imposed the constraint.

        Returns:
            The conflict class determining resolution strategy.
        """
        # Rule 1: Critical → always HARD_FAIL
        if constraint.severity == ConstraintSeverity.CRITICAL:
            return ConflictClass.HARD_FAIL

        # Rule 5: Preferred → always AUTO_RESOLVE
        if constraint.severity == ConstraintSeverity.PREFERRED:
            return ConflictClass.AUTO_RESOLVE

        # Rules 2/4: Required with stability comparison
        stability_gap = abs(their_stability - my_stability)
        if stability_gap <= self.stability_tie_epsilon:
            return ConflictClass.HUMAN_ESCALATION

        return ConflictClass.AUTO_RESOLVE

    def classify_provision_conflict(
        self,
        my_stability: float,
        their_stability: float,
    ) -> ConflictClass:
        """Classify a duplicate-provision conflict.

        Args:
            my_stability: Stability of the resolving intent.
            their_stability: Stability of the existing intent.

        Returns:
            AUTO_RESOLVE if there is a clear stability winner,
            HUMAN_ESCALATION if too close to call.
        """
        stability_gap = abs(their_stability - my_stability)
        if stability_gap <= self.stability_tie_epsilon:
            return ConflictClass.HUMAN_ESCALATION
        return ConflictClass.AUTO_RESOLVE


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ContractViolation(Exception):
    """Raised when a graph operation violates the contract."""

    def __init__(self, invariant: GraphInvariant, message: str) -> None:
        self.invariant = invariant
        super().__init__(f"Contract violation [{invariant.value}]: {message}")


def validate_publish(
    intent: Intent,
    existing_ids: set[str],
    agent_timestamps: dict[str, str] | None = None,
) -> list[ContractViolation]:
    """Validate that publishing an intent would not violate the contract.

    Returns a list of violations (empty = valid).
    """
    violations: list[ContractViolation] = []

    # UNIQUE_IDS: ID must not already exist
    if intent.id in existing_ids:
        violations.append(
            ContractViolation(
                GraphInvariant.UNIQUE_IDS,
                f"Intent ID '{intent.id}' already exists in graph",
            )
        )

    # CAUSAL_ORDERING: parent must exist if specified
    if intent.parent_id is not None and intent.parent_id not in existing_ids:
        violations.append(
            ContractViolation(
                GraphInvariant.CAUSAL_ORDERING,
                f"Parent ID '{intent.parent_id}' does not exist in graph",
            )
        )

    # Must have content: at least one of provides/requires/constraints
    if not intent.provides and not intent.requires and not intent.constraints:
        violations.append(
            ContractViolation(
                GraphInvariant.APPEND_ONLY,
                "Intent must have at least one provides, requires, or constraints entry",
            )
        )

    # agent_id must be non-empty
    if not intent.agent_id:
        violations.append(
            ContractViolation(
                GraphInvariant.APPEND_ONLY,
                "Intent must have a non-empty agent_id",
            )
        )

    return violations


# ---------------------------------------------------------------------------
# Content hashing (for deterministic replay verification)
# ---------------------------------------------------------------------------


def content_hash_intent(intent: Intent) -> str:
    """Compute a deterministic content hash for an intent.

    The hash covers all semantically meaningful fields but excludes
    timestamps (which vary between replays). Evidence descriptions
    and kinds are included; evidence timestamps are excluded.
    """
    canonical = {
        "id": intent.id,
        "agent_id": intent.agent_id,
        "intent": intent.intent,
        "provides": [_spec_canonical(s) for s in intent.provides],
        "requires": [_spec_canonical(s) for s in intent.requires],
        "constraints": [_constraint_canonical(c) for c in intent.constraints],
        "stability": intent.stability,
        "evidence": [_evidence_canonical(e) for e in intent.evidence],
        "parent_id": intent.parent_id,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_hash_intents(intents: list[Intent]) -> str:
    """Compute a deterministic hash over a sorted list of intents.

    Intents are sorted by ID before hashing to ensure order-independence.
    """
    hashes = sorted(content_hash_intent(i) for i in intents)
    combined = "|".join(hashes)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _spec_canonical(spec: Any) -> dict[str, Any]:
    return {
        "name": spec.name,
        "kind": spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind),
        "signature": spec.signature,
        "module_path": spec.module_path,
        "tags": sorted(spec.tags),
    }


def _constraint_canonical(c: Any) -> dict[str, Any]:
    return {
        "target": c.target,
        "requirement": c.requirement,
        "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity),
        "affects_tags": sorted(c.affects_tags),
    }


def _evidence_canonical(e: Any) -> dict[str, Any]:
    return {
        "kind": e.kind.value if hasattr(e.kind, "value") else str(e.kind),
        "description": e.description,
    }


# ---------------------------------------------------------------------------
# The default contract
# ---------------------------------------------------------------------------


DEFAULT_STABILITY_WEIGHTS = StabilityWeights()

DEFAULT_RESOLUTION_POLICY = ResolutionPolicy()


@dataclass(frozen=True)
class IntentGraphContract:
    """The formal contract for the Convergent intent graph.

    This is the complete, self-contained specification. A compliant
    implementation must:

    1. Support all node_types and edge_types
    2. Maintain all invariants at all times
    3. Only allow mutations listed in allowed_mutations
    4. Use the stability_weights for deterministic scoring
    5. Follow the resolution_policy for conflict handling
    6. Produce identical results under deterministic replay

    Version history:
      1.0.0 — Initial formal contract
    """

    version: str = "1.0.0"

    node_types: tuple[str, ...] = tuple(k.value for k in InterfaceKind)
    edge_types: tuple[str, ...] = tuple(e.value for e in EdgeType)
    invariants: tuple[str, ...] = tuple(i.value for i in GraphInvariant)
    allowed_mutations: tuple[str, ...] = tuple(m.value for m in MutationType)
    evidence_kinds: tuple[str, ...] = tuple(e.value for e in EvidenceKind)
    constraint_severities: tuple[str, ...] = tuple(s.value for s in ConstraintSeverity)
    conflict_classes: tuple[str, ...] = tuple(c.value for c in ConflictClass)

    stability_weights: StabilityWeights = field(default_factory=StabilityWeights)
    resolution_policy: ResolutionPolicy = field(default_factory=ResolutionPolicy)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract to a JSON-compatible dict.

        This is the canonical machine-readable form that a second
        client implementation would consume.
        """
        return {
            "contract_version": self.version,
            "node_types": list(self.node_types),
            "edge_types": list(self.edge_types),
            "invariants": list(self.invariants),
            "allowed_mutations": list(self.allowed_mutations),
            "evidence_kinds": list(self.evidence_kinds),
            "constraint_severities": list(self.constraint_severities),
            "conflict_classes": list(self.conflict_classes),
            "stability_weights": {
                "base": self.stability_weights.base,
                "test_pass": self.stability_weights.test_pass,
                "test_pass_cap": self.stability_weights.test_pass_cap,
                "code_committed": self.stability_weights.code_committed,
                "consumed_by_other": self.stability_weights.consumed_by_other,
                "consumed_cap": self.stability_weights.consumed_cap,
                "conflict_penalty": self.stability_weights.conflict_penalty,
                "test_fail_penalty": self.stability_weights.test_fail_penalty,
                "manual_approval": self.stability_weights.manual_approval,
            },
            "resolution_policy": {
                "stability_tie_epsilon": self.resolution_policy.stability_tie_epsilon,
                "rules": [
                    {
                        "condition": "Critical constraint violated",
                        "class": ConflictClass.HARD_FAIL.value,
                        "action": "Halt processing. Cannot proceed.",
                    },
                    {
                        "condition": "Required constraint conflict with stability gap",
                        "class": ConflictClass.AUTO_RESOLVE.value,
                        "action": "Higher-stability constraint wins.",
                    },
                    {
                        "condition": "Duplicate provision with stability gap",
                        "class": ConflictClass.AUTO_RESOLVE.value,
                        "action": "Higher-stability agent keeps provision; "
                        "lower receives ConsumeInstead.",
                    },
                    {
                        "condition": "Equal-stability conflict "
                        f"(gap <= {DEFAULT_RESOLUTION_POLICY.stability_tie_epsilon})",
                        "class": ConflictClass.HUMAN_ESCALATION.value,
                        "action": "Surface to human. Cannot auto-resolve.",
                    },
                    {
                        "condition": "Preferred constraint conflict",
                        "class": ConflictClass.AUTO_RESOLVE.value,
                        "action": "Preferred constraints are advisory; conflict is ignored.",
                    },
                ],
            },
            "matching_rules": {
                "name_overlap": {
                    "description": "Two names overlap if, after normalization "
                    "(lowercase, strip Model/Service/Handler/Controller/"
                    "Spec/Interface suffix, CamelCase split), one is "
                    "equal to, a prefix of, or contained in the other.",
                },
                "tag_overlap": {
                    "description": "Two interface specs overlap if they share 2 or more tags.",
                    "threshold": 2,
                },
                "signature_compatibility": {
                    "description": "Signature B is compatible with A if B's fields "
                    "are a superset of A's fields with type normalization "
                    "(UUID↔uuid, String↔str, i64↔int, f64↔float, "
                    "Optional[X]↔X, Vec<X>↔list[X]).",
                },
                "constraint_applicability": {
                    "description": "A constraint applies to an intent if any of the "
                    "intent's interface tags overlap with the constraint's "
                    "affects_tags.",
                },
                "constraint_conflict": {
                    "description": "Two constraints conflict if their normalized targets "
                    "are equal but their requirements differ.",
                },
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the contract to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# The singleton default contract
DEFAULT_CONTRACT = IntentGraphContract()
