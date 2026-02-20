"""
Layer 1: Constraint Engine — Hard Truth.

Tests, types, schemas, invariants, security policies as first-class
coordination primitives. Unlike the intent graph (which captures decisions
and alignment), the constraint engine enforces correctness.

The constraint engine is the gatekeeper: no intent can be published or
merged unless it satisfies all applicable hard constraints. This prevents
the graph from becoming a "junkyard of inconsistent assertions."

Constraint kinds:
  TYPE_CHECK: Signature/type constraints (fields must have specific types)
  TEST_GATE: Evidence requirements (must have test passes before merge)
  SCHEMA_RULE: Data model constraints (required fields, FK relationships)
  INVARIANT: System-wide invariants (append-only, naming conventions)
  SECURITY_POLICY: Security rules (auth required, input validation)
  INTERFACE_CONTRACT: API contracts (endpoint signatures, versioning)

Validation is deterministic and machine-checkable. No LLM needed.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum

from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    Intent,
)
from convergent.matching import normalize_type, parse_signature

# ---------------------------------------------------------------------------
# Constraint kinds
# ---------------------------------------------------------------------------


class ConstraintKind(str, Enum):
    """Machine-checkable constraint categories."""

    TYPE_CHECK = "type_check"
    TEST_GATE = "test_gate"
    SCHEMA_RULE = "schema_rule"
    INVARIANT = "invariant"
    SECURITY_POLICY = "security_policy"
    INTERFACE_CONTRACT = "interface_contract"


# ---------------------------------------------------------------------------
# Typed constraint
# ---------------------------------------------------------------------------


@dataclass
class TypedConstraint:
    """A constraint with machine-checkable validation rules.

    Extends the base Constraint with structured validation:
    - required_fields: field->type pairs that must appear in signatures
    - forbidden_patterns: regex patterns that must NOT appear
    - required_evidence: evidence kinds that must be present
    - min_stability: minimum stability required for compliance
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: ConstraintKind = ConstraintKind.INVARIANT
    target: str = ""
    requirement: str = ""
    severity: ConstraintSeverity = ConstraintSeverity.REQUIRED
    affects_tags: list[str] = field(default_factory=list)
    # Machine-checkable rules
    required_fields: dict[str, str] = field(default_factory=dict)
    forbidden_patterns: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    min_stability: float = 0.0

    def to_base_constraint(self) -> Constraint:
        """Convert to a base Constraint for graph embedding."""
        return Constraint(
            target=self.target,
            requirement=self.requirement,
            severity=self.severity,
            affects_tags=self.affects_tags,
        )


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------


@dataclass
class ConstraintCheckResult:
    """Result of checking a single constraint against an intent."""

    constraint_id: str
    constraint_kind: ConstraintKind
    satisfied: bool
    violations: list[str] = field(default_factory=list)
    evidence_produced: list[Evidence] = field(default_factory=list)


@dataclass
class GateResult:
    """Result of gating an intent through all applicable constraints.

    The intent can only proceed if `passed` is True.
    """

    intent_id: str
    passed: bool
    check_results: list[ConstraintCheckResult] = field(default_factory=list)
    blocking_violations: list[str] = field(default_factory=list)

    @property
    def total_checks(self) -> int:
        return len(self.check_results)

    @property
    def satisfied_count(self) -> int:
        return sum(1 for r in self.check_results if r.satisfied)

    @property
    def violated_count(self) -> int:
        return sum(1 for r in self.check_results if not r.satisfied)


# ---------------------------------------------------------------------------
# Constraint engine
# ---------------------------------------------------------------------------


class ConstraintEngine:
    """Validates intents against typed, machine-checkable constraints.

    This is the 'hard truth' layer. Constraints are registered once and
    automatically enforced on every intent that matches their tags.

    Usage:
        engine = ConstraintEngine()

        # Register typed constraints
        engine.register(TypedConstraint(
            kind=ConstraintKind.SCHEMA_RULE,
            target="User model",
            requirement="must have id: UUID, email: str",
            affects_tags=["user", "model"],
            required_fields={"id": "UUID", "email": "str"},
        ))

        engine.register(TypedConstraint(
            kind=ConstraintKind.TEST_GATE,
            target="all models",
            requirement="must have passing tests before merge",
            affects_tags=["model"],
            required_evidence=["test_pass"],
        ))

        # Gate an intent
        result = engine.gate(intent)
        if not result.passed:
            # Intent blocked — violations explain why
            for v in result.blocking_violations:
                print(v)
    """

    def __init__(self) -> None:
        self._constraints: dict[str, TypedConstraint] = {}

    def register(self, constraint: TypedConstraint) -> str:
        """Register a typed constraint. Returns constraint ID."""
        self._constraints[constraint.id] = constraint
        return constraint.id

    def unregister(self, constraint_id: str) -> bool:
        """Remove a constraint. Returns True if it existed."""
        return self._constraints.pop(constraint_id, None) is not None

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)

    def constraints_for(self, intent: Intent) -> list[TypedConstraint]:
        """Find all constraints that apply to an intent (by tag matching)."""
        intent_tags: set[str] = set()
        for spec in intent.provides + intent.requires:
            intent_tags.update(spec.tags)

        applicable: list[TypedConstraint] = []
        for c in self._constraints.values():
            if set(c.affects_tags) & intent_tags:
                applicable.append(c)
        return applicable

    def check(self, constraint: TypedConstraint, intent: Intent) -> ConstraintCheckResult:
        """Check a single constraint against an intent."""
        violations: list[str] = []
        evidence: list[Evidence] = []

        # --- TYPE_CHECK / SCHEMA_RULE: Validate required fields ---
        if constraint.required_fields:
            violations.extend(_check_required_fields(constraint, intent))

        # --- SECURITY_POLICY / INVARIANT: Check forbidden patterns ---
        if constraint.forbidden_patterns:
            violations.extend(_check_forbidden_patterns(constraint, intent))

        # --- TEST_GATE: Check required evidence ---
        if constraint.required_evidence:
            violations.extend(_check_required_evidence(constraint, intent))

        # --- Minimum stability ---
        if constraint.min_stability > 0:
            actual = intent.compute_stability()
            if actual < constraint.min_stability:
                violations.append(
                    f"Stability {actual:.2f} below required {constraint.min_stability:.2f}"
                )

        satisfied = len(violations) == 0

        # Produce evidence of compliance
        if satisfied:
            evidence.append(
                Evidence.test_pass(
                    f"Constraint '{constraint.target}' ({constraint.kind.value}) satisfied"
                )
            )

        return ConstraintCheckResult(
            constraint_id=constraint.id,
            constraint_kind=constraint.kind,
            satisfied=satisfied,
            violations=violations,
            evidence_produced=evidence,
        )

    def gate(self, intent: Intent) -> GateResult:
        """Gate an intent through all applicable constraints.

        Returns GateResult with passed=True only if all Required/Critical
        constraints are satisfied. Preferred constraints produce warnings
        but don't block.
        """
        applicable = self.constraints_for(intent)
        results: list[ConstraintCheckResult] = []
        blocking: list[str] = []

        for constraint in applicable:
            result = self.check(constraint, intent)
            results.append(result)

            if not result.satisfied and constraint.severity in (
                ConstraintSeverity.REQUIRED,
                ConstraintSeverity.CRITICAL,
            ):
                for v in result.violations:
                    blocking.append(f"[{constraint.severity.value}] {constraint.target}: {v}")

        passed = len(blocking) == 0

        return GateResult(
            intent_id=intent.id,
            passed=passed,
            check_results=results,
            blocking_violations=blocking,
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_required_fields(constraint: TypedConstraint, intent: Intent) -> list[str]:
    """Check that intent's signatures contain required fields with correct types."""
    violations: list[str] = []

    # Collect all fields from all provides + requires signatures
    all_fields: dict[str, str] = {}
    for spec in intent.provides + intent.requires:
        parsed = parse_signature(spec.signature)
        all_fields.update(parsed)

    for req_field, req_type in constraint.required_fields.items():
        if req_field not in all_fields:
            violations.append(f"Missing required field '{req_field}: {req_type}'")
        elif normalize_type(all_fields[req_field]) != normalize_type(req_type):
            violations.append(
                f"Field '{req_field}' has type '{all_fields[req_field]}', expected '{req_type}'"
            )

    return violations


def _check_forbidden_patterns(constraint: TypedConstraint, intent: Intent) -> list[str]:
    """Check that no forbidden patterns appear in intent signatures or names."""
    violations: list[str] = []

    # Check against all interface names and signatures
    texts_to_check: list[str] = []
    for spec in intent.provides + intent.requires:
        texts_to_check.append(spec.name)
        texts_to_check.append(spec.signature)
        texts_to_check.append(spec.module_path)

    for pattern in constraint.forbidden_patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for text in texts_to_check:
            if compiled.search(text):
                violations.append(f"Forbidden pattern '{pattern}' found in '{text}'")
                break  # One match per pattern is enough

    return violations


def _check_required_evidence(constraint: TypedConstraint, intent: Intent) -> list[str]:
    """Check that intent has required evidence kinds."""
    violations: list[str] = []

    present_kinds = {e.kind.value for e in intent.evidence}
    for required in constraint.required_evidence:
        if required not in present_kinds:
            violations.append(f"Missing required evidence: {required}")

    return violations
