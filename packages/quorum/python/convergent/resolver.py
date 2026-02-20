"""
Intent Resolver — reads the shared intent graph and adjusts local plans
for compatibility. Not a coordinator — a lens.

Uses the Rust core when available for performance. Falls back to pure
Python implementation for development and testing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from convergent.intent import (
    Adjustment,
    ConflictReport,
    Constraint,
    Intent,
    InterfaceSpec,
    ResolutionResult,
)

if TYPE_CHECKING:
    from convergent.semantic import SemanticMatcher, TrajectoryPrediction

logger = logging.getLogger(__name__)


class GraphBackend(Protocol):
    """Protocol for intent graph backends (Rust or Python)."""

    def publish(self, intent: Intent) -> float: ...
    def query_all(self, min_stability: float | None = None) -> list[Intent]: ...
    def query_by_agent(self, agent_id: str) -> list[Intent]: ...
    def find_overlapping(
        self, specs: list[InterfaceSpec], exclude_agent: str, min_stability: float
    ) -> list[Intent]: ...
    def resolve(self, intent: Intent, min_stability: float) -> ResolutionResult: ...
    def count(self) -> int: ...


class PythonGraphBackend:
    """Pure Python intent graph for development and testing.
    Production should use the Rust-backed IntentGraph."""

    def __init__(self) -> None:
        self._intents: list[Intent] = []

    def publish(self, intent: Intent) -> float:
        """Publish intent and return computed stability."""
        stability = intent.compute_stability()
        self._intents.append(intent)
        logger.debug(
            f"Published intent '{intent.intent}' from {intent.agent_id} "
            f"(stability: {stability:.2f})"
        )
        return stability

    def query_all(self, min_stability: float | None = None) -> list[Intent]:
        min_stab = min_stability or 0.0
        return [i for i in self._intents if i.compute_stability() >= min_stab]

    def query_by_agent(self, agent_id: str) -> list[Intent]:
        return [i for i in self._intents if i.agent_id == agent_id]

    def find_overlapping(
        self, specs: list[InterfaceSpec], exclude_agent: str, min_stability: float
    ) -> list[Intent]:
        results = []
        for intent in self._intents:
            if intent.agent_id == exclude_agent:
                continue
            if intent.compute_stability() < min_stability:
                continue

            their_specs = intent.provides + intent.requires
            for my_spec in specs:
                if any(my_spec.structurally_overlaps(ts) for ts in their_specs):
                    results.append(intent)
                    break

        return results

    def count(self) -> int:
        return len(self._intents)


class IntentResolver:
    """Reads the shared intent graph and adjusts local plans for compatibility.

    This is the core intelligence of Convergent. Before each major decision,
    an agent runs its intent through the resolver to check for:

    1. Duplicate provisions (another agent already provides this)
    2. Interface mismatches (signatures don't match)
    3. Applicable constraints (other agents' decisions constrain mine)
    4. Conflicts (irreconcilable differences)

    Optionally enhanced with LLM-powered semantic matching when a
    SemanticMatcher is provided.
    """

    _VALID_EVENTS = frozenset({"publish", "resolve", "conflict"})

    def __init__(
        self,
        backend: GraphBackend | None = None,
        min_stability: float = 0.3,
        semantic_matcher: SemanticMatcher | None = None,
        semantic_confidence_threshold: float = 0.7,
    ) -> None:
        self.backend: GraphBackend = backend or PythonGraphBackend()
        self.min_stability = min_stability
        self.semantic_matcher = semantic_matcher
        self.semantic_confidence_threshold = semantic_confidence_threshold
        self._hooks: dict[str, list] = {e: [] for e in self._VALID_EVENTS}

    def add_hook(self, event: str, callback) -> None:
        """Register a callback for an event.

        Events:
            "publish"  — callback(intent: Intent, stability: float)
            "resolve"  — callback(intent: Intent, result: ResolutionResult)
            "conflict" — callback(intent: Intent, conflict: ConflictReport)

        Raises:
            ValueError: If event name is not recognized.
        """
        if event not in self._VALID_EVENTS:
            raise ValueError(f"Unknown event '{event}'. Valid events: {sorted(self._VALID_EVENTS)}")
        self._hooks[event].append(callback)

    def remove_hook(self, event: str, callback) -> None:
        """Remove a previously registered callback.

        Raises:
            ValueError: If event name is not recognized.
        """
        if event not in self._VALID_EVENTS:
            raise ValueError(f"Unknown event '{event}'. Valid events: {sorted(self._VALID_EVENTS)}")
        self._hooks[event] = [cb for cb in self._hooks[event] if cb is not callback]

    def _fire_hooks(self, event: str, *args) -> None:
        """Fire all callbacks for an event. Exceptions are logged and swallowed."""
        for callback in self._hooks[event]:
            try:
                callback(*args)
            except Exception:
                logger.exception("Hook %r raised an exception", event)

    def publish(self, intent: Intent) -> float:
        """Publish an intent to the shared graph. Returns computed stability."""
        stability = self.backend.publish(intent)
        self._fire_hooks("publish", intent, stability)
        return stability

    def resolve(self, intent: Intent) -> ResolutionResult:
        """Resolve an intent against the current graph state.

        Returns adjustments the agent should make for compatibility,
        any conflicts found, and constraints to adopt.
        """
        adjustments: list[Adjustment] = []
        conflicts: list[ConflictReport] = []
        adopted_constraints: list[Constraint] = []

        my_specs = intent.provides + intent.requires
        my_stability = intent.compute_stability()

        # ── 1. Structural overlap detection (Phase 1) ──────────────────
        overlapping = self.backend.find_overlapping(my_specs, intent.agent_id, self.min_stability)
        structurally_overlapping_ids: set[str] = {o.id for o in overlapping}

        for other in overlapping:
            other_stability = other.compute_stability()

            # Check for duplicate provisions
            for my_provision in intent.provides:
                for their_provision in other.provides:
                    if my_provision.structurally_overlaps(their_provision):
                        if other_stability > my_stability:
                            adjustments.append(
                                Adjustment(
                                    kind="ConsumeInstead",
                                    description=(
                                        f"Drop '{my_provision.name}', consume "
                                        f"'{their_provision.name}' from agent "
                                        f"{other.agent_id} (stability {other_stability:.2f})"
                                    ),
                                    source_intent_id=other.id,
                                    confidence=1.0,
                                )
                            )
                        else:
                            conflicts.append(
                                ConflictReport(
                                    my_intent_id=intent.id,
                                    their_intent_id=other.id,
                                    description=(
                                        f"Both provide '{my_provision.name}' — "
                                        f"my stability {my_stability:.2f} vs "
                                        f"their {other_stability:.2f}"
                                    ),
                                    their_stability=other_stability,
                                    resolution_suggestion=(
                                        "Higher stability should provide; other should consume"
                                    ),
                                    confidence=1.0,
                                )
                            )

            # Check for signature mismatches in required→provided pairs
            for my_req in intent.requires:
                for their_prov in other.provides:
                    if (
                        my_req.structurally_overlaps(their_prov)
                        and not my_req.signature_compatible(their_prov)
                        and other_stability > my_stability
                    ):
                        adjustments.append(
                            Adjustment(
                                kind="AdaptSignature",
                                description=(
                                    f"Adapt '{my_req.name}' signature to match "
                                    f"'{their_prov.name}' from agent {other.agent_id} — "
                                    f"expected '{my_req.signature}', "
                                    f"they provide '{their_prov.signature}'"
                                ),
                                source_intent_id=other.id,
                                confidence=1.0,
                            )
                        )

        # ── 2. LLM-enhanced overlap detection (Phase 2) ───────────────
        if self.semantic_matcher is not None:
            all_intents = self.backend.query_all(self.min_stability)
            non_overlapping = [
                o
                for o in all_intents
                if o.agent_id != intent.agent_id and o.id not in structurally_overlapping_ids
            ]

            # Build pairs for batch checking
            pairs: list[tuple[dict, dict]] = []
            pair_context: list[tuple[InterfaceSpec, InterfaceSpec, Intent]] = []
            for other in non_overlapping:
                for my_prov in intent.provides:
                    for their_prov in other.provides:
                        pairs.append((my_prov.to_dict(), their_prov.to_dict()))
                        pair_context.append((my_prov, their_prov, other))

            if pairs:
                matches = self.semantic_matcher.check_overlap_batch(pairs)
                for match, (my_prov, their_prov, other) in zip(matches, pair_context, strict=True):
                    if match.overlap and match.confidence >= self.semantic_confidence_threshold:
                        other_stability = other.compute_stability()
                        if other_stability > my_stability:
                            adjustments.append(
                                Adjustment(
                                    kind="ConsumeInstead",
                                    description=(
                                        f"Drop '{my_prov.name}', consume "
                                        f"'{their_prov.name}' from agent "
                                        f"{other.agent_id} (stability "
                                        f"{other_stability:.2f}) "
                                        f"[semantic: {match.reasoning}]"
                                    ),
                                    source_intent_id=other.id,
                                    confidence=match.confidence,
                                )
                            )
                        else:
                            conflicts.append(
                                ConflictReport(
                                    my_intent_id=intent.id,
                                    their_intent_id=other.id,
                                    description=(
                                        f"Both provide '{my_prov.name}' / "
                                        f"'{their_prov.name}' — "
                                        f"my stability {my_stability:.2f} vs "
                                        f"their {other_stability:.2f} "
                                        f"[semantic: {match.reasoning}]"
                                    ),
                                    their_stability=other_stability,
                                    resolution_suggestion=(
                                        "Higher stability should provide; other should consume"
                                    ),
                                    confidence=match.confidence,
                                )
                            )

        # ── 3. Structural constraint checking (Phase 1) ────────────────
        all_intents_for_constraints = self.backend.query_all(self.min_stability)
        structurally_applied_constraints: set[tuple[str, str]] = set()

        for other in all_intents_for_constraints:
            if other.agent_id == intent.agent_id:
                continue

            for constraint in other.constraints:
                if constraint.applies_to(intent):
                    constraint_key = (constraint.target, constraint.requirement)
                    structurally_applied_constraints.add(constraint_key)

                    # Check for conflicts with our constraints
                    has_conflict = any(mc.conflicts_with(constraint) for mc in intent.constraints)

                    if has_conflict:
                        other_stability = other.compute_stability()
                        conflicts.append(
                            ConflictReport(
                                my_intent_id=intent.id,
                                their_intent_id=other.id,
                                description=(f"Constraint conflict on '{constraint.target}'"),
                                their_stability=other_stability,
                                resolution_suggestion=("Higher stability constraint should win"),
                                confidence=1.0,
                            )
                        )
                    else:
                        adopted_constraints.append(constraint)
                        adjustments.append(
                            Adjustment(
                                kind="AdoptConstraint",
                                description=(
                                    f"Adopt constraint: {constraint.target} — "
                                    f"{constraint.requirement}"
                                ),
                                source_intent_id=other.id,
                                confidence=1.0,
                            )
                        )

        # ── 4. LLM-enhanced constraint checking (Phase 2) ─────────────
        if self.semantic_matcher is not None:
            for other in all_intents_for_constraints:
                if other.agent_id == intent.agent_id:
                    continue

                for constraint in other.constraints:
                    constraint_key = (constraint.target, constraint.requirement)
                    if constraint_key in structurally_applied_constraints:
                        continue  # Already handled structurally

                    result = self.semantic_matcher.check_constraint_applies(
                        constraint.to_dict(),
                        intent.to_dict(),
                    )
                    if result.applies and result.confidence >= self.semantic_confidence_threshold:
                        has_conflict = any(
                            mc.conflicts_with(constraint) for mc in intent.constraints
                        )

                        if has_conflict:
                            other_stability = other.compute_stability()
                            conflicts.append(
                                ConflictReport(
                                    my_intent_id=intent.id,
                                    their_intent_id=other.id,
                                    description=(
                                        f"Constraint conflict on '{constraint.target}' "
                                        f"[semantic: {result.reasoning}]"
                                    ),
                                    their_stability=other_stability,
                                    resolution_suggestion=(
                                        "Higher stability constraint should win"
                                    ),
                                    confidence=result.confidence,
                                )
                            )
                        else:
                            adopted_constraints.append(constraint)
                            adjustments.append(
                                Adjustment(
                                    kind="AdoptConstraint",
                                    description=(
                                        f"Adopt constraint: {constraint.target} — "
                                        f"{constraint.requirement} "
                                        f"[semantic: {result.reasoning}]"
                                    ),
                                    source_intent_id=other.id,
                                    confidence=result.confidence,
                                )
                            )

        resolution = ResolutionResult(
            original_intent_id=intent.id,
            adjustments=adjustments,
            conflicts=conflicts,
            adopted_constraints=adopted_constraints,
        )

        logger.info(
            f"Resolved intent '{intent.intent}' from {intent.agent_id}: "
            f"{len(adjustments)} adjustments, {len(conflicts)} conflicts"
        )

        self._fire_hooks("resolve", intent, resolution)
        for conflict in conflicts:
            self._fire_hooks("conflict", intent, conflict)

        return resolution

    def predict_trajectories(
        self, agent_ids: list[str] | None = None
    ) -> dict[str, TrajectoryPrediction]:
        """Predict future moves for agents based on their intent history.

        Returns empty dict if no semantic_matcher is configured.
        """
        if self.semantic_matcher is None:
            return {}

        all_intents = self.backend.query_all()
        agents: dict[str, list[Intent]] = {}
        for intent in all_intents:
            agents.setdefault(intent.agent_id, []).append(intent)

        target_ids = agent_ids if agent_ids is not None else list(agents.keys())
        results: dict[str, TrajectoryPrediction] = {}

        for aid in target_ids:
            history = agents.get(aid)
            if not history:
                continue
            prediction = self.semantic_matcher.predict_trajectory([i.to_dict() for i in history])
            results[aid] = prediction

        return results

    @property
    def intent_count(self) -> int:
        return self.backend.count()
