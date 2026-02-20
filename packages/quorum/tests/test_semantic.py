"""
Tests for Phase 2: LLM-powered semantic matching.

All LLM calls are mocked — no API keys needed. Tests verify:
1. Semantic overlap detection catches synonym matches
2. Structural matches don't trigger unnecessary LLM calls
3. Confidence scoring works correctly
4. Trajectory prediction from agent history
5. Backward compatibility with semantic_matcher=None
"""

from __future__ import annotations

from typing import Any

from convergent.agent import SimulationRunner
from convergent.demo import build_agent_a, build_agent_b, build_agent_c
from convergent.intent import (
    Adjustment,
    ConflictReport,
    Constraint,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
    ResolutionResult,
)
from convergent.resolver import IntentResolver
from convergent.semantic import (
    ConstraintApplicability,
    SemanticMatch,
    SemanticMatcher,
    TrajectoryPrediction,
    _SemanticCache,
)

# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockSemanticMatcher:
    """Mock SemanticMatcher for testing — no LLM calls."""

    def __init__(self) -> None:
        self.overlap_results: dict[tuple[str, str], SemanticMatch] = {}
        self.constraint_results: dict[tuple[str, str], ConstraintApplicability] = {}
        self.trajectory_result: TrajectoryPrediction | None = None
        self.overlap_call_count = 0
        self.overlap_batch_call_count = 0
        self.constraint_call_count = 0
        self.trajectory_call_count = 0

    def check_overlap(self, spec_a: dict[str, Any], spec_b: dict[str, Any]) -> SemanticMatch:
        self.overlap_call_count += 1
        key = (spec_a.get("name", ""), spec_b.get("name", ""))
        return self.overlap_results.get(
            key, SemanticMatch(overlap=False, confidence=0.0, reasoning="no match configured")
        )

    def check_overlap_batch(
        self, pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[SemanticMatch]:
        self.overlap_batch_call_count += 1
        results = []
        for spec_a, spec_b in pairs:
            key = (spec_a.get("name", ""), spec_b.get("name", ""))
            results.append(
                self.overlap_results.get(
                    key,
                    SemanticMatch(overlap=False, confidence=0.0, reasoning="no match configured"),
                )
            )
        return results

    def check_constraint_applies(
        self, constraint: dict[str, Any], intent: dict[str, Any]
    ) -> ConstraintApplicability:
        self.constraint_call_count += 1
        key = (constraint.get("target", ""), intent.get("intent", ""))
        return self.constraint_results.get(
            key,
            ConstraintApplicability(applies=False, confidence=0.0, reasoning="no match"),
        )

    def predict_trajectory(self, agent_history: list[dict[str, Any]]) -> TrajectoryPrediction:
        self.trajectory_call_count += 1
        if self.trajectory_result is not None:
            return self.trajectory_result
        agent_id = agent_history[0].get("agent_id", "unknown") if agent_history else "unknown"
        return TrajectoryPrediction(agent_id=agent_id, confidence=0.0)


# Verify mock satisfies protocol at import time
assert isinstance(MockSemanticMatcher(), SemanticMatcher)


# ---------------------------------------------------------------------------
# Test: Semantic Overlap Detection
# ---------------------------------------------------------------------------


class TestSemanticOverlapDetection:
    """LLM-powered semantic overlap catches synonym matches."""

    def test_synonym_overlap_detected(self):
        """AccountManager ↔ UserHandler detected as semantic overlap."""
        matcher = MockSemanticMatcher()
        matcher.overlap_results[("UserHandler", "AccountManager")] = SemanticMatch(
            overlap=True,
            confidence=0.85,
            reasoning="AccountManager and UserHandler manage the same user domain",
        )

        resolver = IntentResolver(min_stability=0.0, semantic_matcher=matcher)

        # Agent A provides AccountManager with high stability
        intent_a = Intent(
            agent_id="agent-a",
            intent="Account management module",
            provides=[
                InterfaceSpec(
                    name="AccountManager",
                    kind=InterfaceKind.CLASS,
                    signature="create(email) -> User",
                    tags=["account"],
                ),
            ],
            evidence=[Evidence.code_committed("account.py"), Evidence.test_pass("t1")],
        )
        resolver.publish(intent_a)

        # Agent B provides UserHandler (synonym — different name, no structural overlap)
        intent_b = Intent(
            agent_id="agent-b",
            intent="User handling module",
            provides=[
                InterfaceSpec(
                    name="UserHandler",
                    kind=InterfaceKind.CLASS,
                    signature="handle(request) -> Response",
                    tags=["handler"],
                ),
            ],
        )

        result = resolver.resolve(intent_b)
        assert result.has_adjustments
        consume = [a for a in result.adjustments if a.kind == "ConsumeInstead"]
        assert len(consume) == 1
        assert consume[0].confidence == 0.85
        assert "semantic" in consume[0].description

    def test_structural_matches_skip_llm(self):
        """When structural matching already detects overlap, LLM is NOT called for that pair."""
        matcher = MockSemanticMatcher()
        resolver = IntentResolver(min_stability=0.0, semantic_matcher=matcher)

        # Both provide "User" — structural match
        intent_a = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID"),
            ],
            evidence=[Evidence.code_committed("auth.py")],
        )
        resolver.publish(intent_a)

        intent_b = Intent(
            agent_id="agent-b",
            intent="User module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: int"),
            ],
        )

        resolver.resolve(intent_b)
        # Structural overlap found → no LLM batch call needed (no non-overlapping intents)
        assert matcher.overlap_batch_call_count == 0

    def test_resolver_without_matcher_no_semantic(self):
        """Resolver with semantic_matcher=None behaves exactly like Phase 1."""
        resolver = IntentResolver(min_stability=0.0, semantic_matcher=None)

        intent_a = Intent(
            agent_id="agent-a",
            intent="Account module",
            provides=[
                InterfaceSpec(
                    name="AccountManager",
                    kind=InterfaceKind.CLASS,
                    signature="create(email) -> User",
                    tags=["account"],
                ),
            ],
            evidence=[Evidence.code_committed("account.py")],
        )
        resolver.publish(intent_a)

        intent_b = Intent(
            agent_id="agent-b",
            intent="User handling",
            provides=[
                InterfaceSpec(
                    name="UserHandler",
                    kind=InterfaceKind.CLASS,
                    signature="handle(req) -> Response",
                    tags=["handler"],
                ),
            ],
        )

        result = resolver.resolve(intent_b)
        # No structural overlap, no semantic matcher → clean
        assert result.is_clean
        assert not result.has_adjustments

    def test_below_threshold_ignored(self):
        """Semantic matches below confidence threshold are ignored."""
        matcher = MockSemanticMatcher()
        matcher.overlap_results[("Foo", "Bar")] = SemanticMatch(
            overlap=True, confidence=0.4, reasoning="weak match"
        )

        resolver = IntentResolver(
            min_stability=0.0,
            semantic_matcher=matcher,
            semantic_confidence_threshold=0.7,
        )

        intent_a = Intent(
            agent_id="agent-a",
            intent="Foo module",
            provides=[
                InterfaceSpec(name="Foo", kind=InterfaceKind.CLASS, signature="run()"),
            ],
            evidence=[Evidence.code_committed("foo.py")],
        )
        resolver.publish(intent_a)

        intent_b = Intent(
            agent_id="agent-b",
            intent="Bar module",
            provides=[
                InterfaceSpec(name="Bar", kind=InterfaceKind.CLASS, signature="execute()"),
            ],
        )

        result = resolver.resolve(intent_b)
        assert result.is_clean
        assert not result.has_adjustments


# ---------------------------------------------------------------------------
# Test: Confidence Scoring
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    """Confidence scores on adjustments and conflicts."""

    def test_structural_adjustments_confidence_1(self):
        """Structural matches get confidence=1.0."""
        resolver = IntentResolver(min_stability=0.0)

        intent_a = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID"),
            ],
            evidence=[Evidence.code_committed("auth.py")],
        )
        resolver.publish(intent_a)

        intent_b = Intent(
            agent_id="agent-b",
            intent="User module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: int"),
            ],
        )

        result = resolver.resolve(intent_b)
        assert result.has_adjustments
        for adj in result.adjustments:
            assert adj.confidence == 1.0

    def test_min_confidence_returns_lowest(self):
        """min_confidence property returns the lowest score."""
        result = ResolutionResult(
            original_intent_id="test",
            adjustments=[
                Adjustment(kind="A", description="a", source_intent_id="x", confidence=1.0),
                Adjustment(kind="B", description="b", source_intent_id="y", confidence=0.75),
            ],
        )
        assert result.min_confidence == 0.75

    def test_min_confidence_empty(self):
        """min_confidence is 1.0 when there are no adjustments or conflicts."""
        result = ResolutionResult(original_intent_id="test")
        assert result.min_confidence == 1.0

    def test_min_confidence_includes_conflicts(self):
        """min_confidence considers both adjustments and conflicts."""
        result = ResolutionResult(
            original_intent_id="test",
            adjustments=[
                Adjustment(kind="A", description="a", source_intent_id="x", confidence=0.9),
            ],
            conflicts=[
                ConflictReport(
                    my_intent_id="a",
                    their_intent_id="b",
                    description="conflict",
                    their_stability=0.5,
                    resolution_suggestion="fix",
                    confidence=0.6,
                ),
            ],
        )
        assert result.min_confidence == 0.6

    def test_adjustments_above_filters(self):
        """adjustments_above() filters by threshold."""
        result = ResolutionResult(
            original_intent_id="test",
            adjustments=[
                Adjustment(kind="A", description="a", source_intent_id="x", confidence=1.0),
                Adjustment(kind="B", description="b", source_intent_id="y", confidence=0.75),
                Adjustment(kind="C", description="c", source_intent_id="z", confidence=0.5),
            ],
        )
        above_08 = result.adjustments_above(0.8)
        assert len(above_08) == 1
        assert above_08[0].kind == "A"

        above_07 = result.adjustments_above(0.7)
        assert len(above_07) == 2

    def test_default_confidence_is_1(self):
        """Adjustment and ConflictReport default confidence is 1.0."""
        adj = Adjustment(kind="X", description="x", source_intent_id="a")
        assert adj.confidence == 1.0

        cr = ConflictReport(
            my_intent_id="a",
            their_intent_id="b",
            description="c",
            their_stability=0.5,
            resolution_suggestion="d",
        )
        assert cr.confidence == 1.0


# ---------------------------------------------------------------------------
# Test: Trajectory Prediction
# ---------------------------------------------------------------------------


class TestTrajectoryPrediction:
    """Predictive convergence from agent history."""

    def test_predicts_from_history(self):
        """predict_trajectories() calls SemanticMatcher with agent history."""
        matcher = MockSemanticMatcher()
        matcher.trajectory_result = TrajectoryPrediction(
            agent_id="agent-a",
            predicted_provisions=["UserSettings", "PreferencesService"],
            predicted_requirements=["DatabaseConnection"],
            predicted_constraints=["settings must be JSON-serializable"],
            confidence=0.8,
            reasoning="Agent A has been building user-related services",
        )

        resolver = IntentResolver(min_stability=0.0, semantic_matcher=matcher)

        intent_a = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID"),
            ],
        )
        resolver.publish(intent_a)

        predictions = resolver.predict_trajectories()
        assert "agent-a" in predictions
        pred = predictions["agent-a"]
        assert pred.confidence == 0.8
        assert "UserSettings" in pred.predicted_provisions
        assert matcher.trajectory_call_count == 1

    def test_returns_empty_without_matcher(self):
        """predict_trajectories() returns {} when semantic_matcher is None."""
        resolver = IntentResolver(min_stability=0.0, semantic_matcher=None)

        intent = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID"),
            ],
        )
        resolver.publish(intent)

        predictions = resolver.predict_trajectories()
        assert predictions == {}

    def test_skips_agents_without_history(self):
        """Agents with no published intents are skipped."""
        matcher = MockSemanticMatcher()
        resolver = IntentResolver(min_stability=0.0, semantic_matcher=matcher)

        intent = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID"),
            ],
        )
        resolver.publish(intent)

        # Request prediction for agent-b (no history)
        predictions = resolver.predict_trajectories(agent_ids=["agent-b"])
        assert "agent-b" not in predictions
        assert matcher.trajectory_call_count == 0


# ---------------------------------------------------------------------------
# Test: Semantic Constraint Detection
# ---------------------------------------------------------------------------


class TestSemanticConstraintDetection:
    """LLM catches constraint applicability beyond tag matching."""

    def test_llm_constraint_adoption(self):
        """Constraint that doesn't match tags but LLM says applies."""
        matcher = MockSemanticMatcher()
        matcher.constraint_results[("API versioning", "Build GraphQL API")] = (
            ConstraintApplicability(
                applies=True,
                confidence=0.9,
                reasoning="GraphQL endpoints are a type of API",
            )
        )

        resolver = IntentResolver(min_stability=0.0, semantic_matcher=matcher)

        # Agent A constrains "API versioning" with tags ["api", "rest"]
        intent_a = Intent(
            agent_id="agent-a",
            intent="REST API module",
            provides=[
                InterfaceSpec(
                    name="RestEndpoint",
                    kind=InterfaceKind.ENDPOINT,
                    signature="GET /users",
                    tags=["api", "rest"],
                ),
            ],
            constraints=[
                Constraint(
                    target="API versioning",
                    requirement="all endpoints must include /v1/ prefix",
                    affects_tags=["api", "rest"],
                ),
            ],
        )
        resolver.publish(intent_a)

        # Agent B builds GraphQL (different tags — no structural match)
        intent_b = Intent(
            agent_id="agent-b",
            intent="Build GraphQL API",
            provides=[
                InterfaceSpec(
                    name="GraphQLEndpoint",
                    kind=InterfaceKind.ENDPOINT,
                    signature="POST /graphql",
                    tags=["graphql", "endpoint"],
                ),
            ],
        )

        result = resolver.resolve(intent_b)
        adopt = [a for a in result.adjustments if a.kind == "AdoptConstraint"]
        assert len(adopt) == 1
        assert adopt[0].confidence == 0.9
        assert "semantic" in adopt[0].description
        assert len(result.adopted_constraints) == 1


# ---------------------------------------------------------------------------
# Test: Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Phase 1 behavior is preserved when semantic_matcher=None."""

    def test_three_agent_demo_converges_without_matcher(self):
        """The full demo converges with semantic_matcher=None (default)."""
        resolver = IntentResolver(min_stability=0.0)

        agent_a = build_agent_a(resolver)
        agent_b = build_agent_b(resolver)
        agent_c = build_agent_c(resolver)

        runner = SimulationRunner(resolver)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        runner.add_agent(agent_c)

        result = runner.run()
        assert result.all_converged, f"Agents did not converge: {result.total_conflicts}"

    def test_three_agent_demo_converges_with_noop_matcher(self):
        """Demo converges even with a matcher that finds nothing extra."""
        matcher = MockSemanticMatcher()  # Returns no-overlap for everything
        resolver = IntentResolver(min_stability=0.0, semantic_matcher=matcher)

        agent_a = build_agent_a(resolver)
        agent_b = build_agent_b(resolver)
        agent_c = build_agent_c(resolver)

        runner = SimulationRunner(resolver)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        runner.add_agent(agent_c)

        result = runner.run()
        assert result.all_converged


# ---------------------------------------------------------------------------
# Test: SemanticCache
# ---------------------------------------------------------------------------


class TestSemanticCache:
    """Tests for the in-memory content-hash cache."""

    def test_cache_hit(self):
        cache = _SemanticCache(max_size=10)
        cache.set({"a": 1, "b": 2}, "result")
        assert cache.get({"a": 1, "b": 2}) == "result"

    def test_cache_miss(self):
        cache = _SemanticCache(max_size=10)
        assert cache.get({"a": 1}) is None

    def test_cache_eviction(self):
        """When cache is full, oldest quarter is evicted."""
        cache = _SemanticCache(max_size=4)
        cache.set("key0", "v0")
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.set("key3", "v3")
        # Cache is full (4/4). Next insert evicts oldest quarter (1 entry).
        cache.set("key4", "v4")
        assert len(cache) == 4
        # key0 should have been evicted
        assert cache.get("key0") is None
        assert cache.get("key4") == "v4"
