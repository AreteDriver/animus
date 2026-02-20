"""Performance benchmarks for Convergent core operations.

Run: PYTHONPATH=python pytest tests/test_benchmarks.py --benchmark-only
"""

import pytest

pytest.importorskip("pytest_benchmark")

from convergent import (
    Intent,
    IntentResolver,
    InterfaceKind,
    InterfaceSpec,
    PythonGraphBackend,
    ScenarioType,
    run_benchmark,
)
from convergent.constraints import ConstraintEngine, ConstraintKind, TypedConstraint
from convergent.score_store import ScoreStore
from convergent.scoring import PhiScorer

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_intent(agent_id: str, name: str, provides: list[str], requires: list[str]) -> Intent:
    """Create a test intent with interface specs."""
    return Intent(
        agent_id=agent_id,
        intent=f"Implement {name}",
        provides=[
            InterfaceSpec(
                name=n,
                kind=InterfaceKind.FUNCTION,
                signature="(x: str) -> str",
                tags=["api", name.lower()],
            )
            for n in provides
        ],
        requires=[
            InterfaceSpec(
                name=n,
                kind=InterfaceKind.FUNCTION,
                signature="(x: str) -> str",
                tags=["api", name.lower()],
            )
            for n in requires
        ],
        constraints=[],
    )


def _seed_resolver(count: int) -> IntentResolver:
    """Create a resolver with N intents from distinct agents."""
    backend = PythonGraphBackend()
    resolver = IntentResolver(backend=backend)
    for i in range(count):
        intent = _make_intent(
            agent_id=f"agent_{i}",
            name=f"service_{i}",
            provides=[f"provide_{i}"],
            requires=[f"require_{i % 5}"],  # Some overlap in requires
        )
        resolver.publish(intent)
    return resolver


# ═══════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════


class TestResolverBenchmark:
    """IntentResolver publish + resolve performance."""

    def test_resolve_50_intents(self, benchmark):
        """Benchmark: IntentResolver.resolve with 50 pre-existing intents."""
        resolver = _seed_resolver(50)
        new_intent = _make_intent(
            agent_id="agent_new",
            name="new_service",
            provides=["require_0"],  # Overlaps with existing requires
            requires=["provide_10"],
        )

        result = benchmark(resolver.resolve, new_intent)
        assert result is not None


class TestStructuralOverlapBenchmark:
    """InterfaceSpec structural overlap detection."""

    def test_structural_overlaps_1000(self, benchmark):
        """Benchmark: structurally_overlaps x1000 pairs."""
        specs_a = [
            InterfaceSpec(
                name=f"func_{i}",
                kind=InterfaceKind.FUNCTION,
                signature="(x: str) -> str",
                tags=["api", f"tag_{i % 10}"],
            )
            for i in range(50)
        ]
        specs_b = [
            InterfaceSpec(
                name=f"func_{i}",
                kind=InterfaceKind.FUNCTION,
                signature="(x: str) -> str",
                tags=["api", f"tag_{i % 10}"],
            )
            for i in range(20)
        ]

        def overlap_check():
            for a in specs_a:
                for b in specs_b:
                    a.structurally_overlaps(b)

        benchmark(overlap_check)


class TestConstraintBenchmark:
    """ConstraintEngine validation throughput."""

    def test_validate_20_constraints(self, benchmark):
        """Benchmark: gate() with 20 registered constraints."""
        engine = ConstraintEngine()
        for i in range(20):
            engine.register(
                TypedConstraint(
                    kind=ConstraintKind.TYPE_CHECK,
                    target=f"service_{i}",
                    requirement=f"Must have field_{i % 5}",
                    required_fields={f"field_{i % 5}": "str"},
                )
            )

        intent = _make_intent("agent_test", "test_svc", ["output"], ["input"])

        result = benchmark(engine.gate, intent)
        assert result is not None


class TestPhiScoringBenchmark:
    """Phi-weighted scoring computation."""

    def test_phi_score_100_outcomes(self, benchmark):
        """Benchmark: calculate_phi_score with 100 historical outcomes."""
        store = ScoreStore(":memory:")
        scorer = PhiScorer(store=store)
        outcomes = (
            [("approved", float(i * 0.5)) for i in range(70)]
            + [("rejected", float(i * 0.3)) for i in range(20)]
            + [("failed", float(i * 0.8)) for i in range(10)]
        )

        score = benchmark(scorer.calculate_phi_score, outcomes)
        assert 0.0 < score < 1.0


class TestRealisticScenarioBenchmark:
    """End-to-end coordination scenario."""

    def test_realistic_25_agents(self, benchmark):
        """Benchmark: run_benchmark(REALISTIC, 25) — full coordination."""
        result = benchmark(run_benchmark, ScenarioType.REALISTIC, 25)
        assert result.total_intents > 0
        assert result.all_converged


class TestPublishThroughputBenchmark:
    """Intent publish throughput."""

    def test_publish_100_intents(self, benchmark):
        """Benchmark: publish 100 intents sequentially."""

        def publish_batch():
            backend = PythonGraphBackend()
            resolver = IntentResolver(backend=backend)
            for i in range(100):
                intent = _make_intent(
                    agent_id=f"agent_{i}",
                    name=f"service_{i}",
                    provides=[f"provide_{i}"],
                    requires=[f"require_{i % 5}"],
                )
                resolver.publish(intent)
            return resolver.intent_count

        count = benchmark(publish_batch)
        assert count == 100
