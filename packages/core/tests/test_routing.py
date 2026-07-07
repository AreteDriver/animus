"""Tests for history-aware provider routing.

Validates P-20260706-003: History-Aware Provider Routing.
"""

import json
import tempfile

import pytest

from animus.routing.graph import (
    ProviderNode,
    RoutingEdge,
    RoutingGraph,
    RoutingOutcome,
    TaskSignature,
)
from animus.routing.scorer import ScoreWeights, TrajectoryScorer
from animus.routing.router import ProviderRouter, RouterConfig, RoutingDecision


class TestTaskSignature:
    """Task signature extraction and similarity."""

    def test_from_prompt_extracts_keywords(self):
        sig = TaskSignature.from_prompt("Read the file and summarize contents")
        assert "read" in sig.keywords
        assert "file" in sig.keywords
        assert sig.task_type == "summarization"

    def test_from_prompt_code_task(self):
        sig = TaskSignature.from_prompt("Implement a function to sort a list")
        assert sig.task_type == "code"

    def test_similarity_identical(self):
        sig1 = TaskSignature.from_prompt("read file contents")
        sig2 = TaskSignature.from_prompt("read file contents")
        assert sig1.similarity(sig2) == 1.0

    def test_similarity_partial(self):
        sig1 = TaskSignature.from_prompt("read file contents")
        sig2 = TaskSignature.from_prompt("read directory listing")
        assert 0 < sig1.similarity(sig2) < 1.0

    def test_similarity_none(self):
        sig1 = TaskSignature.from_prompt("weather forecast")
        sig2 = TaskSignature.from_prompt("sort algorithm")
        assert sig1.similarity(sig2) == 0.0


class TestRoutingGraph:
    """Graph operations and statistics."""

    def test_register_provider(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="test", provider_type="mock", capabilities={"general"})
        graph.register_provider(provider)
        assert "test" in graph.providers

    def test_record_outcome(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"code"})
        graph.register_provider(provider)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="abc123")
        outcome = RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.8)
        graph.record_outcome("p1", sig, outcome)

        edge = graph.get_edge("p1", sig)
        assert edge is not None
        assert edge.success_rate == 1.0

    def test_get_similar_edges(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"general"})
        graph.register_provider(provider)

        sig1 = TaskSignature(task_type="general", keywords=("read", "file"), hash="h1")
        sig2 = TaskSignature(task_type="general", keywords=("read", "file", "content"), hash="h2")

        graph.record_outcome("p1", sig1, RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.8))
        graph.record_outcome("p1", sig2, RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.8))

        similar = graph.get_similar_edges("p1", sig1, min_similarity=0.3)
        assert len(similar) >= 1

    def test_provider_stats(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"code"})
        graph.register_provider(provider)

        for i in range(5):
            sig = TaskSignature(task_type="code", keywords=("test",), hash=f"h{i}")
            graph.record_outcome(
                "p1", sig,
                RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.8)
            )

        stats = graph.get_provider_stats("p1")
        assert stats["total_attempts"] == 5
        assert stats["overall_success_rate"] == 1.0

    def test_edge_pruning(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"general"})
        graph.register_provider(provider)
        sig = TaskSignature(task_type="general", keywords=("test",), hash="h1")

        for i in range(105):
            graph.record_outcome(
                "p1", sig,
                RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.8)
            )

        edge = graph.get_edge("p1", sig)
        assert len(edge.outcomes) == 100  # Pruned to last 100

    def test_serialization(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"code"})
        graph.register_provider(provider)
        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        graph.record_outcome("p1", sig, RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.8))

        data = graph.to_dict()
        restored = RoutingGraph.from_dict(data)
        assert "p1" in restored.providers
        assert len(restored.edges) == 1


class TestTrajectoryScorer:
    """Provider scoring based on trajectory history."""

    def test_score_provider_with_history(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"code"})
        graph.register_provider(provider)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        for _ in range(5):
            graph.record_outcome(
                "p1", sig,
                RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.9)
            )

        scorer = TrajectoryScorer(graph)
        score = scorer.score_provider(provider, sig)
        assert score > 0.7  # High score due to strong history

    def test_score_provider_no_history(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"code"})
        graph.register_provider(provider)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        scorer = TrajectoryScorer(graph)
        score = scorer.score_provider(provider, sig)
        assert score == 0.6  # Baseline for capability match

    def test_score_provider_no_capability(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"general"})
        graph.register_provider(provider)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        scorer = TrajectoryScorer(graph)
        score = scorer.score_provider(provider, sig)
        assert score == 0.4  # Lower baseline for general provider

    def test_score_provider_cannot_handle(self):
        graph = RoutingGraph()
        provider = ProviderNode(name="p1", provider_type="mock", capabilities={"general"}, max_tokens=100)
        graph.register_provider(provider)

        sig = TaskSignature(task_type="general", keywords=("test",), hash="h1")
        scorer = TrajectoryScorer(graph)
        score = scorer.score_provider(provider, sig, estimated_tokens=500)
        assert score == 0.0  # Cannot handle due to token limit

    def test_rank_providers(self):
        graph = RoutingGraph()
        p1 = ProviderNode(name="good", provider_type="mock", capabilities={"code"})
        p2 = ProviderNode(name="bad", provider_type="mock", capabilities={"code"})
        graph.register_provider(p1)
        graph.register_provider(p2)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        for _ in range(5):
            graph.record_outcome("good", sig, RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.9))
            graph.record_outcome("bad", sig, RoutingOutcome(success=False, latency_ms=5000, tokens_used=50, quality_score=0.2))

        scorer = TrajectoryScorer(graph)
        ranked = scorer.rank_providers(sig)
        assert ranked[0][0].name == "good"
        assert ranked[1][0].name == "bad"

    def test_get_best_provider(self):
        graph = RoutingGraph()
        p1 = ProviderNode(name="best", provider_type="mock", capabilities={"code"})
        graph.register_provider(p1)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        graph.record_outcome("best", sig, RoutingOutcome(success=True, latency_ms=100, tokens_used=50, quality_score=0.9))

        scorer = TrajectoryScorer(graph)
        provider, score = scorer.get_best_provider(sig, min_score=0.5)
        assert provider is not None
        assert provider.name == "best"

    def test_get_best_provider_below_threshold(self):
        graph = RoutingGraph()
        p1 = ProviderNode(name="weak", provider_type="mock", capabilities={"general"})
        graph.register_provider(p1)

        sig = TaskSignature(task_type="code", keywords=("test",), hash="h1")
        scorer = TrajectoryScorer(graph)
        provider, score = scorer.get_best_provider(sig, min_score=0.8)
        assert provider is None


class TestProviderRouter:
    """End-to-end routing decisions."""

    def test_select_with_history(self):
        config = RouterConfig(
            enabled=True,
            providers=[
                ProviderNode(name="primary", provider_type="anthropic", capabilities={"code", "reasoning"}),
                ProviderNode(name="local", provider_type="ollama", capabilities={"general"}),
            ],
            exploration_rate=0.0,
        )
        router = ProviderRouter(config)

        # Seed with history
        for _ in range(5):
            router.record_success("primary", "implement a function", latency_ms=500, quality_score=0.9, task_type="code")
            router.record_failure("local", "implement a function", latency_ms=100, error_type="bad_response", task_type="code")

        decision = router.select("implement a function to sort a list")
        assert decision.provider_name == "primary"
        assert decision.score > 0.5

    def test_select_no_history(self):
        config = RouterConfig(
            enabled=True,
            providers=[
                ProviderNode(name="primary", provider_type="anthropic", capabilities={"general"}),
            ],
        )
        router = ProviderRouter(config)
        decision = router.select("hello world")
        assert decision.provider_name == "primary"

    def test_disabled_router(self):
        config = RouterConfig(enabled=False, default_provider="fallback")
        router = ProviderRouter(config)
        decision = router.select("anything")
        assert decision.provider_name == "fallback"
        assert "disabled" in decision.reason

    def test_exploration(self):
        config = RouterConfig(
            enabled=True,
            providers=[
                ProviderNode(name="p1", provider_type="mock", capabilities={"general"}),
                ProviderNode(name="p2", provider_type="mock", capabilities={"general"}),
            ],
            exploration_rate=1.0,  # Always explore
        )
        router = ProviderRouter(config)

        # Seed p1 as better
        for _ in range(10):
            router.record_success("p1", "test", quality_score=0.9)
            router.record_failure("p2", "test", error_type="timeout")

        decision = router.select("test prompt")
        assert decision.was_exploration is True
        assert decision.provider_name == "p2"  # Second-best for exploration

    def test_record_outcomes(self):
        config = RouterConfig(
            enabled=True,
            providers=[
                ProviderNode(name="p1", provider_type="mock", capabilities={"general"}),
            ],
        )
        router = ProviderRouter(config)

        # select() creates a decision; outcomes are recorded separately
        router.select("test prompt")
        router.record_success("p1", "test prompt", latency_ms=100, quality_score=0.8)
        router.record_failure("p1", "test prompt", latency_ms=200, error_type="timeout")

        stats = router.get_stats()
        assert stats["decisions"] == 1  # Only select() adds decisions
        assert stats["providers"]["p1"]["total_attempts"] == 2
        assert stats["providers"]["p1"]["overall_success_rate"] == 0.5

    def test_save_load(self):
        config = RouterConfig(
            enabled=True,
            providers=[
                ProviderNode(name="p1", provider_type="mock", capabilities={"general"}),
            ],
        )
        router = ProviderRouter(config)
        router.record_success("p1", "test", quality_score=0.8)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            router.save(path)

            new_router = ProviderRouter(config)
            new_router.load(path)
            assert "p1" in new_router.graph.providers
            assert len(new_router.graph.edges) == 1
        finally:
            import os
            os.unlink(path)

    def test_select_excludes_providers(self):
        config = RouterConfig(
            enabled=True,
            providers=[
                ProviderNode(name="p1", provider_type="mock", capabilities={"general"}),
                ProviderNode(name="p2", provider_type="mock", capabilities={"general"}),
            ],
        )
        router = ProviderRouter(config)
        router.record_success("p1", "test", quality_score=0.8)
        router.record_success("p2", "test", quality_score=0.9)

        decision = router.select("test", excluded=["p2"])
        assert decision.provider_name == "p1"
