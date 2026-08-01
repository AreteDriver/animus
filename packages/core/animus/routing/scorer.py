"""TrajectoryScorer: score providers based on historical performance + capability match.

Per ACE-Router paper: use graph-based expansion to explore candidate providers
beyond immediate semantic neighbors. Score based on trajectory success rates
weighted by recency and similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

from animus.logging import get_logger
from animus.routing.graph import ProviderNode, RoutingEdge, RoutingGraph, TaskSignature

logger = get_logger("routing.scorer")


@dataclass
class ScoreWeights:
    """Weights for trajectory scoring components."""

    success_rate: float = 0.35  # Historical success rate
    recent_success: float = 0.25  # Recent success rate (last 5)
    quality: float = 0.20  # Average quality score
    latency: float = 0.10  # Inverse latency (faster = higher)
    capability_match: float = 0.10  # Exact capability match bonus


class TrajectoryScorer:
    """Score providers for a given task using trajectory-based reasoning.

    Unlike static matching, this considers:
    - Historical success on similar tasks
    - Recent performance trends
    - Quality outcomes from rubric eval
    - Latency characteristics
    """

    def __init__(self, graph: RoutingGraph, weights: ScoreWeights | None = None):
        self.graph = graph
        self.weights = weights or ScoreWeights()

    def score_provider(
        self,
        provider: ProviderNode,
        task_signature: TaskSignature,
        estimated_tokens: int = 0,
    ) -> float:
        """Score a single provider for a task.

        Returns score 0.0–1.0. Higher = better candidate.
        """
        # Check capability
        if not provider.can_handle(task_signature.task_type, estimated_tokens):
            return 0.0

        # Get exact edge
        exact_edge = self.graph.get_edge(provider.name, task_signature)

        # Get similar edges for trajectory expansion
        similar_edges = self.graph.get_similar_edges(
            provider.name, task_signature, min_similarity=0.3
        )

        # Combine exact + similar edges weighted by similarity
        all_edges: list[tuple[RoutingEdge, float]] = []
        if exact_edge:
            all_edges.append((exact_edge, 1.0))
        all_edges.extend(similar_edges)

        if not all_edges:
            # No history — use capability match as baseline
            return self._baseline_score(provider, task_signature)

        # Compute weighted statistics across all edges
        total_weight = 0.0
        weighted_success = 0.0
        weighted_recent = 0.0
        weighted_quality = 0.0
        weighted_latency = 0.0

        for edge, sim_weight in all_edges:
            if edge.total_attempts == 0:
                continue

            w = sim_weight  # Similarity weight
            total_weight += w

            weighted_success += edge.success_rate * w
            weighted_recent += edge.recent_success_rate * w
            weighted_quality += edge.avg_quality * w
            # Normalize latency: lower is better, clamp to reasonable range
            latency_score = max(0.0, 1.0 - (edge.avg_latency_ms / 10000))
            weighted_latency += latency_score * w

        if total_weight == 0:
            return self._baseline_score(provider, task_signature)

        # Normalize
        avg_success = weighted_success / total_weight
        avg_recent = weighted_recent / total_weight
        avg_quality = weighted_quality / total_weight
        avg_latency = weighted_latency / total_weight

        # Capability match bonus
        capability_bonus = 1.0 if task_signature.task_type in provider.capabilities else 0.5

        # Combine weighted components
        score = (
            self.weights.success_rate * avg_success
            + self.weights.recent_success * avg_recent
            + self.weights.quality * avg_quality
            + self.weights.latency * avg_latency
            + self.weights.capability_match * capability_bonus
        )

        return min(1.0, max(0.0, score))

    def _baseline_score(self, provider: ProviderNode, task_signature: TaskSignature) -> float:
        """Score when no history exists — use provider capabilities."""
        if task_signature.task_type in provider.capabilities:
            return 0.6  # Moderate confidence for capability match
        if "general" in provider.capabilities:
            return 0.4  # Lower confidence for general-purpose
        return 0.2  # Low confidence for unknown match

    def rank_providers(
        self,
        task_signature: TaskSignature,
        provider_names: list[str] | None = None,
        estimated_tokens: int = 0,
    ) -> list[tuple[ProviderNode, float]]:
        """Rank all providers for a task.

        Returns list of (provider, score) sorted descending by score.
        """
        providers = [
            self.graph.providers[name]
            for name in (provider_names or list(self.graph.providers.keys()))
            if name in self.graph.providers
        ]

        scored = [
            (provider, self.score_provider(provider, task_signature, estimated_tokens))
            for provider in providers
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_best_provider(
        self,
        task_signature: TaskSignature,
        provider_names: list[str] | None = None,
        estimated_tokens: int = 0,
        min_score: float = 0.3,
    ) -> tuple[ProviderNode | None, float]:
        """Get the best provider for a task.

        Returns (provider, score). Provider is None if no provider meets min_score.
        """
        ranked = self.rank_providers(task_signature, provider_names, estimated_tokens)
        if ranked and ranked[0][1] >= min_score:
            return ranked[0]
        return None, 0.0
