"""ProviderRouter: history-aware model selection for Animus.

Replaces static keyword matching with trajectory-based provider selection.
Integrates with CognitiveLayer to route tasks to the best model based on
historical performance, not just prompt keywords.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from animus.logging import get_logger
from animus.routing.graph import (
    ProviderNode,
    RoutingGraph,
    RoutingOutcome,
    TaskSignature,
)
from animus.routing.scorer import ScoreWeights, TrajectoryScorer

logger = get_logger("routing.router")


@dataclass
class RouterConfig:
    """Configuration for ProviderRouter."""

    enabled: bool = True
    min_score_threshold: float = 0.3
    default_provider: str = "primary"
    # Exploration: sometimes route to non-optimal provider to gather data
    exploration_rate: float = 0.1  # 10% of requests explore
    # Provider capabilities
    providers: list[ProviderNode] = field(default_factory=list)
    # Score weights
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    # Whether to use rubric quality scores when recording outcomes
    use_rubric_quality: bool = True


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    provider_name: str
    score: float
    reason: str
    was_exploration: bool = False
    alternatives: list[tuple[str, float]] = field(default_factory=list)


class ProviderRouter:
    """History-aware router for selecting models/providers.

    Maintains a routing graph with performance history and uses
    trajectory-based scoring to select the best provider for each task.

    Usage:
        router = ProviderRouter(config)
        decision = router.select(prompt, estimated_tokens=500)
        # ... send to decision.provider_name ...
        router.record_success(decision.provider_name, prompt, latency_ms=1200)
    """

    _MAX_DECISION_HISTORY = 1000

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.graph = RoutingGraph()
        self.scorer = TrajectoryScorer(self.graph, self.config.weights)

        # Register configured providers
        for provider in self.config.providers:
            self.graph.register_provider(provider)

        # Track routing decisions for post-hoc analysis
        self._decision_history: list[RoutingDecision] = []

    def register_provider(self, provider: ProviderNode) -> None:
        """Register a new provider."""
        self.graph.register_provider(provider)

    def select(
        self,
        prompt: str,
        task_type: str | None = None,
        estimated_tokens: int = 0,
        excluded: list[str] | None = None,
    ) -> RoutingDecision:
        """Select the best provider for a task.

        Args:
            prompt: User prompt/task description.
            task_type: Optional explicit task type override.
            estimated_tokens: Estimated token count for the task.
            excluded: Provider names to exclude from selection.

        Returns:
            RoutingDecision with selected provider and alternatives.
        """
        if not self.config.enabled or not self.graph.providers:
            return RoutingDecision(
                provider_name=self.config.default_provider,
                score=0.5,
                reason="Router disabled or no providers registered",
            )

        # Extract task signature
        signature = TaskSignature.from_prompt(prompt, task_type)

        # Get candidate providers
        available = [
            name for name in self.graph.providers.keys() if not excluded or name not in excluded
        ]

        # Rank providers
        ranked = self.scorer.rank_providers(signature, available, estimated_tokens)

        if not ranked:
            return RoutingDecision(
                provider_name=self.config.default_provider,
                score=0.0,
                reason="No providers available",
            )

        # Exploration: occasionally pick non-optimal provider for data gathering
        is_exploration = random.random() < self.config.exploration_rate
        if is_exploration and len(ranked) > 1:
            # Pick second-best for exploration
            selected = ranked[1]
            reason = f"Exploration: trying {selected[0].name} (score={selected[1]:.3f})"
        else:
            selected = ranked[0]
            if selected[1] < self.config.min_score_threshold:
                reason = (
                    f"Best provider {selected[0].name} below threshold "
                    f"({selected[1]:.3f} < {self.config.min_score_threshold}), "
                    f"using default"
                )
                selected = (None, selected[1])
            else:
                reason = f"Selected {selected[0].name} based on trajectory score={selected[1]:.3f}"

        provider_name = selected[0].name if selected[0] else self.config.default_provider
        alternatives = [(name, score) for name, score in ranked[:3] if name != provider_name]

        decision = RoutingDecision(
            provider_name=provider_name,
            score=selected[1],
            reason=reason,
            was_exploration=is_exploration,
            alternatives=alternatives,
        )
        self._decision_history.append(decision)
        if len(self._decision_history) > self._MAX_DECISION_HISTORY:
            self._decision_history = self._decision_history[-self._MAX_DECISION_HISTORY :]
        logger.info(f"Router: {decision.reason}")
        return decision

    def update_quality_score(
        self,
        provider_name: str,
        prompt: str,
        quality_score: float,
        task_type: str | None = None,
    ) -> bool:
        """Update the quality score of the most recent matching edge.

        Allows post-hoc rubric evaluation scores to retroactively improve
        the routing graph after external quality assessment.

        Args:
            provider_name: The provider that handled the task.
            prompt: The original prompt.
            quality_score: New quality score (0.0–1.0).
            task_type: Optional task type override.

        Returns:
            True if an edge was updated, False otherwise.
        """
        signature = TaskSignature.from_prompt(prompt, task_type)
        edge = self.graph.get_edge(provider_name, signature)
        if edge is None or not edge.outcomes:
            return False
        # Update the most recent outcome's quality score
        edge.outcomes[-1].quality_score = quality_score
        logger.debug(
            f"Updated quality score for {provider_name} on {signature.task_type}: "
            f"{quality_score:.2f}"
        )
        return True

    def record_outcome(
        self,
        provider_name: str,
        prompt: str,
        success: bool,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        quality_score: float = 0.5,
        error_type: str | None = None,
        task_type: str | None = None,
    ) -> None:
        """Record the outcome of a routing decision.

        Call this after the provider completes (success or failure).
        """
        signature = TaskSignature.from_prompt(prompt, task_type)
        outcome = RoutingOutcome(
            success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            quality_score=quality_score,
            error_type=error_type,
        )
        self.graph.record_outcome(provider_name, signature, outcome)
        logger.debug(
            f"Recorded outcome for {provider_name} on {signature.task_type}: "
            f"success={success}, quality={quality_score:.2f}"
        )

    def record_success(
        self,
        provider_name: str,
        prompt: str,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        quality_score: float = 0.5,
        task_type: str | None = None,
    ) -> None:
        """Convenience: record a successful outcome."""
        self.record_outcome(
            provider_name=provider_name,
            prompt=prompt,
            success=True,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            quality_score=quality_score,
            task_type=task_type,
        )

    def record_failure(
        self,
        provider_name: str,
        prompt: str,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        error_type: str = "unknown",
        task_type: str | None = None,
    ) -> None:
        """Convenience: record a failed outcome."""
        self.record_outcome(
            provider_name=provider_name,
            prompt=prompt,
            success=False,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            quality_score=0.0,
            error_type=error_type,
            task_type=task_type,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate routing statistics."""
        stats = {
            "providers": {},
            "decisions": len(self._decision_history),
            "explorations": sum(1 for d in self._decision_history if d.was_exploration),
        }
        for name in self.graph.providers:
            stats["providers"][name] = self.graph.get_provider_stats(name)
        return stats

    def save(self, path: str) -> None:
        """Save routing graph to JSON file."""
        import json
        from pathlib import Path

        data = self.graph.to_dict()
        data["config"] = {
            "default_provider": self.config.default_provider,
            "exploration_rate": self.config.exploration_rate,
            "min_score_threshold": self.config.min_score_threshold,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Saved routing graph to {path}")

    def load(self, path: str) -> None:
        """Load routing graph from JSON file."""
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        self.graph = RoutingGraph.from_dict(data)
        self.scorer = TrajectoryScorer(self.graph, self.config.weights)
        logger.info(f"Loaded routing graph from {path}")
