"""Intelligent provider routing for the Gorgon multi-agent orchestration framework.

Selects the optimal AI provider and model for each task based on historical
outcomes, cost efficiency, and latency. Makes providers interchangeable
commodities by abstracting selection behind a scoring system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from animus_forge.metrics.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RoutingStrategy(Enum):
    """Strategy used to select a provider+model pair."""

    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    QUALITY = "quality"
    BALANCED = "balanced"
    FALLBACK = "fallback"


@dataclass
class ProviderSelection:
    """Result of a routing decision.

    Attributes:
        provider: Handler name matching the workflow executor
            (e.g. ``"claude_code"`` or ``"openai"``).
        model: Specific model identifier.
        reason: Human-readable explanation for the selection.
        confidence: Value between 0.0 and 1.0 indicating routing confidence.
        fallback: Alternative selection to try when the primary fails.
    """

    provider: str
    model: str
    reason: str
    confidence: float
    fallback: ProviderSelection | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        result: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "confidence": self.confidence,
        }
        if self.fallback is not None:
            result["fallback"] = self.fallback.to_dict()
        return result


@dataclass
class ProviderCapability:
    """Describes what a registered provider can do.

    Attributes:
        name: Provider handler name (e.g. ``"claude_code"``).
        models: Available model identifiers.
        strengths: Task categories the provider excels at.
        cost_tier: Rough cost bucket — ``"low"``, ``"medium"``, or ``"high"``.
    """

    name: str
    models: list[str]
    strengths: list[str]
    cost_tier: str  # "low", "medium", "high"


@dataclass
class _EMAStats:
    """Exponential moving average stats for a provider+model on a role."""

    quality: float = 0.5
    latency: float = 1.0  # seconds
    cost: float = 0.001  # USD per call
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

_DEFAULT_CAPABILITIES: list[ProviderCapability] = [
    ProviderCapability(
        name="claude_code",
        models=["claude-sonnet-4-20250514", "claude-3-haiku"],
        strengths=["code_generation", "review", "architecture", "analysis"],
        cost_tier="medium",
    ),
    ProviderCapability(
        name="openai",
        models=["gpt-4o", "gpt-4o-mini"],
        strengths=["general", "creative", "summarization", "analysis"],
        cost_tier="medium",
    ),
]

# Mapping of (provider, model) to approximate cost-per-1k-tokens (average of
# input+output) used when no CostTracker data is available.
_DEFAULT_COST_PER_1K: dict[tuple[str, str], float] = {
    ("claude_code", "claude-sonnet-4-20250514"): 0.009,
    ("claude_code", "claude-3-haiku"): 0.00075,
    ("openai", "gpt-4o"): 0.00625,
    ("openai", "gpt-4o-mini"): 0.000375,
}

# Rough relative latency multiplier (1.0 = baseline).
_DEFAULT_LATENCY: dict[tuple[str, str], float] = {
    ("claude_code", "claude-sonnet-4-20250514"): 1.2,
    ("claude_code", "claude-3-haiku"): 0.5,
    ("openai", "gpt-4o"): 1.0,
    ("openai", "gpt-4o-mini"): 0.4,
}

# Rough quality tier (0-1 scale).
_DEFAULT_QUALITY: dict[tuple[str, str], float] = {
    ("claude_code", "claude-sonnet-4-20250514"): 0.92,
    ("claude_code", "claude-3-haiku"): 0.65,
    ("openai", "gpt-4o"): 0.90,
    ("openai", "gpt-4o-mini"): 0.68,
}

# Map complexity levels to quality tiers for model filtering.
_COMPLEXITY_QUALITY_FLOOR: dict[str, float] = {
    "low": 0.0,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.85,
}


# ---------------------------------------------------------------------------
# ProviderRouter
# ---------------------------------------------------------------------------


class ProviderRouter:
    """Routes tasks to the optimal provider+model based on outcomes, cost, and constraints.

    The router maintains a registry of available providers and an internal EMA
    stats table that is updated after every execution.  When no historical data
    is available it falls back to sensible hardcoded defaults so that routing
    works out of the box.

    Args:
        outcome_tracker: An ``OutcomeTracker`` instance (or compatible object)
            that exposes ``get_provider_stats()`` and
            ``get_best_provider_for_role(role)``.
        cost_tracker: Optional ``CostTracker`` for live cost data.  When
            *None* the global singleton is used.
        strategy: Default routing strategy name (must match a
            ``RoutingStrategy`` value).  Defaults to ``"balanced"``.
    """

    # EMA smoothing factor — higher values weigh recent observations more.
    _EMA_ALPHA: float = 0.3

    # Default weight vector for BALANCED scoring.
    _BALANCED_WEIGHTS: dict[str, float] = {
        "quality": 0.4,
        "cost": 0.3,
        "latency": 0.3,
    }

    def __init__(
        self,
        outcome_tracker: Any,
        cost_tracker: CostTracker | None = None,
        strategy: str = "balanced",
    ) -> None:
        self._outcome_tracker = outcome_tracker
        self._cost_tracker = cost_tracker
        self._default_strategy = RoutingStrategy(strategy)

        # Registered providers keyed by handler name.
        self._providers: dict[str, ProviderCapability] = {}

        # EMA stats keyed by (agent_role, provider_name, model).
        self._stats: dict[tuple[str, str, str], _EMAStats] = {}

        # Seed the registry with built-in defaults.
        for cap in _DEFAULT_CAPABILITIES:
            self.register_provider(cap.name, cap.models, cap.strengths, cap.cost_tier)

    # ------------------------------------------------------------------
    # Provider registry
    # ------------------------------------------------------------------

    def register_provider(
        self,
        name: str,
        models: list[str],
        capabilities: list[str],
        cost_tier: str = "medium",
    ) -> None:
        """Register an available provider and its models.

        Args:
            name: Provider handler name (e.g. ``"claude_code"``).
            models: List of model identifiers.
            capabilities: Strength categories (e.g. ``["code_generation"]``).
            cost_tier: ``"low"``, ``"medium"``, or ``"high"``.
        """
        self._providers[name] = ProviderCapability(
            name=name,
            models=models,
            strengths=capabilities,
            cost_tier=cost_tier,
        )
        logger.debug("Registered provider %s with models %s", name, models)

    # ------------------------------------------------------------------
    # Main routing entry point
    # ------------------------------------------------------------------

    def select_provider(
        self,
        agent_role: str,
        task_complexity: str = "medium",
        strategy: str | None = None,
        budget_remaining: float | None = None,
    ) -> ProviderSelection:
        """Select the best provider+model for a given agent role and task.

        Args:
            agent_role: The agent role requesting a provider (e.g.
                ``"builder"``, ``"reviewer"``).
            task_complexity: One of ``"low"``, ``"medium"``, ``"high"``,
                ``"critical"``.
            strategy: Override the default routing strategy for this call.
            budget_remaining: Remaining budget in USD.  When provided, the
                router will exclude models whose estimated cost exceeds this
                value.

        Returns:
            A ``ProviderSelection`` describing the chosen provider, model, and
            rationale.
        """
        active_strategy = RoutingStrategy(strategy) if strategy else self._default_strategy
        logger.info(
            "Routing for role=%s complexity=%s strategy=%s budget=%.4f",
            agent_role,
            task_complexity,
            active_strategy.value,
            budget_remaining if budget_remaining is not None else -1,
        )

        candidates = self._build_candidate_list(agent_role, task_complexity, budget_remaining)

        if not candidates:
            logger.warning("No candidates available for role=%s; returning default", agent_role)
            return self._default_selection(agent_role)

        if active_strategy == RoutingStrategy.CHEAPEST:
            return self._pick_cheapest(candidates, agent_role)
        if active_strategy == RoutingStrategy.FASTEST:
            return self._pick_fastest(candidates, agent_role)
        if active_strategy == RoutingStrategy.QUALITY:
            return self._pick_quality(candidates, agent_role)
        if active_strategy == RoutingStrategy.FALLBACK:
            return self._pick_fallback(candidates, agent_role)
        # BALANCED (default)
        return self._pick_balanced(candidates, agent_role)

    # ------------------------------------------------------------------
    # Routing table
    # ------------------------------------------------------------------

    def get_routing_table(self) -> dict[str, ProviderSelection]:
        """Return the current optimal provider for every known agent role.

        Returns:
            Mapping of agent role to the ``ProviderSelection`` that the
            default strategy would choose right now.
        """
        roles = self._known_roles()
        table: dict[str, ProviderSelection] = {}
        for role in roles:
            table[role] = self.select_provider(role)
        return table

    # ------------------------------------------------------------------
    # Post-execution feedback
    # ------------------------------------------------------------------

    def update_after_execution(
        self,
        agent_role: str,
        provider: str,
        model: str,
        outcome: dict[str, Any],
    ) -> None:
        """Update internal EMA stats after a step completes.

        Args:
            agent_role: The agent role that executed.
            provider: Provider handler name.
            model: Model identifier.
            outcome: Dictionary with optional keys ``quality_score`` (float
                0-1), ``latency_seconds`` (float), and ``cost_usd`` (float).
        """
        key = (agent_role, provider, model)
        stats = self._stats.get(key, _EMAStats())

        alpha = self._EMA_ALPHA

        if "quality_score" in outcome:
            stats.quality = alpha * outcome["quality_score"] + (1 - alpha) * stats.quality

        if "latency_seconds" in outcome:
            stats.latency = alpha * outcome["latency_seconds"] + (1 - alpha) * stats.latency

        if "cost_usd" in outcome:
            stats.cost = alpha * outcome["cost_usd"] + (1 - alpha) * stats.cost

        stats.sample_count += 1
        self._stats[key] = stats

        logger.debug(
            "Updated stats for %s: quality=%.3f latency=%.3f cost=%.5f (n=%d)",
            key,
            stats.quality,
            stats.latency,
            stats.cost,
            stats.sample_count,
        )

    # ------------------------------------------------------------------
    # Internal: candidate building
    # ------------------------------------------------------------------

    def _build_candidate_list(
        self,
        agent_role: str,
        task_complexity: str,
        budget_remaining: float | None,
    ) -> list[tuple[str, str, _EMAStats]]:
        """Build scored candidate tuples of (provider, model, stats).

        Filters by complexity floor and budget constraint.
        """
        quality_floor = _COMPLEXITY_QUALITY_FLOOR.get(task_complexity, 0.6)
        candidates: list[tuple[str, str, _EMAStats]] = []

        for prov_name, cap in self._providers.items():
            for model in cap.models:
                stats = self._get_stats(agent_role, prov_name, model)

                # Filter out models below the quality floor for this complexity.
                if stats.quality < quality_floor:
                    continue

                # Filter out models that would exceed the remaining budget.
                if budget_remaining is not None and stats.cost > budget_remaining:
                    continue

                candidates.append((prov_name, model, stats))

        return candidates

    def _get_stats(self, agent_role: str, provider: str, model: str) -> _EMAStats:
        """Retrieve EMA stats for a triple, falling back to defaults."""
        key = (agent_role, provider, model)
        if key in self._stats:
            return self._stats[key]

        # Seed from outcome_tracker if available.
        try:
            provider_stats = self._outcome_tracker.get_provider_stats()
            if provider_stats and provider in provider_stats:
                ps = provider_stats[provider]
                return _EMAStats(
                    quality=ps.get("quality_score", _DEFAULT_QUALITY.get((provider, model), 0.7)),
                    latency=ps.get("avg_latency", _DEFAULT_LATENCY.get((provider, model), 1.0)),
                    cost=ps.get("avg_cost", _DEFAULT_COST_PER_1K.get((provider, model), 0.005)),
                    sample_count=ps.get("count", 0),
                )
        except Exception:
            logger.debug("Could not fetch provider stats from outcome_tracker")

        # Hardcoded defaults.
        return _EMAStats(
            quality=_DEFAULT_QUALITY.get((provider, model), 0.7),
            latency=_DEFAULT_LATENCY.get((provider, model), 1.0),
            cost=_DEFAULT_COST_PER_1K.get((provider, model), 0.005),
            sample_count=0,
        )

    # ------------------------------------------------------------------
    # Internal: strategy pickers
    # ------------------------------------------------------------------

    def _pick_cheapest(
        self,
        candidates: list[tuple[str, str, _EMAStats]],
        agent_role: str,
    ) -> ProviderSelection:
        """Select the candidate with the lowest cost."""
        best = min(candidates, key=lambda c: c[2].cost)
        return ProviderSelection(
            provider=best[0],
            model=best[1],
            reason=f"Cheapest option for {agent_role} (est. ${best[2].cost:.5f}/call)",
            confidence=self._confidence(best[2]),
        )

    def _pick_fastest(
        self,
        candidates: list[tuple[str, str, _EMAStats]],
        agent_role: str,
    ) -> ProviderSelection:
        """Select the candidate with the lowest latency."""
        best = min(candidates, key=lambda c: c[2].latency)
        return ProviderSelection(
            provider=best[0],
            model=best[1],
            reason=f"Fastest option for {agent_role} (est. {best[2].latency:.2f}s)",
            confidence=self._confidence(best[2]),
        )

    def _pick_quality(
        self,
        candidates: list[tuple[str, str, _EMAStats]],
        agent_role: str,
    ) -> ProviderSelection:
        """Select the candidate with the highest quality score."""
        best = max(candidates, key=lambda c: c[2].quality)
        return ProviderSelection(
            provider=best[0],
            model=best[1],
            reason=f"Highest quality for {agent_role} (score {best[2].quality:.2f})",
            confidence=self._confidence(best[2]),
        )

    def _pick_balanced(
        self,
        candidates: list[tuple[str, str, _EMAStats]],
        agent_role: str,
    ) -> ProviderSelection:
        """Score candidates using the weighted BALANCED formula."""
        weights = self._BALANCED_WEIGHTS

        # Collect raw values for normalization.
        qualities = [c[2].quality for c in candidates]
        costs = [c[2].cost for c in candidates]
        latencies = [c[2].latency for c in candidates]

        q_min, q_max = min(qualities), max(qualities)
        c_min, c_max = min(costs), max(costs)
        l_min, l_max = min(latencies), max(latencies)

        def _normalize(val: float, lo: float, hi: float) -> float:
            if hi == lo:
                return 1.0
            return (val - lo) / (hi - lo)

        scored: list[tuple[float, int]] = []
        for idx, (_, _, stats) in enumerate(candidates):
            nq = _normalize(stats.quality, q_min, q_max)
            nc = _normalize(stats.cost, c_min, c_max)
            nl = _normalize(stats.latency, l_min, l_max)

            score = (
                weights["quality"] * nq
                + weights["cost"] * (1.0 - nc)
                + weights["latency"] * (1.0 - nl)
            )
            scored.append((score, idx))

        scored.sort(key=lambda s: s[0], reverse=True)
        best_idx = scored[0][1]
        best = candidates[best_idx]

        # Attach a fallback if a runner-up exists.
        fallback: ProviderSelection | None = None
        if len(scored) > 1:
            runner_idx = scored[1][1]
            runner = candidates[runner_idx]
            fallback = ProviderSelection(
                provider=runner[0],
                model=runner[1],
                reason="Runner-up by balanced score",
                confidence=self._confidence(runner[2]),
            )

        return ProviderSelection(
            provider=best[0],
            model=best[1],
            reason=(
                f"Balanced selection for {agent_role} "
                f"(q={best[2].quality:.2f} c=${best[2].cost:.5f} "
                f"l={best[2].latency:.2f}s)"
            ),
            confidence=self._confidence(best[2]),
            fallback=fallback,
        )

    def _pick_fallback(
        self,
        candidates: list[tuple[str, str, _EMAStats]],
        agent_role: str,
    ) -> ProviderSelection:
        """Pick the best quality candidate and attach the runner-up as fallback."""
        sorted_cands = sorted(candidates, key=lambda c: c[2].quality, reverse=True)
        best = sorted_cands[0]

        fallback: ProviderSelection | None = None
        if len(sorted_cands) > 1:
            runner = sorted_cands[1]
            fallback = ProviderSelection(
                provider=runner[0],
                model=runner[1],
                reason="Fallback provider",
                confidence=self._confidence(runner[2]),
            )

        return ProviderSelection(
            provider=best[0],
            model=best[1],
            reason=f"Primary for {agent_role} with fallback configured",
            confidence=self._confidence(best[2]),
            fallback=fallback,
        )

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _confidence(self, stats: _EMAStats) -> float:
        """Compute a confidence score based on sample count.

        Confidence ramps from 0.3 (no samples) to 0.95 (50+ samples).
        """
        if stats.sample_count == 0:
            return 0.3
        # Asymptotic approach to 0.95.
        return min(0.95, 0.3 + 0.65 * (1 - 1 / (1 + stats.sample_count / 10)))

    def _known_roles(self) -> set[str]:
        """Return all agent roles that have appeared in stats or the outcome tracker."""
        roles: set[str] = {key[0] for key in self._stats}

        try:
            provider_stats = self._outcome_tracker.get_provider_stats()
            if provider_stats:
                for prov_data in provider_stats.values():
                    if isinstance(prov_data, dict) and "roles" in prov_data:
                        roles.update(prov_data["roles"])
        except Exception:
            pass  # Non-critical fallback: outcome tracker unavailable, use known roles only

        # Ensure common roles are always present.
        roles.update({"planner", "builder", "tester", "reviewer", "architect", "documenter"})
        return roles

    def _default_selection(self, agent_role: str) -> ProviderSelection:
        """Return a safe default when no candidates pass filters."""
        return ProviderSelection(
            provider="claude_code",
            model="claude-sonnet-4-20250514",
            reason=f"Default fallback for {agent_role} (no candidates matched filters)",
            confidence=0.2,
            fallback=ProviderSelection(
                provider="openai",
                model="gpt-4o",
                reason="Secondary default fallback",
                confidence=0.2,
            ),
        )
