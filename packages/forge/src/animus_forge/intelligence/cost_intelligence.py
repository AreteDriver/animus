"""Cost intelligence and recommendation engine for Gorgon.

Analyzes spending patterns, recommends model switches to reduce costs,
forecasts future spend, and quantifies the ROI of Gorgon's intelligence
layer. Produces actionable insights like:

    "You spent $47 on OpenAI this week, but switching your Builder from
    gpt-4o to Claude Sonnet would save 30% with no quality drop based on
    your last 200 executions."

Depends on:
    - ``CostTracker`` for raw cost entries.
    - ``OutcomeTracker`` (optional) for quality/latency signals.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from animus_forge.metrics.cost_tracker import CostEntry, CostTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpendingAnalysis:
    """Result of ``analyze_spending``.

    Attributes:
        period_days: Look-back window in days.
        total_usd: Total spend in USD over the period.
        by_provider: Spend keyed by provider name.
        by_model: Spend keyed by model identifier.
        by_role: Spend keyed by agent role.
        daily_avg: Average daily spend in USD.
        trend: One of ``"increasing"``, ``"decreasing"``, ``"stable"``.
        top_workflows: Up to 3 most expensive workflows as (name, cost) tuples.
        top_roles: Up to 3 most expensive roles as (role, cost) tuples.
    """

    period_days: int
    total_usd: float
    by_provider: dict[str, float]
    by_model: dict[str, float]
    by_role: dict[str, float]
    daily_avg: float
    trend: str  # "increasing", "decreasing", "stable"
    top_workflows: list[tuple[str, float]]
    top_roles: list[tuple[str, float]]


@dataclass
class SavingsRecommendation:
    """A single recommendation to switch models for an agent role.

    Attributes:
        agent_role: The agent role this recommendation targets.
        current_model: Model currently used.
        recommended_model: Cheaper alternative model.
        current_cost_monthly: Projected monthly cost at current usage.
        projected_cost_monthly: Projected monthly cost after switching.
        estimated_savings_monthly: Dollar savings per month.
        quality_impact: Human-readable quality impact descriptor.
        confidence: ``"high"``, ``"medium"``, or ``"low"``.
        reasoning: Full human-readable explanation.
    """

    agent_role: str
    current_model: str
    recommended_model: str
    current_cost_monthly: float
    projected_cost_monthly: float
    estimated_savings_monthly: float
    quality_impact: str  # "none", "minimal (<5%)", "moderate (5-15%)", "significant (>15%)"
    confidence: str  # "high", "medium", "low"
    reasoning: str


@dataclass
class SwitchImpact:
    """Projected impact of switching a role from one model to another.

    Attributes:
        agent_role: The agent role being evaluated.
        from_model: Current model.
        to_model: Target model.
        cost_change_pct: Percentage cost change (negative = savings).
        quality_change_pct: Percentage quality change (negative = worse).
        latency_change_pct: Percentage latency change (negative = faster).
        monthly_savings_usd: Projected monthly savings in USD.
        risk_level: ``"low"``, ``"medium"``, or ``"high"``.
    """

    agent_role: str
    from_model: str
    to_model: str
    cost_change_pct: float
    quality_change_pct: float
    latency_change_pct: float
    monthly_savings_usd: float
    risk_level: str  # "low", "medium", "high"


@dataclass
class ROIReport:
    """Return-on-investment report for Gorgon's intelligence layer.

    Attributes:
        total_savings_usd: Total cumulative savings attributed to Gorgon.
        routing_savings_usd: Savings from intelligent provider routing.
        enforcement_savings_usd: Savings from contract enforcement preventing reruns.
        total_executions: Number of tracked executions.
        avg_cost_per_execution: Average cost per execution in USD.
        efficiency_score: Score 0-100 where higher means more cost-efficient.
    """

    total_savings_usd: float
    routing_savings_usd: float
    enforcement_savings_usd: float
    total_executions: int
    avg_cost_per_execution: float
    efficiency_score: float  # 0-100


@dataclass
class SpendForecast:
    """Spending forecast for a future window.

    Attributes:
        daily_rate: Current daily spend rate in USD.
        projected_30d: Projected spend for the next 30 days.
        budget_limit: Configured budget limit, or ``None``.
        days_until_budget: Days until the budget is exhausted, or ``None``.
        trend: One of ``"increasing"``, ``"decreasing"``, ``"stable"``.
    """

    daily_rate: float
    projected_30d: float
    budget_limit: float | None
    days_until_budget: int | None
    trend: str


# ---------------------------------------------------------------------------
# CostIntelligence
# ---------------------------------------------------------------------------


class CostIntelligence:
    """Analyses cost data and produces actionable savings recommendations.

    Args:
        cost_tracker: ``CostTracker`` instance with historical cost entries.
        outcome_tracker: Optional ``OutcomeTracker`` for quality/latency data.
            When provided, quality-aware recommendations become available.
    """

    # Models considered for alternative recommendations, ordered roughly by
    # ascending cost so that cheaper alternatives are checked first.
    _CANDIDATE_MODELS: list[tuple[str, str]] = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-haiku"),
        ("anthropic", "claude-3-5-haiku"),
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-sonnet"),
        ("anthropic", "claude-3-5-sonnet"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4-turbo"),
        ("anthropic", "claude-3-opus"),
        ("openai", "gpt-4"),
    ]

    def __init__(
        self,
        cost_tracker: CostTracker,
        outcome_tracker: Any | None = None,
    ) -> None:
        self._cost_tracker = cost_tracker
        self._outcome_tracker = outcome_tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_spending(self, days: int = 30) -> SpendingAnalysis:
        """Analyse spending over a time window.

        Args:
            days: Number of days to look back.

        Returns:
            A ``SpendingAnalysis`` summarising costs by provider, model,
            role, and workflow, plus a daily trend indicator.
        """
        entries = self._entries_in_window(days)

        total_usd = sum(e.cost_usd for e in entries)

        by_provider: dict[str, float] = defaultdict(float)
        by_model: dict[str, float] = defaultdict(float)
        by_role: dict[str, float] = defaultdict(float)
        by_workflow: dict[str, float] = defaultdict(float)
        by_day: dict[str, float] = defaultdict(float)

        for entry in entries:
            by_provider[entry.provider.value] += entry.cost_usd
            by_model[entry.model] += entry.cost_usd
            by_role[entry.agent_role or "unknown"] += entry.cost_usd
            if entry.workflow_id:
                by_workflow[entry.workflow_id] += entry.cost_usd
            day_key = entry.timestamp.strftime("%Y-%m-%d")
            by_day[day_key] += entry.cost_usd

        daily_avg = total_usd / max(days, 1)
        trend = self._compute_trend(by_day, days)

        top_workflows = sorted(by_workflow.items(), key=lambda x: x[1], reverse=True)[:3]
        top_roles = sorted(by_role.items(), key=lambda x: x[1], reverse=True)[:3]

        logger.info(
            "Spending analysis: $%.2f over %d days (trend=%s)",
            total_usd,
            days,
            trend,
        )

        return SpendingAnalysis(
            period_days=days,
            total_usd=round(total_usd, 2),
            by_provider=dict(by_provider),
            by_model=dict(by_model),
            by_role=dict(by_role),
            daily_avg=round(daily_avg, 4),
            trend=trend,
            top_workflows=top_workflows,
            top_roles=top_roles,
        )

    def recommend_savings(self, days: int = 30) -> list[SavingsRecommendation]:
        """Identify model switches that reduce cost without sacrificing quality.

        For each (agent_role, model) pair in the cost history, the method
        checks whether a cheaper model achieves comparable quality (within
        10%) at more than 20% lower cost.

        Args:
            days: Look-back window for cost and quality data.

        Returns:
            Recommendations sorted by ``estimated_savings_monthly`` descending.
        """
        entries = self._entries_in_window(days)
        role_model_costs = self._aggregate_role_model(entries)
        recommendations: list[SavingsRecommendation] = []

        for (role, model), stats in role_model_costs.items():
            current_quality = self._get_quality_score(role, model, days)
            current_cost_per_call = stats["cost"] / max(stats["calls"], 1)
            monthly_calls = stats["calls"] * (30 / max(days, 1))
            current_monthly = current_cost_per_call * monthly_calls

            best_rec = self._find_best_alternative(
                role,
                model,
                current_quality,
                current_cost_per_call,
                monthly_calls,
                current_monthly,
                days,
            )
            if best_rec is not None:
                recommendations.append(best_rec)

        recommendations.sort(key=lambda r: r.estimated_savings_monthly, reverse=True)
        logger.info(
            "Found %d savings recommendations totalling $%.2f/month",
            len(recommendations),
            sum(r.estimated_savings_monthly for r in recommendations),
        )
        return recommendations

    def estimate_switch_impact(
        self,
        agent_role: str,
        from_model: str,
        to_model: str,
        days: int = 30,
    ) -> SwitchImpact:
        """Project the cost, quality, and latency impact of switching models.

        Args:
            agent_role: The agent role to evaluate.
            from_model: Current model identifier.
            to_model: Target model identifier.
            days: Look-back window in days.

        Returns:
            A ``SwitchImpact`` with projected percentage changes and savings.
        """
        entries = self._entries_in_window(days)
        role_entries = [
            e
            for e in entries
            if (e.agent_role or "unknown") == agent_role and e.model == from_model
        ]

        current_cost_per_call = sum(e.cost_usd for e in role_entries) / max(len(role_entries), 1)
        monthly_calls = len(role_entries) * (30 / max(days, 1))

        from_quality = self._get_quality_score(agent_role, from_model, days)
        to_quality = self._get_quality_score(agent_role, to_model, days)

        from_latency = self._get_avg_latency(agent_role, from_model, days)
        to_latency = self._get_avg_latency(agent_role, to_model, days)

        # Estimate new cost by applying the pricing ratio.
        price_ratio = self._pricing_ratio(from_model, to_model)
        projected_cost_per_call = current_cost_per_call * price_ratio

        cost_change_pct = (
            ((projected_cost_per_call - current_cost_per_call) / current_cost_per_call) * 100
            if current_cost_per_call > 0
            else 0.0
        )
        quality_change_pct = (
            ((to_quality - from_quality) / from_quality) * 100 if from_quality > 0 else 0.0
        )
        latency_change_pct = (
            ((to_latency - from_latency) / from_latency) * 100 if from_latency > 0 else 0.0
        )

        monthly_savings = (current_cost_per_call - projected_cost_per_call) * monthly_calls
        risk_level = self._assess_risk(quality_change_pct, cost_change_pct)

        return SwitchImpact(
            agent_role=agent_role,
            from_model=from_model,
            to_model=to_model,
            cost_change_pct=round(cost_change_pct, 2),
            quality_change_pct=round(quality_change_pct, 2),
            latency_change_pct=round(latency_change_pct, 2),
            monthly_savings_usd=round(monthly_savings, 2),
            risk_level=risk_level,
        )

    def get_roi_report(self) -> ROIReport:
        """Compute the cumulative ROI of Gorgon's intelligence layer.

        Compares actual spending against a "naive baseline" where every call
        uses the most expensive available model. Also estimates savings from
        contract enforcement preventing failed re-runs.

        Returns:
            An ``ROIReport`` summarising total savings and efficiency.
        """
        entries = self._cost_tracker.entries

        if not entries:
            return ROIReport(
                total_savings_usd=0.0,
                routing_savings_usd=0.0,
                enforcement_savings_usd=0.0,
                total_executions=0,
                avg_cost_per_execution=0.0,
                efficiency_score=50.0,
            )

        actual_total = sum(e.cost_usd for e in entries)
        total_executions = len(entries)
        avg_cost = actual_total / total_executions

        # Naive baseline: what if every call used the most expensive model?
        naive_total = self._compute_naive_baseline(entries)
        routing_savings = max(0.0, naive_total - actual_total)

        # Enforcement savings: estimate prevented re-runs from outcome data.
        enforcement_savings = self._compute_enforcement_savings(entries)

        total_savings = routing_savings + enforcement_savings

        # Efficiency score: 100 means actual cost is 0% of naive; 0 means at
        # or above naive cost. Scale linearly.
        if naive_total > 0:
            ratio = actual_total / naive_total
            efficiency_score = max(0.0, min(100.0, (1.0 - ratio) * 100))
        else:
            efficiency_score = 50.0

        logger.info(
            "ROI report: $%.2f total savings (%d executions, efficiency=%.1f)",
            total_savings,
            total_executions,
            efficiency_score,
        )

        return ROIReport(
            total_savings_usd=round(total_savings, 2),
            routing_savings_usd=round(routing_savings, 2),
            enforcement_savings_usd=round(enforcement_savings, 2),
            total_executions=total_executions,
            avg_cost_per_execution=round(avg_cost, 6),
            efficiency_score=round(efficiency_score, 1),
        )

    def forecast_spend(self, days_ahead: int = 30) -> SpendForecast:
        """Forecast future spending based on recent daily rates.

        Uses the last 14 days of data for the daily rate and trend
        calculation. Projects forward by ``days_ahead``.

        Args:
            days_ahead: Number of days to forecast.

        Returns:
            A ``SpendForecast`` with daily rate, projection, and budget info.
        """
        lookback = 14
        entries = self._entries_in_window(lookback)
        by_day: dict[str, float] = defaultdict(float)

        for entry in entries:
            day_key = entry.timestamp.strftime("%Y-%m-%d")
            by_day[day_key] += entry.cost_usd

        if not by_day:
            return SpendForecast(
                daily_rate=0.0,
                projected_30d=0.0,
                budget_limit=self._cost_tracker.budget_limit_usd,
                days_until_budget=None,
                trend="stable",
            )

        # Use the actual number of days with data for accurate daily rate.
        daily_values = list(by_day.values())
        daily_rate = statistics.mean(daily_values) if daily_values else 0.0

        projected = daily_rate * days_ahead
        trend = self._compute_trend(by_day, lookback)

        budget_limit = self._cost_tracker.budget_limit_usd
        days_until_budget: int | None = None
        if budget_limit is not None and daily_rate > 0:
            monthly_used = self._cost_tracker.get_monthly_cost()
            remaining = budget_limit - monthly_used
            if remaining > 0:
                days_until_budget = int(remaining / daily_rate)
            else:
                days_until_budget = 0

        logger.info(
            "Spend forecast: $%.2f/day, $%.2f projected %dd",
            daily_rate,
            projected,
            days_ahead,
        )

        return SpendForecast(
            daily_rate=round(daily_rate, 4),
            projected_30d=round(projected, 2),
            budget_limit=budget_limit,
            days_until_budget=days_until_budget,
            trend=trend,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entries_in_window(self, days: int) -> list[CostEntry]:
        """Return cost entries within the last ``days`` days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self._cost_tracker.entries if e.timestamp >= cutoff]

    def _compute_trend(self, by_day: dict[str, float], days: int) -> str:
        """Determine whether daily spending is increasing, decreasing, or stable.

        Splits the window into two halves and compares average spend.
        A difference of more than 15% is considered a trend change.
        """
        if len(by_day) < 2:
            return "stable"

        sorted_days = sorted(by_day.keys())
        mid = len(sorted_days) // 2
        first_half = [by_day[d] for d in sorted_days[:mid]]
        second_half = [by_day[d] for d in sorted_days[mid:]]

        avg_first = statistics.mean(first_half) if first_half else 0.0
        avg_second = statistics.mean(second_half) if second_half else 0.0

        if avg_first == 0:
            return "stable"

        change_pct = (avg_second - avg_first) / avg_first
        if change_pct > 0.15:
            return "increasing"
        if change_pct < -0.15:
            return "decreasing"
        return "stable"

    def _aggregate_role_model(
        self, entries: list[CostEntry]
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Aggregate cost entries by (agent_role, model).

        Returns:
            Mapping of ``(role, model)`` to ``{"cost": ..., "calls": ...,
            "tokens": ...}``.
        """
        result: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0, "tokens": 0}
        )
        for e in entries:
            key = (e.agent_role or "unknown", e.model)
            result[key]["cost"] += e.cost_usd
            result[key]["calls"] += 1
            result[key]["tokens"] += e.tokens.total_tokens
        return dict(result)

    def _get_quality_score(self, agent_role: str, model: str, days: int) -> float:
        """Retrieve the average quality score for a role+model pair.

        Falls back to estimating from the pricing tier when no outcome
        tracker is available.
        """
        if self._outcome_tracker is not None:
            try:
                # Determine provider from model name.
                provider = self._provider_for_model(model)
                stats = self._outcome_tracker.get_provider_stats(
                    provider=provider, model=model, days=days
                )
                if stats.total_calls > 0:
                    return stats.success_rate
            except Exception:
                logger.debug("Could not fetch quality for role=%s model=%s", agent_role, model)

        # Heuristic fallback based on pricing tier (more expensive = higher
        # assumed quality).
        return self._estimated_quality(model)

    def _get_avg_latency(self, agent_role: str, model: str, days: int) -> float:
        """Retrieve average latency in ms for a role+model pair."""
        if self._outcome_tracker is not None:
            try:
                provider = self._provider_for_model(model)
                stats = self._outcome_tracker.get_provider_stats(
                    provider=provider, model=model, days=days
                )
                if stats.total_calls > 0:
                    return stats.avg_latency_ms
            except Exception:
                logger.debug("Could not fetch latency for role=%s model=%s", agent_role, model)

        # Heuristic: assume latency roughly correlates with model size.
        return self._estimated_latency(model)

    def _find_best_alternative(
        self,
        role: str,
        current_model: str,
        current_quality: float,
        current_cost_per_call: float,
        monthly_calls: float,
        current_monthly: float,
        days: int,
    ) -> SavingsRecommendation | None:
        """Find the best cheaper alternative model for a role.

        Returns ``None`` if no alternative passes the quality and cost
        thresholds.
        """
        best: SavingsRecommendation | None = None

        for _provider, candidate_model in self._CANDIDATE_MODELS:
            if candidate_model == current_model:
                continue

            candidate_quality = self._get_quality_score(role, candidate_model, days)
            quality_diff_pct = (
                ((current_quality - candidate_quality) / current_quality) * 100
                if current_quality > 0
                else 0.0
            )

            # Skip if quality drops more than 10%.
            if quality_diff_pct > 10.0:
                continue

            price_ratio = self._pricing_ratio(current_model, candidate_model)
            projected_cost_per_call = current_cost_per_call * price_ratio
            projected_monthly = projected_cost_per_call * monthly_calls
            savings = current_monthly - projected_monthly

            # Skip if savings are less than 20% of current cost.
            if current_monthly > 0 and (savings / current_monthly) < 0.20:
                continue

            # Skip if no meaningful absolute savings.
            if savings < 0.01:
                continue

            quality_impact = self._describe_quality_impact(quality_diff_pct)
            confidence = self._recommendation_confidence(role, current_model, candidate_model, days)

            savings_pct = (savings / current_monthly * 100) if current_monthly > 0 else 0
            reasoning = (
                f"{role.capitalize()} uses {current_model} "
                f"(${current_cost_per_call:.4f}/call) — "
                f"{candidate_model} has "
                f"{candidate_quality:.0%} vs {current_quality:.0%} quality "
                f"for this role. "
                f"Switch to save ~${savings:.2f}/month ({savings_pct:.0f}%)."
            )

            rec = SavingsRecommendation(
                agent_role=role,
                current_model=current_model,
                recommended_model=candidate_model,
                current_cost_monthly=round(current_monthly, 2),
                projected_cost_monthly=round(projected_monthly, 2),
                estimated_savings_monthly=round(savings, 2),
                quality_impact=quality_impact,
                confidence=confidence,
                reasoning=reasoning,
            )

            if best is None or rec.estimated_savings_monthly > best.estimated_savings_monthly:
                best = rec

        return best

    def _pricing_ratio(self, from_model: str, to_model: str) -> float:
        """Compute the cost ratio between two models based on published pricing.

        Returns ``to_price / from_price``. A value < 1 means the target is
        cheaper.
        """
        pricing = CostTracker.PRICING
        from_pricing = pricing.get(from_model, {"input": 1.0, "output": 2.0})
        to_pricing = pricing.get(to_model, {"input": 1.0, "output": 2.0})

        # Use a weighted average (assume ~3:1 output:input token ratio for
        # generative tasks).
        from_avg = from_pricing["input"] * 0.25 + from_pricing["output"] * 0.75
        to_avg = to_pricing["input"] * 0.25 + to_pricing["output"] * 0.75

        if from_avg == 0:
            return 1.0
        return to_avg / from_avg

    def _estimated_quality(self, model: str) -> float:
        """Heuristic quality estimate based on model tier.

        Higher-priced models are assumed to have higher quality when no
        empirical data is available.
        """
        quality_map: dict[str, float] = {
            "gpt-4": 0.92,
            "gpt-4-turbo": 0.91,
            "gpt-4o": 0.90,
            "gpt-4o-mini": 0.72,
            "gpt-3.5-turbo": 0.65,
            "claude-3-opus": 0.93,
            "claude-3-sonnet": 0.88,
            "claude-sonnet-4-20250514": 0.92,
            "claude-3-5-sonnet": 0.91,
            "claude-3-haiku": 0.68,
            "claude-3-5-haiku": 0.74,
        }
        return quality_map.get(model, 0.75)

    def _estimated_latency(self, model: str) -> float:
        """Heuristic latency estimate in milliseconds."""
        latency_map: dict[str, float] = {
            "gpt-4": 3000.0,
            "gpt-4-turbo": 2000.0,
            "gpt-4o": 1500.0,
            "gpt-4o-mini": 600.0,
            "gpt-3.5-turbo": 500.0,
            "claude-3-opus": 3500.0,
            "claude-3-sonnet": 1800.0,
            "claude-sonnet-4-20250514": 1800.0,
            "claude-3-5-sonnet": 1700.0,
            "claude-3-haiku": 500.0,
            "claude-3-5-haiku": 600.0,
        }
        return latency_map.get(model, 1500.0)

    def _provider_for_model(self, model: str) -> str:
        """Map a model identifier to its provider name."""
        if model.startswith("gpt") or model.startswith("o1"):
            return "openai"
        if model.startswith("claude"):
            return "anthropic"
        return "unknown"

    def _describe_quality_impact(self, quality_diff_pct: float) -> str:
        """Return a human-readable quality impact label."""
        if quality_diff_pct <= 0:
            return "none"
        if quality_diff_pct < 5:
            return "minimal (<5%)"
        if quality_diff_pct <= 15:
            return "moderate (5-15%)"
        return "significant (>15%)"

    def _recommendation_confidence(
        self,
        role: str,
        current_model: str,
        candidate_model: str,
        days: int,
    ) -> str:
        """Determine recommendation confidence based on data availability."""
        if self._outcome_tracker is None:
            return "low"

        try:
            current_provider = self._provider_for_model(current_model)
            candidate_provider = self._provider_for_model(candidate_model)
            current_stats = self._outcome_tracker.get_provider_stats(
                provider=current_provider, model=current_model, days=days
            )
            candidate_stats = self._outcome_tracker.get_provider_stats(
                provider=candidate_provider, model=candidate_model, days=days
            )
            min_calls = min(current_stats.total_calls, candidate_stats.total_calls)
            if min_calls >= 50:
                return "high"
            if min_calls >= 10:
                return "medium"
        except Exception:
            logger.debug(
                "Could not determine confidence for %s -> %s",
                current_model,
                candidate_model,
            )

        return "low"

    def _assess_risk(self, quality_change_pct: float, cost_change_pct: float) -> str:
        """Assess the risk level of a model switch."""
        # quality_change_pct is negative when quality drops
        if quality_change_pct < -15:
            return "high"
        if quality_change_pct < -5:
            return "medium"
        return "low"

    def _compute_naive_baseline(self, entries: list[CostEntry]) -> float:
        """Compute what spending would be if every call used the priciest model.

        Uses the most expensive model per provider from PRICING.
        """
        # Find the single most expensive model overall.
        pricing = CostTracker.PRICING
        if not pricing:
            return sum(e.cost_usd for e in entries)

        max_model = max(
            pricing.items(),
            key=lambda item: item[1]["input"] * 0.25 + item[1]["output"] * 0.75,
        )
        max_model_name = max_model[0]

        total = 0.0
        for entry in entries:
            naive_cost = self._cost_tracker.calculate_cost(max_model_name, entry.tokens)
            total += naive_cost
        return total

    def _compute_enforcement_savings(self, entries: list[CostEntry]) -> float:
        """Estimate savings from contract enforcement preventing re-runs.

        Uses the outcome tracker's success rate to estimate how many calls
        would have failed without enforcement, assuming each failure would
        cost one additional re-run.
        """
        if self._outcome_tracker is None:
            return 0.0

        total_savings = 0.0
        # Group entries by (provider, model) for aggregate stats.
        groups: dict[tuple[str, str], list[CostEntry]] = defaultdict(list)
        for entry in entries:
            provider = self._provider_for_model(entry.model)
            groups[(provider, entry.model)].append(entry)

        for (provider, model), group_entries in groups.items():
            try:
                stats = self._outcome_tracker.get_provider_stats(provider=provider, model=model)
                if stats.total_calls == 0:
                    continue

                # Assume a baseline failure rate of 15% without enforcement,
                # and the tracked success rate reflects the enforcement benefit.
                baseline_failure_rate = 0.15
                actual_failure_rate = 1.0 - stats.success_rate
                prevented_failures = max(0.0, baseline_failure_rate - actual_failure_rate)

                group_cost = sum(e.cost_usd for e in group_entries)
                avg_call_cost = group_cost / len(group_entries)
                estimated_prevented_reruns = prevented_failures * len(group_entries)
                total_savings += estimated_prevented_reruns * avg_call_cost
            except Exception:
                logger.debug(
                    "Could not compute enforcement savings for %s/%s",
                    provider,
                    model,
                )

        return total_savings
