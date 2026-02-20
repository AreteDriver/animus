"""Skill performance analysis — detects declining, costly, and underperforming skills."""

from __future__ import annotations

import logging
from typing import Any

from .metrics import SkillMetricsAggregator
from .models import CapabilityGap, SkillMetrics

logger = logging.getLogger(__name__)

# Configurable thresholds
DECLINING_THRESHOLD: float = 0.05
LOW_SUCCESS_THRESHOLD: float = 0.50
COST_ANOMALY_MULTIPLIER: float = 2.0
MIN_INVOCATIONS_DEFAULT: int = 10


class SkillAnalyzer:
    """Analyzes skill metrics to find problems and opportunities.

    Args:
        aggregator: A ``SkillMetricsAggregator`` for querying skill stats.
        backend: A ``DatabaseBackend`` for querying raw outcome records.
    """

    def __init__(self, aggregator: SkillMetricsAggregator, backend: Any = None) -> None:
        self._aggregator = aggregator
        self._backend = backend

    def find_declining_skills(
        self,
        days: int = 30,
    ) -> list[tuple[str, SkillMetrics]]:
        """Find skills whose success rate is trending downward.

        Args:
            days: Look-back window.

        Returns:
            List of (skill_name, metrics) tuples for declining skills.
        """
        all_metrics = self._aggregator.get_all_skill_metrics(days)
        declining: list[tuple[str, SkillMetrics]] = []

        for m in all_metrics:
            if m.total_invocations < MIN_INVOCATIONS_DEFAULT:
                continue
            trend = self._aggregator.get_skill_trend(m.skill_name, m.skill_version or None, days)
            if trend == "declining":
                declining.append((m.skill_name, m))

        return declining

    def find_cost_anomalies(
        self,
        days: int = 30,
    ) -> list[tuple[str, str, float]]:
        """Find skills whose cost is significantly above the fleet average.

        Args:
            days: Look-back window.

        Returns:
            List of (skill_name, description, cost_ratio) tuples.
        """
        all_metrics = self._aggregator.get_all_skill_metrics(days)
        if not all_metrics:
            return []

        costs = [m.avg_cost_usd for m in all_metrics if m.avg_cost_usd > 0]
        if not costs:
            return []

        avg_fleet_cost = sum(costs) / len(costs)
        if avg_fleet_cost <= 0:
            return []

        anomalies: list[tuple[str, str, float]] = []
        for m in all_metrics:
            if m.total_invocations < MIN_INVOCATIONS_DEFAULT:
                continue
            ratio = m.avg_cost_usd / avg_fleet_cost
            if ratio > COST_ANOMALY_MULTIPLIER:
                desc = (
                    f"{m.skill_name} costs ${m.avg_cost_usd:.4f}/call "
                    f"({ratio:.1f}x fleet average ${avg_fleet_cost:.4f})"
                )
                anomalies.append((m.skill_name, desc, ratio))

        anomalies.sort(key=lambda x: -x[2])
        return anomalies

    def find_underperformers(
        self,
        success_threshold: float = LOW_SUCCESS_THRESHOLD,
        min_invocations: int = MIN_INVOCATIONS_DEFAULT,
        days: int = 30,
    ) -> list[tuple[str, SkillMetrics]]:
        """Find skills with success rates below the threshold.

        Args:
            success_threshold: Minimum acceptable success rate.
            min_invocations: Minimum calls to consider.
            days: Look-back window.

        Returns:
            List of (skill_name, metrics) for underperformers.
        """
        all_metrics = self._aggregator.get_all_skill_metrics(days)
        underperformers: list[tuple[str, SkillMetrics]] = []

        for m in all_metrics:
            if m.total_invocations < min_invocations:
                continue
            if m.success_rate < success_threshold:
                underperformers.append((m.skill_name, m))

        underperformers.sort(key=lambda x: x[1].success_rate)
        return underperformers

    def detect_capability_gaps(
        self,
        days: int = 30,
    ) -> list[CapabilityGap]:
        """Detect outcomes where no skill was invoked (empty skill_name).

        These represent tasks that the system handled without a matching
        skill — potential gaps in the skill library.

        Args:
            days: Look-back window.

        Returns:
            List of ``CapabilityGap`` objects.
        """
        if not self._backend:
            return []

        from datetime import UTC, datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        query = (
            "SELECT agent_role, COUNT(*) AS cnt, "
            "GROUP_CONCAT(DISTINCT workflow_id) AS workflows "
            "FROM outcome_records "
            "WHERE (skill_name = '' OR skill_name IS NULL) "
            "AND timestamp >= ? "
            "GROUP BY agent_role "
            "HAVING cnt >= 3 "
            "ORDER BY cnt DESC"
        )
        rows = self._backend.fetchall(query, (cutoff,))

        gaps: list[CapabilityGap] = []
        for row in rows:
            workflows = str(row.get("workflows", "")).split(",")[:5]
            gaps.append(
                CapabilityGap(
                    description=(
                        f"Agent role '{row['agent_role']}' executed {row['cnt']} times "
                        f"without a mapped skill"
                    ),
                    failure_contexts=workflows,
                    suggested_agent=str(row["agent_role"]),
                    confidence=min(0.5 + (int(row["cnt"]) * 0.02), 0.95),
                )
            )

        return gaps

    def generate_analysis_report(self, days: int = 30) -> dict[str, Any]:
        """Generate a full analysis report covering all dimensions.

        Args:
            days: Look-back window.

        Returns:
            Dict with keys: declining, cost_anomalies, underperformers,
            capability_gaps, summary.
        """
        declining = self.find_declining_skills(days)
        cost_anomalies = self.find_cost_anomalies(days)
        underperformers = self.find_underperformers(days=days)
        gaps = self.detect_capability_gaps(days)

        total_issues = len(declining) + len(cost_anomalies) + len(underperformers) + len(gaps)

        return {
            "declining": [(name, m.success_rate) for name, m in declining],
            "cost_anomalies": [(name, desc, ratio) for name, desc, ratio in cost_anomalies],
            "underperformers": [(name, m.success_rate) for name, m in underperformers],
            "capability_gaps": [
                {
                    "description": g.description,
                    "agent": g.suggested_agent,
                    "confidence": g.confidence,
                }
                for g in gaps
            ],
            "total_issues": total_issues,
            "summary": (
                f"Found {total_issues} issues: {len(declining)} declining, "
                f"{len(cost_anomalies)} cost anomalies, {len(underperformers)} underperformers, "
                f"{len(gaps)} capability gaps."
            ),
        }
