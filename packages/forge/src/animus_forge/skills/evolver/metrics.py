"""Skill-level metrics aggregation from outcome_records."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from animus_forge.state.backends import DatabaseBackend

from .models import SkillMetrics

logger = logging.getLogger(__name__)


class SkillMetricsAggregator:
    """Aggregates outcome_records into per-skill performance metrics.

    Thread-safe aggregator backed by a ``DatabaseBackend``.  Queries the
    ``skill_name`` / ``skill_version`` columns added by migration 015 and
    stores materialised results in the ``skill_metrics`` table.

    Args:
        backend: A ``DatabaseBackend`` instance.
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cutoff_iso(days: int) -> str:
        return (datetime.now(UTC) - timedelta(days=days)).isoformat()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _row_to_metrics(row: dict) -> SkillMetrics:
        return SkillMetrics(
            skill_name=str(row["skill_name"]),
            skill_version=str(row.get("skill_version", "")),
            period_start=str(row.get("period_start", "")),
            period_end=str(row.get("period_end", "")),
            total_invocations=int(row.get("total_invocations", 0)),
            success_count=int(row.get("success_count", 0)),
            failure_count=int(row.get("failure_count", 0)),
            success_rate=float(row.get("success_rate", 0.0)),
            avg_quality_score=float(row.get("avg_quality_score", 0.0)),
            avg_cost_usd=float(row.get("avg_cost_usd", 0.0)),
            avg_latency_ms=float(row.get("avg_latency_ms", 0.0)),
            total_cost_usd=float(row.get("total_cost_usd", 0.0)),
            trend=str(row.get("trend", "stable")),
            computed_at=str(row.get("computed_at", "")),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_skill_metrics(
        self,
        skill_name: str,
        version: str | None = None,
        days: int = 30,
    ) -> SkillMetrics | None:
        """Get aggregated metrics for a skill, computed on the fly.

        Args:
            skill_name: Skill to query.
            version: Optional version filter.
            days: Look-back window in days.

        Returns:
            ``SkillMetrics`` or ``None`` if no data.
        """
        cutoff = self._cutoff_iso(days)
        now = self._now_iso()

        if version is not None:
            query = (
                "SELECT skill_name, skill_version, "
                "COUNT(*) AS total_invocations, "
                "SUM(success) AS success_count, "
                "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failure_count, "
                "AVG(success) AS success_rate, "
                "AVG(quality_score) AS avg_quality_score, "
                "AVG(cost_usd) AS avg_cost_usd, "
                "AVG(latency_ms) AS avg_latency_ms, "
                "SUM(cost_usd) AS total_cost_usd "
                "FROM outcome_records "
                "WHERE skill_name = ? AND skill_version = ? AND timestamp >= ? "
                "GROUP BY skill_name, skill_version"
            )
            params: tuple = (skill_name, version, cutoff)
        else:
            query = (
                "SELECT skill_name, '' AS skill_version, "
                "COUNT(*) AS total_invocations, "
                "SUM(success) AS success_count, "
                "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failure_count, "
                "AVG(success) AS success_rate, "
                "AVG(quality_score) AS avg_quality_score, "
                "AVG(cost_usd) AS avg_cost_usd, "
                "AVG(latency_ms) AS avg_latency_ms, "
                "SUM(cost_usd) AS total_cost_usd "
                "FROM outcome_records "
                "WHERE skill_name = ? AND timestamp >= ? "
                "GROUP BY skill_name"
            )
            params = (skill_name, cutoff)

        with self._lock:
            row = self._backend.fetchone(query, params)

        if not row or row.get("total_invocations", 0) == 0:
            return None

        return SkillMetrics(
            skill_name=str(row["skill_name"]),
            skill_version=str(row.get("skill_version", "")),
            period_start=cutoff,
            period_end=now,
            total_invocations=int(row["total_invocations"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            success_rate=float(row["success_rate"]),
            avg_quality_score=float(row["avg_quality_score"]),
            avg_cost_usd=float(row["avg_cost_usd"]),
            avg_latency_ms=float(row["avg_latency_ms"]),
            total_cost_usd=float(row["total_cost_usd"]),
            computed_at=now,
        )

    def get_all_skill_metrics(self, days: int = 30) -> list[SkillMetrics]:
        """Get metrics for all skills with recorded outcomes.

        Args:
            days: Look-back window in days.

        Returns:
            List of ``SkillMetrics``, one per skill_name.
        """
        cutoff = self._cutoff_iso(days)
        now = self._now_iso()

        query = (
            "SELECT skill_name, skill_version, "
            "COUNT(*) AS total_invocations, "
            "SUM(success) AS success_count, "
            "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failure_count, "
            "AVG(success) AS success_rate, "
            "AVG(quality_score) AS avg_quality_score, "
            "AVG(cost_usd) AS avg_cost_usd, "
            "AVG(latency_ms) AS avg_latency_ms, "
            "SUM(cost_usd) AS total_cost_usd "
            "FROM outcome_records "
            "WHERE skill_name != '' AND timestamp >= ? "
            "GROUP BY skill_name, skill_version "
            "ORDER BY total_invocations DESC"
        )

        with self._lock:
            rows = self._backend.fetchall(query, (cutoff,))

        results: list[SkillMetrics] = []
        for row in rows:
            results.append(
                SkillMetrics(
                    skill_name=str(row["skill_name"]),
                    skill_version=str(row.get("skill_version", "")),
                    period_start=cutoff,
                    period_end=now,
                    total_invocations=int(row["total_invocations"]),
                    success_count=int(row["success_count"]),
                    failure_count=int(row["failure_count"]),
                    success_rate=float(row["success_rate"]),
                    avg_quality_score=float(row["avg_quality_score"]),
                    avg_cost_usd=float(row["avg_cost_usd"]),
                    avg_latency_ms=float(row["avg_latency_ms"]),
                    total_cost_usd=float(row["total_cost_usd"]),
                    computed_at=now,
                )
            )
        return results

    def get_skill_trend(
        self,
        skill_name: str,
        version: str | None = None,
        days: int = 30,
    ) -> str:
        """Determine whether a skill is improving, declining, or stable.

        Splits the window in half and compares success rates.

        Args:
            skill_name: Skill to analyze.
            version: Optional version filter.
            days: Look-back window.

        Returns:
            ``"improving"``, ``"declining"``, or ``"stable"``.
        """
        now = datetime.now(UTC)
        midpoint = (now - timedelta(days=days // 2)).isoformat()
        cutoff = (now - timedelta(days=days)).isoformat()

        if version is not None:
            where = "skill_name = ? AND skill_version = ? AND timestamp >= ?"
            params_early: tuple = (skill_name, version, cutoff)
            params_late: tuple = (skill_name, version, midpoint)
        else:
            where = "skill_name = ? AND timestamp >= ?"
            params_early = (skill_name, cutoff)
            params_late = (skill_name, midpoint)

        q_early = (
            f"SELECT AVG(success) AS rate FROM outcome_records WHERE {where} AND timestamp < ?"
        )
        q_late = f"SELECT AVG(success) AS rate FROM outcome_records WHERE {where}"

        with self._lock:
            row_early = self._backend.fetchone(q_early, (*params_early, midpoint))
            row_late = self._backend.fetchone(q_late, params_late)

        early_rate = (
            float(row_early["rate"]) if row_early and row_early.get("rate") is not None else 0.0
        )
        late_rate = (
            float(row_late["rate"]) if row_late and row_late.get("rate") is not None else 0.0
        )

        diff = late_rate - early_rate
        if diff > 0.05:
            return "improving"
        if diff < -0.05:
            return "declining"
        return "stable"

    def compute_and_store_metrics(self, days: int = 30) -> int:
        """Compute metrics for all skills and persist to skill_metrics table.

        Args:
            days: Look-back window.

        Returns:
            Number of rows written.
        """
        metrics = self.get_all_skill_metrics(days)
        if not metrics:
            return 0

        # Compute trends
        enriched: list[SkillMetrics] = []
        for m in metrics:
            trend = self.get_skill_trend(m.skill_name, m.skill_version or None, days)
            enriched.append(
                SkillMetrics(
                    skill_name=m.skill_name,
                    skill_version=m.skill_version,
                    period_start=m.period_start,
                    period_end=m.period_end,
                    total_invocations=m.total_invocations,
                    success_count=m.success_count,
                    failure_count=m.failure_count,
                    success_rate=m.success_rate,
                    avg_quality_score=m.avg_quality_score,
                    avg_cost_usd=m.avg_cost_usd,
                    avg_latency_ms=m.avg_latency_ms,
                    total_cost_usd=m.total_cost_usd,
                    trend=trend,
                    computed_at=m.computed_at,
                )
            )

        query = (
            "INSERT OR REPLACE INTO skill_metrics "
            "(skill_name, skill_version, period_start, period_end, "
            "total_invocations, success_count, failure_count, success_rate, "
            "avg_quality_score, avg_cost_usd, avg_latency_ms, total_cost_usd, "
            "trend, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        rows_data = [
            (
                m.skill_name,
                m.skill_version,
                m.period_start,
                m.period_end,
                m.total_invocations,
                m.success_count,
                m.failure_count,
                m.success_rate,
                m.avg_quality_score,
                m.avg_cost_usd,
                m.avg_latency_ms,
                m.total_cost_usd,
                m.trend,
                m.computed_at,
            )
            for m in enriched
        ]

        with self._lock:
            with self._backend.transaction():
                self._backend.executemany(query, rows_data)

        return len(rows_data)

    def get_comparative_metrics(
        self,
        skill_name: str,
        version_a: str,
        version_b: str,
        days: int = 30,
    ) -> tuple[SkillMetrics | None, SkillMetrics | None]:
        """Get metrics for two versions of a skill side by side.

        Args:
            skill_name: Skill to compare.
            version_a: First version.
            version_b: Second version.
            days: Look-back window.

        Returns:
            Tuple of (metrics_a, metrics_b), either may be ``None``.
        """
        a = self.get_skill_metrics(skill_name, version_a, days)
        b = self.get_skill_metrics(skill_name, version_b, days)
        return (a, b)
