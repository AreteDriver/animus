"""A/B testing for skill versions — deterministic traffic routing."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime

from animus_forge.state.backends import DatabaseBackend

from .metrics import SkillMetricsAggregator
from .models import ExperimentConfig, ExperimentResult, SkillMetrics

logger = logging.getLogger(__name__)


class ABTestManager:
    """Manages A/B test experiments between skill versions.

    Provides deterministic routing based on workflow_id hashing, so the same
    workflow always sees the same variant within an experiment.

    Args:
        backend: A ``DatabaseBackend`` instance.
        aggregator: A ``SkillMetricsAggregator`` for evaluating experiments.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        aggregator: SkillMetricsAggregator,
    ) -> None:
        self._backend = backend
        self._aggregator = aggregator
        self._lock = threading.Lock()

    def create_experiment(
        self,
        skill_name: str,
        control_version: str,
        variant_version: str,
        traffic_split: float = 0.5,
        min_invocations: int = 100,
    ) -> ExperimentConfig:
        """Create a new A/B test experiment.

        Args:
            skill_name: Skill being tested.
            control_version: Existing version (control).
            variant_version: New version to test (variant).
            traffic_split: Fraction of traffic to send to variant (0.0-1.0).
            min_invocations: Minimum calls before experiment can conclude.

        Returns:
            The experiment configuration.
        """
        experiment_id = str(uuid.uuid4())[:8]
        now = datetime.now(UTC).isoformat()

        query = (
            "INSERT INTO skill_experiments "
            "(id, skill_name, control_version, variant_version, "
            "traffic_split, status, min_invocations, start_date, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with self._lock:
            with self._backend.transaction():
                self._backend.execute(
                    query,
                    (
                        experiment_id,
                        skill_name,
                        control_version,
                        variant_version,
                        traffic_split,
                        "active",
                        min_invocations,
                        now,
                        now,
                    ),
                )

        logger.info(
            "Created experiment %s: %s v%s vs v%s (%.0f%% traffic split)",
            experiment_id,
            skill_name,
            control_version,
            variant_version,
            traffic_split * 100,
        )

        return ExperimentConfig(
            experiment_id=experiment_id,
            skill_name=skill_name,
            control_version=control_version,
            variant_version=variant_version,
            traffic_split=traffic_split,
            min_invocations=min_invocations,
            start_date=now,
        )

    def route_skill_version(self, skill_name: str, workflow_id: str) -> str | None:
        """Route a skill invocation to the correct version for any active experiment.

        Uses deterministic hashing so the same workflow always sees the same version.

        Args:
            skill_name: Skill being invoked.
            workflow_id: Workflow identifier for deterministic routing.

        Returns:
            Version string to use, or ``None`` if no active experiment.
        """
        query = "SELECT * FROM skill_experiments WHERE skill_name = ? AND status = 'active' LIMIT 1"
        with self._lock:
            row = self._backend.fetchone(query, (skill_name,))

        if not row:
            return None

        traffic_split = float(row["traffic_split"])
        bucket = hash(workflow_id) % 100

        if bucket < int(traffic_split * 100):
            return str(row["variant_version"])
        return str(row["control_version"])

    def evaluate_experiment(self, experiment_id: str) -> ExperimentResult | None:
        """Evaluate an experiment to see if it can be concluded.

        Returns ``None`` if insufficient data (below min_invocations).

        Args:
            experiment_id: The experiment to evaluate.

        Returns:
            ``ExperimentResult`` if sufficient data, else ``None``.
        """
        query = "SELECT * FROM skill_experiments WHERE id = ?"
        with self._lock:
            row = self._backend.fetchone(query, (experiment_id,))

        if not row:
            return None

        skill_name = str(row["skill_name"])
        control_version = str(row["control_version"])
        variant_version = str(row["variant_version"])
        min_invocations = int(row["min_invocations"])

        control_metrics = self._aggregator.get_skill_metrics(skill_name, control_version)
        variant_metrics = self._aggregator.get_skill_metrics(skill_name, variant_version)

        # Check minimum data threshold
        control_count = control_metrics.total_invocations if control_metrics else 0
        variant_count = variant_metrics.total_invocations if variant_metrics else 0

        if control_count + variant_count < min_invocations:
            return None

        # Determine winner based on success_rate and quality_score
        control_score = self._composite_score(control_metrics)
        variant_score = self._composite_score(variant_metrics)

        if variant_score > control_score + 0.02:
            winner = variant_version
            reason = f"Variant outperformed control ({variant_score:.3f} vs {control_score:.3f})"
        elif control_score > variant_score + 0.02:
            winner = control_version
            reason = f"Control outperformed variant ({control_score:.3f} vs {variant_score:.3f})"
        else:
            winner = control_version
            reason = "No significant difference — defaulting to control"

        significance = abs(variant_score - control_score)

        return ExperimentResult(
            experiment_id=experiment_id,
            skill_name=skill_name,
            control_version=control_version,
            variant_version=variant_version,
            control_metrics=control_metrics,
            variant_metrics=variant_metrics,
            winner=winner,
            statistical_significance=min(significance * 10, 1.0),
            conclusion_reason=reason,
        )

    def conclude_experiment(
        self,
        experiment_id: str,
        winner: str,
        reason: str,
    ) -> None:
        """Mark an experiment as concluded.

        Args:
            experiment_id: Experiment to conclude.
            winner: Winning version string.
            reason: Reason for conclusion.
        """
        now = datetime.now(UTC).isoformat()
        query = (
            "UPDATE skill_experiments "
            "SET status = 'concluded', winner = ?, conclusion_reason = ?, "
            "end_date = ?, concluded_at = ? "
            "WHERE id = ?"
        )
        with self._lock:
            with self._backend.transaction():
                self._backend.execute(query, (winner, reason, now, now, experiment_id))

        logger.info("Concluded experiment %s: winner=%s", experiment_id, winner)

    def cancel_experiment(self, experiment_id: str, reason: str) -> None:
        """Cancel an active experiment.

        Args:
            experiment_id: Experiment to cancel.
            reason: Reason for cancellation.
        """
        now = datetime.now(UTC).isoformat()
        query = (
            "UPDATE skill_experiments "
            "SET status = 'cancelled', conclusion_reason = ?, "
            "end_date = ?, concluded_at = ? "
            "WHERE id = ?"
        )
        with self._lock:
            with self._backend.transaction():
                self._backend.execute(query, (reason, now, now, experiment_id))

        logger.info("Cancelled experiment %s: %s", experiment_id, reason)

    def promote_winner(self, experiment_id: str) -> dict | None:
        """Get the winning version for promotion.

        Args:
            experiment_id: Concluded experiment.

        Returns:
            Dict with skill_name and winner version, or ``None``.
        """
        query = (
            "SELECT skill_name, winner FROM skill_experiments "
            "WHERE id = ? AND status = 'concluded' AND winner IS NOT NULL"
        )
        with self._lock:
            row = self._backend.fetchone(query, (experiment_id,))

        if not row or not row.get("winner"):
            return None

        return {
            "skill_name": str(row["skill_name"]),
            "winner": str(row["winner"]),
        }

    def get_active_experiments(self) -> list[dict]:
        """Get all active experiments.

        Returns:
            List of experiment dicts.
        """
        query = "SELECT * FROM skill_experiments WHERE status = 'active'"
        with self._lock:
            return self._backend.fetchall(query, ())

    @staticmethod
    def _composite_score(metrics: SkillMetrics | None) -> float:
        """Compute a composite score from metrics for comparison.

        Weights: success_rate (0.6) + avg_quality_score (0.4).
        """
        if not metrics:
            return 0.0
        return (metrics.success_rate * 0.6) + (metrics.avg_quality_score * 0.4)
