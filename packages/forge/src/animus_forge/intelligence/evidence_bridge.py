"""Evidence bridge: closes the loop from eval results to memory learnings.

When an evaluation suite completes, the bridge decides whether the outcome
warrants a persistent learning, feeds per-case outcomes to the
OutcomeTracker, and returns a ``MissionEvidence`` record that links the
eval run to its originating workflow and mission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from animus_forge.intelligence.outcome_tracker import OutcomeRecord

if TYPE_CHECKING:
    from animus_forge.evaluation.runner import SuiteResult
    from animus_forge.evaluation.store import EvalStore
    from animus_forge.intelligence.cross_workflow_memory import CrossWorkflowMemory
    from animus_forge.intelligence.outcome_tracker import OutcomeTracker

logger = logging.getLogger(__name__)

# Thresholds for auto-learning (mirrors Phase-1 plan)
_AUTO_LEARN_PASS_RATE_THRESHOLD = 0.8
_AUTO_LEARN_VARIANCE_THRESHOLD = 0.15
_MIN_IMPORTANCE = 0.2


@dataclass
class MissionEvidence:
    """Snapshot of evidence produced by a single eval run within a mission."""

    mission_id: str
    workflow_id: str | None
    run_id: str
    suite_name: str
    pass_rate: float
    score_variance: float
    total_cases: int
    failed_cases: int
    learned_insights: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "pass_rate": self.pass_rate,
            "score_variance": self.score_variance,
            "total_cases": self.total_cases,
            "failed_cases": self.failed_cases,
            "learned_insights": self.learned_insights,
            "timestamp": self.timestamp.isoformat(),
        }


class EvidenceBridge:
    """Closes the eval → evidence → memory loop.

    The bridge is intentionally decoupled from both the runner and the
    orchestrator.  It is invoked by whichever layer owns the mission
    lifecycle (e.g. a ResearchCitizen or an API route).

    Args:
        eval_store: Persistent store for eval suite runs.
        outcome_tracker: Tracker for per-step outcomes.
        cross_memory: Global cross-workflow memory for learnings.
        auto_learn: Whether to auto-record learnings on poor results.
    """

    def __init__(
        self,
        eval_store: EvalStore,
        outcome_tracker: OutcomeTracker,
        cross_memory: CrossWorkflowMemory,
        auto_learn: bool = True,
    ):
        self.eval_store = eval_store
        self.outcome_tracker = outcome_tracker
        self.cross_memory = cross_memory
        self.auto_learn = auto_learn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_eval_complete(
        self,
        suite_result: SuiteResult,
        *,
        workflow_id: str | None = None,
        mission_id: str | None = None,
        agent_role: str | None = None,
        model: str | None = None,
        run_mode: str = "live",
    ) -> MissionEvidence:
        """Process a completed evaluation run.

        1. Records the run in ``EvalStore`` with mission metadata.
        2. Feeds per-case outcomes to ``OutcomeTracker``.
        3. Optionally records a learning in ``CrossWorkflowMemory``.

        Returns:
            A ``MissionEvidence`` linking the eval run to the mission.
        """
        mission_id = mission_id or "orphan"
        workflow_id = workflow_id or "orphan"
        agent_role = agent_role or "unknown"
        suite_name = suite_result.suite.name

        # 1. Record eval run with mission context
        metadata: dict[str, Any] = {
            "source": "evidence_bridge",
            "mission_id": mission_id,
            "workflow_id": workflow_id,
        }
        run_id = self.eval_store.record_run(
            suite_name=suite_name,
            result=suite_result,
            agent_role=agent_role,
            model=model,
            run_mode=run_mode,
            metadata=metadata,
        )

        # 2. Feed outcomes
        self._feed_outcomes(suite_result, run_id, workflow_id, agent_role, model)

        # 3. Auto-learning
        insights: list[str] = []
        if self.auto_learn and self._should_learn(suite_result):
            insight = self._build_insight(suite_result, suite_name)
            importance = max(_MIN_IMPORTANCE, 1.0 - suite_result.pass_rate)
            tags = [
                "regression",
                "eval_failure",
                f"suite:{suite_name}",
            ]
            memory_id = self.cross_memory.record_learning(
                agent_role=agent_role,
                insight=insight,
                source_workflow_id=workflow_id,
                importance=importance,
                tags=tags,
            )
            insights.append(f"memory:{memory_id}")
            logger.info(
                "Auto-learning recorded for mission %s (importance %.2f)",
                mission_id,
                importance,
            )

        evidence = MissionEvidence(
            mission_id=mission_id,
            workflow_id=workflow_id,
            run_id=run_id,
            suite_name=suite_name,
            pass_rate=suite_result.pass_rate,
            score_variance=suite_result.score_variance,
            total_cases=suite_result.total,
            failed_cases=suite_result.failed + suite_result.errors,
            learned_insights=insights,
        )
        return evidence

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_learn(self, result: SuiteResult) -> bool:
        """Return True if the result warrants a learning entry."""
        return (
            result.pass_rate < _AUTO_LEARN_PASS_RATE_THRESHOLD
            or result.score_variance > _AUTO_LEARN_VARIANCE_THRESHOLD
        )

    def _build_insight(self, result: SuiteResult, suite_name: str) -> str:
        failed = [r for r in result.results if r.status.value in ("failed", "error")]
        failed_names = [r.case.name for r in failed[:5]]
        return (
            f"Suite '{suite_name}' scored {result.pass_rate:.0%} "
            f"with variance {result.score_variance:.2f}. "
            f"Failed cases ({len(failed)}): {', '.join(failed_names) or 'none'}."
        )

    def _feed_outcomes(
        self,
        result: SuiteResult,
        run_id: str,
        workflow_id: str,
        agent_role: str,
        model: str | None,
    ) -> None:
        """Create OutcomeRecord entries for each case result."""
        if not result.results:
            return

        records: list[OutcomeRecord] = []
        for cr in result.results:
            records.append(
                OutcomeRecord(
                    step_id=f"eval-{run_id[:8]}-{cr.case.name}",
                    workflow_id=workflow_id,
                    agent_role=agent_role,
                    provider="eval",
                    model=model or "unknown",
                    success=cr.status.value == "passed",
                    quality_score=cr.score,
                    cost_usd=0.0,
                    tokens_used=cr.tokens_used,
                    latency_ms=cr.latency_ms,
                    metadata={
                        "source": "evidence_bridge",
                        "run_id": run_id,
                        "suite": result.suite.name,
                        "case": cr.case.name,
                    },
                )
            )

        self.outcome_tracker.record_many(records)
