"""Feedback loop engine for the Gorgon intelligence layer.

Closes the loop between execution outcomes and future agent behavior by
analyzing step and workflow results, detecting notable events, generating
learnings, and feeding insights back into the memory, routing, and outcome
tracking systems.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from animus_forge.intelligence.outcome_tracker import OutcomeRecord, OutcomeTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeedbackResult:
    """Result of processing a single step execution.

    Attributes:
        step_id: Identifier for the step that was processed.
        outcome_recorded: Whether the outcome was persisted successfully.
        learning_generated: The insight string, if a notable event occurred.
        provider_updated: Whether the provider router was notified.
        notable: Whether this step was flagged as a significant event.
    """

    step_id: str
    outcome_recorded: bool
    learning_generated: str | None
    provider_updated: bool
    notable: bool


@dataclass
class WorkflowFeedback:
    """Aggregate feedback for a completed workflow run.

    Attributes:
        workflow_id: Identifier for the workflow.
        total_steps: Number of steps in the workflow.
        successful_steps: Number of steps that succeeded.
        total_cost_usd: Cumulative cost of the workflow in USD.
        insights: Human-readable insights derived from the run.
        learnings_stored: Number of new learnings persisted to memory.
    """

    workflow_id: str
    total_steps: int
    successful_steps: int
    total_cost_usd: float
    insights: list[str]
    learnings_stored: int


@dataclass
class AgentTrajectory:
    """Trend analysis for an agent role over a time period.

    Attributes:
        agent_role: The role being analyzed.
        period_days: Length of the analysis window in days.
        current_success_rate: Success rate in the recent half of the period.
        previous_success_rate: Success rate in the earlier half of the period.
        trend: One of ``"improving"``, ``"declining"``, or ``"stable"``.
        avg_quality_score: Mean quality score across all executions in the period.
        avg_cost_per_call: Mean cost per call in USD.
        total_executions: Total number of step executions in the period.
    """

    agent_role: str
    period_days: int
    current_success_rate: float
    previous_success_rate: float
    trend: str
    avg_quality_score: float
    avg_cost_per_call: float
    total_executions: int


@dataclass
class Suggestion:
    """An actionable suggestion for improving a workflow.

    Attributes:
        step_id: The step this suggestion applies to, or ``None`` for
            workflow-level suggestions.
        category: Type of suggestion (e.g. ``"provider_upgrade"``).
        description: Human-readable explanation.
        estimated_impact: Expected impact level (``"high"``, ``"medium"``,
            or ``"low"``).
        confidence: Confidence in the suggestion from 0.0 to 1.0.
    """

    step_id: str | None
    category: str
    description: str
    estimated_impact: str
    confidence: float


# ---------------------------------------------------------------------------
# Notability thresholds
# ---------------------------------------------------------------------------

_LATENCY_MULTIPLIER_THRESHOLD: float = 2.0
_COST_MULTIPLIER_THRESHOLD: float = 3.0
_LOW_QUALITY_THRESHOLD: float = 0.3
_TREND_CHANGE_THRESHOLD: float = 0.05
_SUCCESS_RATE_CONCERN_THRESHOLD: float = 0.70
_HIGH_COST_FRACTION: float = 0.30


# ---------------------------------------------------------------------------
# FeedbackEngine
# ---------------------------------------------------------------------------


class FeedbackEngine:
    """Analyzes workflow outcomes and feeds insights back into the intelligence layer.

    Acts as the central integration point between the ``OutcomeTracker``,
    ``CrossWorkflowMemory``, and ``ProviderRouter``, ensuring that every
    execution result is recorded, analyzed, and — when notable — turned into
    a persistent learning that improves future agent behaviour.

    Args:
        outcome_tracker: Tracker for recording and querying step outcomes.
        cross_memory: Cross-workflow memory for storing and retrieving
            agent learnings.
        provider_router: Router that adapts provider selection based on
            execution outcomes.
    """

    def __init__(
        self,
        outcome_tracker: OutcomeTracker,
        cross_memory: Any,  # CrossWorkflowMemory
        provider_router: Any,  # ProviderRouter
    ) -> None:
        self._tracker = outcome_tracker
        self._memory = cross_memory
        self._router = provider_router
        # Cache of seen (agent_role, provider, model) combos for baseline detection
        self._seen_combos: set[tuple[str, str, str]] = set()
        self._load_seen_combos()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_seen_combos(self) -> None:
        """Pre-populate the seen combos set from existing outcome records.

        Queries the outcome tracker's backend directly to avoid a full table
        scan on every ``process_step_result`` call.
        """
        try:
            query = "SELECT DISTINCT agent_role, provider, model FROM outcome_records"
            with self._tracker._lock:
                rows = self._tracker._backend.fetchall(query, ())
            for row in rows:
                combo = (
                    str(row["agent_role"]),
                    str(row["provider"]),
                    str(row["model"]),
                )
                self._seen_combos.add(combo)
        except Exception:
            logger.debug("Could not pre-load seen combos; starting fresh.")

    # ------------------------------------------------------------------
    # Step-level feedback
    # ------------------------------------------------------------------

    def process_step_result(
        self,
        step_id: str,
        workflow_id: str,
        agent_role: str,
        provider: str,
        model: str,
        step_result: dict[str, Any],
        cost_usd: float,
        tokens_used: int,
    ) -> FeedbackResult:
        """Process the result of a single workflow step execution.

        Records the outcome, updates the provider router, detects notability,
        and generates learnings when warranted.

        Args:
            step_id: Unique identifier for this step execution.
            workflow_id: Parent workflow run identifier.
            agent_role: Role of the agent that executed the step.
            provider: AI provider used (e.g. ``"openai"``).
            model: Model identifier (e.g. ``"gpt-4o"``).
            step_result: Dictionary containing at least ``"success"`` (bool),
                optionally ``"quality_score"`` (float 0-1),
                ``"latency_ms"`` (float), and ``"metadata"`` (dict).
            cost_usd: Cost of the API call in USD.
            tokens_used: Total tokens consumed.

        Returns:
            A ``FeedbackResult`` summarising what actions were taken.
        """
        success: bool = step_result.get("success", False)
        quality_score: float = step_result.get("quality_score", 1.0 if success else 0.0)
        latency_ms: float = step_result.get("latency_ms", 0.0)
        metadata: dict[str, Any] = step_result.get("metadata", {})

        # 1. Record the outcome
        outcome = OutcomeRecord(
            step_id=step_id,
            workflow_id=workflow_id,
            agent_role=agent_role,
            provider=provider,
            model=model,
            success=success,
            quality_score=quality_score,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata=metadata,
        )
        outcome_recorded = False
        try:
            self._tracker.record(outcome)
            outcome_recorded = True
        except Exception:
            logger.exception("Failed to record outcome for step %s", step_id)

        # 2. Update provider router
        provider_updated = False
        try:
            self._router.update_after_execution(agent_role, provider, model, outcome)
            provider_updated = True
        except Exception:
            logger.exception("Failed to update provider router for step %s", step_id)

        # 3. Detect notability and generate learnings
        notable, learning = self._evaluate_notability(
            step_id=step_id,
            workflow_id=workflow_id,
            agent_role=agent_role,
            provider=provider,
            model=model,
            success=success,
            quality_score=quality_score,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

        learning_generated: str | None = None
        if notable and learning:
            learning_generated = learning
            try:
                importance = self._learning_importance(success, quality_score)
                self._memory.record_learning(
                    agent_role=agent_role,
                    insight=learning,
                    source_workflow_id=workflow_id,
                    importance=importance,
                    tags=self._learning_tags(success, quality_score, cost_usd, latency_ms),
                )
            except Exception:
                logger.exception("Failed to store learning for step %s", step_id)

        return FeedbackResult(
            step_id=step_id,
            outcome_recorded=outcome_recorded,
            learning_generated=learning_generated,
            provider_updated=provider_updated,
            notable=notable,
        )

    # ------------------------------------------------------------------
    # Workflow-level feedback
    # ------------------------------------------------------------------

    def process_workflow_result(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_result: dict[str, Any],
    ) -> WorkflowFeedback:
        """Analyze a completed workflow and generate aggregate feedback.

        Retrieves all step outcomes for the workflow, computes summary
        statistics, generates workflow-level insights, stores learnings,
        and triggers memory decay.

        Args:
            workflow_id: Identifier for the workflow run.
            workflow_name: Human-readable name of the workflow.
            execution_result: Dictionary with at least ``"steps"`` — a list
                of step result dicts each containing ``"step_id"``,
                ``"agent_role"``, and ``"success"``.

        Returns:
            A ``WorkflowFeedback`` summarising the analysis.
        """
        outcomes = self._tracker.get_workflow_outcomes(workflow_id)

        total_steps = len(outcomes)
        successful_steps = sum(1 for o in outcomes if o.success)
        total_cost = sum(o.cost_usd for o in outcomes)

        insights: list[str] = []
        learnings_stored = 0

        if total_steps == 0:
            insights.append(
                f"Workflow '{workflow_name}' completed but no step outcomes were recorded."
            )
            return WorkflowFeedback(
                workflow_id=workflow_id,
                total_steps=0,
                successful_steps=0,
                total_cost_usd=0.0,
                insights=insights,
                learnings_stored=0,
            )

        success_rate = successful_steps / total_steps
        insights.append(
            f"Workflow '{workflow_name}' finished with {successful_steps}/{total_steps} "
            f"steps succeeding ({success_rate:.0%}), total cost ${total_cost:.4f}."
        )

        # Analyze per-role performance within the workflow
        role_outcomes: dict[str, list[OutcomeRecord]] = {}
        for o in outcomes:
            role_outcomes.setdefault(o.agent_role, []).append(o)

        for role, records in role_outcomes.items():
            role_successes = sum(1 for r in records if r.success)
            role_total = len(records)
            role_rate = role_successes / role_total

            if role_rate < 1.0 and role_total > 1:
                insights.append(
                    f"{role.capitalize()} succeeded {role_successes}/{role_total} "
                    f"times ({role_rate:.0%})."
                )

        # Detect expensive steps
        if total_cost > 0:
            for o in outcomes:
                fraction = o.cost_usd / total_cost
                if fraction > _HIGH_COST_FRACTION:
                    insights.append(
                        f"Step '{o.step_id}' ({o.agent_role}) consumed "
                        f"{fraction:.0%} of total workflow cost."
                    )

        # Detect sequential failure patterns
        consecutive_failures = 0
        max_consecutive = 0
        for o in outcomes:
            if not o.success:
                consecutive_failures += 1
                max_consecutive = max(max_consecutive, consecutive_failures)
            else:
                consecutive_failures = 0
        if max_consecutive >= 2:
            insights.append(
                f"{max_consecutive} consecutive step failures detected — "
                f"consider adding early termination or fallback logic."
            )

        # Store workflow-level learnings
        for insight in insights[1:]:  # skip the summary line
            try:
                self._memory.record_learning(
                    agent_role="orchestrator",
                    insight=f"[{workflow_name}] {insight}",
                    source_workflow_id=workflow_id,
                    importance=0.6,
                    tags=["workflow_analysis", workflow_name],
                )
                learnings_stored += 1
            except Exception:
                logger.exception("Failed to store workflow learning")

        # Trigger memory decay
        try:
            self._memory.decay_memories(half_life_days=90)
        except Exception:
            logger.debug("Memory decay failed; non-critical.", exc_info=True)

        return WorkflowFeedback(
            workflow_id=workflow_id,
            total_steps=total_steps,
            successful_steps=successful_steps,
            total_cost_usd=total_cost,
            insights=insights,
            learnings_stored=learnings_stored,
        )

    # ------------------------------------------------------------------
    # Trajectory analysis
    # ------------------------------------------------------------------

    def analyze_agent_trajectory(self, agent_role: str, days: int = 30) -> AgentTrajectory:
        """Analyze whether an agent role is improving, declining, or stable.

        Splits the requested window in half and compares success rates
        between the two halves.  Also computes average quality and cost.

        Args:
            agent_role: The agent role to analyze (e.g. ``"builder"``).
            days: Look-back window in days. Defaults to 30.

        Returns:
            An ``AgentTrajectory`` with trend data.
        """
        now = datetime.now(UTC)
        midpoint = now - timedelta(days=days // 2)
        cutoff = now - timedelta(days=days)

        cutoff_iso = cutoff.isoformat()
        midpoint_iso = midpoint.isoformat()

        # Fetch all records in the window
        query = (
            "SELECT * FROM outcome_records "
            "WHERE agent_role = ? AND timestamp >= ? "
            "ORDER BY timestamp"
        )
        with self._tracker._lock:
            rows = self._tracker._backend.fetchall(query, (agent_role, cutoff_iso))

        records = [self._tracker._row_to_record(r) for r in rows]

        if not records:
            return AgentTrajectory(
                agent_role=agent_role,
                period_days=days,
                current_success_rate=0.0,
                previous_success_rate=0.0,
                trend="stable",
                avg_quality_score=0.0,
                avg_cost_per_call=0.0,
                total_executions=0,
            )

        earlier = [r for r in records if r.timestamp < midpoint_iso]
        later = [r for r in records if r.timestamp >= midpoint_iso]

        prev_rate = sum(1 for r in earlier if r.success) / len(earlier) if earlier else 0.0
        curr_rate = sum(1 for r in later if r.success) / len(later) if later else 0.0

        diff = curr_rate - prev_rate
        if diff > _TREND_CHANGE_THRESHOLD:
            trend = "improving"
        elif diff < -_TREND_CHANGE_THRESHOLD:
            trend = "declining"
        else:
            trend = "stable"

        avg_quality = statistics.mean(r.quality_score for r in records)
        avg_cost = statistics.mean(r.cost_usd for r in records)

        return AgentTrajectory(
            agent_role=agent_role,
            period_days=days,
            current_success_rate=curr_rate,
            previous_success_rate=prev_rate,
            trend=trend,
            avg_quality_score=avg_quality,
            avg_cost_per_call=avg_cost,
            total_executions=len(records),
        )

    # ------------------------------------------------------------------
    # Workflow improvement suggestions
    # ------------------------------------------------------------------

    def suggest_workflow_improvements(self, workflow_id: str) -> list[Suggestion]:
        """Generate actionable improvement suggestions for a workflow.

        Analyzes historical outcomes for the given workflow and applies
        heuristic rules to identify provider upgrades, cost reductions,
        step removals, and reordering opportunities.

        Args:
            workflow_id: The workflow identifier to analyze.

        Returns:
            A list of ``Suggestion`` objects, ordered by estimated impact.
        """
        outcomes = self._tracker.get_workflow_outcomes(workflow_id)
        if not outcomes:
            return []

        suggestions: list[Suggestion] = []
        total_cost = sum(o.cost_usd for o in outcomes)

        # Group by step position / agent_role to find per-step stats
        # across potentially multiple runs of the same workflow
        step_stats: dict[str, list[OutcomeRecord]] = {}
        for o in outcomes:
            step_stats.setdefault(o.agent_role, []).append(o)

        for role, records in step_stats.items():
            role_total = len(records)
            role_successes = sum(1 for r in records if r.success)
            role_rate = role_successes / role_total if role_total else 0.0
            role_cost = sum(r.cost_usd for r in records)
            cost_fraction = role_cost / total_cost if total_cost > 0 else 0.0

            # Rule: step with <70% success rate -> suggest provider upgrade
            if role_rate < _SUCCESS_RATE_CONCERN_THRESHOLD and role_total >= 3:
                model_used = records[-1].model
                suggestions.append(
                    Suggestion(
                        step_id=records[-1].step_id,
                        category="provider_upgrade",
                        description=(
                            f"{role.capitalize()} step has a {role_rate:.0%} success rate "
                            f"with {model_used} — consider upgrading to a more capable model."
                        ),
                        estimated_impact="high",
                        confidence=min(0.5 + (role_total * 0.05), 0.95),
                    )
                )

            # Rule: step that always succeeds and adds >30% of cost -> downgrade
            if role_rate == 1.0 and cost_fraction > _HIGH_COST_FRACTION and role_total >= 3:
                suggestions.append(
                    Suggestion(
                        step_id=records[-1].step_id,
                        category="cost_reduction",
                        description=(
                            f"{role.capitalize()} step always succeeds but accounts for "
                            f"{cost_fraction:.0%} of workflow cost — consider a cheaper model."
                        ),
                        estimated_impact="medium",
                        confidence=min(0.4 + (role_total * 0.05), 0.90),
                    )
                )

            # Rule: check if a cheaper model has comparable success
            models_used: dict[str, list[OutcomeRecord]] = {}
            for r in records:
                models_used.setdefault(r.model, []).append(r)

            if len(models_used) >= 2:
                model_perf: list[tuple[str, float, float]] = []
                for m, m_records in models_used.items():
                    m_rate = sum(1 for r in m_records if r.success) / len(m_records)
                    m_cost = statistics.mean(r.cost_usd for r in m_records)
                    model_perf.append((m, m_rate, m_cost))

                model_perf.sort(key=lambda x: x[2])  # sort by cost ascending
                cheapest_model, cheapest_rate, cheapest_cost = model_perf[0]
                most_expensive_model, expensive_rate, expensive_cost = model_perf[-1]

                if cheapest_rate >= expensive_rate - 0.05 and cheapest_cost < expensive_cost * 0.7:
                    suggestions.append(
                        Suggestion(
                            step_id=records[-1].step_id,
                            category="cost_reduction",
                            description=(
                                f"{role.capitalize()} step: {cheapest_model} achieves "
                                f"similar success ({cheapest_rate:.0%}) as "
                                f"{most_expensive_model} ({expensive_rate:.0%}) "
                                f"at {cheapest_cost / expensive_cost:.0%} of the cost."
                            ),
                            estimated_impact="medium",
                            confidence=0.75,
                        )
                    )

        # Rule: sequential steps that could be parallelized
        # Heuristic: two adjacent steps with no shared agent_role that both
        # succeed independently are candidates.
        if len(outcomes) >= 2:
            for i in range(len(outcomes) - 1):
                a, b = outcomes[i], outcomes[i + 1]
                if (
                    a.agent_role != b.agent_role
                    and a.success
                    and b.success
                    and a.agent_role in step_stats
                    and b.agent_role in step_stats
                ):
                    a_rate = sum(1 for r in step_stats[a.agent_role] if r.success) / len(
                        step_stats[a.agent_role]
                    )
                    b_rate = sum(1 for r in step_stats[b.agent_role] if r.success) / len(
                        step_stats[b.agent_role]
                    )
                    if a_rate >= 0.9 and b_rate >= 0.9:
                        suggestions.append(
                            Suggestion(
                                step_id=None,
                                category="reorder",
                                description=(
                                    f"Steps '{a.agent_role}' and '{b.agent_role}' both have "
                                    f"high success rates and run sequentially — "
                                    f"consider parallelizing them."
                                ),
                                estimated_impact="low",
                                confidence=0.5,
                            )
                        )
                        break  # only suggest once

        # Sort by impact priority
        impact_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: (impact_order.get(s.estimated_impact, 3), -s.confidence))

        return suggestions

    # ------------------------------------------------------------------
    # Notability detection (private)
    # ------------------------------------------------------------------

    def _evaluate_notability(
        self,
        step_id: str,
        workflow_id: str,
        agent_role: str,
        provider: str,
        model: str,
        success: bool,
        quality_score: float,
        cost_usd: float,
        latency_ms: float,
    ) -> tuple[bool, str | None]:
        """Determine whether a step result is notable and generate a learning.

        Args:
            step_id: Step identifier.
            workflow_id: Workflow identifier.
            agent_role: Agent role for the step.
            provider: Provider used.
            model: Model used.
            success: Whether the step succeeded.
            quality_score: Quality score (0-1).
            cost_usd: Cost in USD.
            latency_ms: Latency in milliseconds.

        Returns:
            A tuple of ``(is_notable, learning_string_or_none)``.
        """
        combo = (agent_role, provider, model)

        # First execution of this combo -> baseline recording
        if combo not in self._seen_combos:
            self._seen_combos.add(combo)
            status = "succeeded" if success else "failed"
            return (
                True,
                f"First execution of {agent_role} with {provider}/{model}: "
                f"{status} (quality={quality_score:.2f}, cost=${cost_usd:.4f}, "
                f"latency={latency_ms:.0f}ms).",
            )

        # Failure after retries
        if not success:
            return (
                True,
                f"{agent_role} failed on step {step_id} using {provider}/{model}. "
                f"Quality={quality_score:.2f}, latency={latency_ms:.0f}ms.",
            )

        # Get baseline stats for comparison
        try:
            stats = self._tracker.get_provider_stats(provider, model, days=30)
        except Exception:
            logger.debug("Could not fetch provider stats for notability check.")
            return (False, None)

        if stats.total_calls < 2:
            return (False, None)

        # Slow execution: >2x average latency
        if (
            stats.avg_latency_ms > 0
            and latency_ms > stats.avg_latency_ms * _LATENCY_MULTIPLIER_THRESHOLD
        ):
            return (
                True,
                f"{agent_role} succeeded but took {latency_ms:.0f}ms "
                f"({latency_ms / stats.avg_latency_ms:.1f}x average) "
                f"with {provider}/{model}.",
            )

        # Cost anomaly: >3x average cost
        if stats.avg_cost_usd > 0 and cost_usd > stats.avg_cost_usd * _COST_MULTIPLIER_THRESHOLD:
            return (
                True,
                f"{agent_role} cost ${cost_usd:.4f} "
                f"({cost_usd / stats.avg_cost_usd:.1f}x average) "
                f"with {provider}/{model}.",
            )

        # Low quality despite success
        if quality_score < _LOW_QUALITY_THRESHOLD:
            return (
                True,
                f"{agent_role} succeeded but with low quality ({quality_score:.2f}) "
                f"using {provider}/{model}.",
            )

        return (False, None)

    # ------------------------------------------------------------------
    # Learning helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _learning_importance(success: bool, quality_score: float) -> float:
        """Compute importance score for a learning based on the outcome.

        Args:
            success: Whether the step succeeded.
            quality_score: Quality score (0-1).

        Returns:
            Importance value between 0.0 and 1.0.
        """
        if not success:
            return 0.8
        if quality_score < _LOW_QUALITY_THRESHOLD:
            return 0.7
        return 0.5

    @staticmethod
    def _learning_tags(
        success: bool,
        quality_score: float,
        cost_usd: float,
        latency_ms: float,
    ) -> list[str]:
        """Generate descriptive tags for a learning entry.

        Args:
            success: Whether the step succeeded.
            quality_score: Quality score (0-1).
            cost_usd: Step cost in USD.
            latency_ms: Step latency in milliseconds.

        Returns:
            List of string tags describing the event characteristics.
        """
        tags: list[str] = ["feedback"]
        if not success:
            tags.append("failure")
        else:
            tags.append("success")
        if quality_score < _LOW_QUALITY_THRESHOLD:
            tags.append("low_quality")
        if cost_usd > 0.10:
            tags.append("high_cost")
        if latency_ms > 10_000:
            tags.append("high_latency")
        return tags
