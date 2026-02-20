"""Tests for feedback engine module.

Comprehensively tests FeedbackEngine with mocked OutcomeTracker, memory, and router.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from animus_forge.intelligence.feedback_engine import (
    _COST_MULTIPLIER_THRESHOLD,
    _HIGH_COST_FRACTION,
    _LATENCY_MULTIPLIER_THRESHOLD,
    _LOW_QUALITY_THRESHOLD,
    _SUCCESS_RATE_CONCERN_THRESHOLD,
    _TREND_CHANGE_THRESHOLD,
    AgentTrajectory,
    FeedbackEngine,
    FeedbackResult,
    Suggestion,
    WorkflowFeedback,
)
from animus_forge.intelligence.outcome_tracker import (
    OutcomeRecord,
    OutcomeTracker,
    ProviderStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outcome(**overrides) -> OutcomeRecord:
    """Create an OutcomeRecord with sensible defaults."""
    defaults = dict(
        step_id="s1",
        workflow_id="w1",
        agent_role="builder",
        provider="openai",
        model="gpt-4o",
        success=True,
        quality_score=0.9,
        cost_usd=0.02,
        tokens_used=1000,
        latency_ms=800,
        metadata={},
    )
    defaults.update(overrides)
    return OutcomeRecord(**defaults)


def _make_engine(
    tracker=None,
    memory=None,
    router=None,
    preload_combos=False,
):
    """Create a FeedbackEngine with mocked dependencies."""
    if tracker is None:
        tracker = MagicMock()
        tracker._lock = MagicMock()
        tracker._lock.__enter__ = MagicMock()
        tracker._lock.__exit__ = MagicMock(return_value=False)
        tracker._backend = MagicMock()
        tracker._backend.fetchall.return_value = []
        tracker._row_to_record = OutcomeTracker._row_to_record
    if memory is None:
        memory = MagicMock()
    if router is None:
        router = MagicMock()

    engine = FeedbackEngine(tracker, memory, router)
    if not preload_combos:
        # Clear combos so tests control first-seen behavior
        engine._seen_combos.clear()
    return engine


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Tests for feedback engine data classes."""

    def test_feedback_result(self):
        result = FeedbackResult(
            step_id="s1",
            outcome_recorded=True,
            learning_generated="Test learning",
            provider_updated=True,
            notable=True,
        )
        assert result.step_id == "s1"
        assert result.outcome_recorded is True
        assert result.learning_generated == "Test learning"
        assert result.notable is True

    def test_workflow_feedback(self):
        wf = WorkflowFeedback(
            workflow_id="w1",
            total_steps=5,
            successful_steps=4,
            total_cost_usd=0.10,
            insights=["Step 1 failed"],
            learnings_stored=1,
        )
        assert wf.total_steps == 5
        assert wf.successful_steps == 4

    def test_agent_trajectory(self):
        at = AgentTrajectory(
            agent_role="builder",
            period_days=30,
            current_success_rate=0.95,
            previous_success_rate=0.85,
            trend="improving",
            avg_quality_score=0.9,
            avg_cost_per_call=0.02,
            total_executions=100,
        )
        assert at.trend == "improving"

    def test_suggestion(self):
        s = Suggestion(
            step_id="s1",
            category="provider_upgrade",
            description="Upgrade model",
            estimated_impact="high",
            confidence=0.85,
        )
        assert s.category == "provider_upgrade"
        assert s.confidence == 0.85

    def test_suggestion_no_step(self):
        s = Suggestion(
            step_id=None,
            category="reorder",
            description="Parallelize steps",
            estimated_impact="low",
            confidence=0.5,
        )
        assert s.step_id is None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests that constants are set to expected values."""

    def test_thresholds_are_positive(self):
        assert _LATENCY_MULTIPLIER_THRESHOLD > 0
        assert _COST_MULTIPLIER_THRESHOLD > 0
        assert _LOW_QUALITY_THRESHOLD > 0
        assert _TREND_CHANGE_THRESHOLD > 0
        assert _SUCCESS_RATE_CONCERN_THRESHOLD > 0
        assert _HIGH_COST_FRACTION > 0


# ---------------------------------------------------------------------------
# FeedbackEngine init
# ---------------------------------------------------------------------------


class TestFeedbackEngineInit:
    """Tests for FeedbackEngine initialization."""

    def test_init_loads_seen_combos(self):
        tracker = MagicMock()
        tracker._lock = MagicMock()
        tracker._lock.__enter__ = MagicMock()
        tracker._lock.__exit__ = MagicMock(return_value=False)
        tracker._backend = MagicMock()
        tracker._backend.fetchall.return_value = [
            {"agent_role": "builder", "provider": "openai", "model": "gpt-4o"},
            {
                "agent_role": "reviewer",
                "provider": "anthropic",
                "model": "claude-3-sonnet",
            },
        ]

        engine = FeedbackEngine(tracker, MagicMock(), MagicMock())
        assert ("builder", "openai", "gpt-4o") in engine._seen_combos
        assert ("reviewer", "anthropic", "claude-3-sonnet") in engine._seen_combos

    def test_init_handles_load_failure(self):
        tracker = MagicMock()
        tracker._lock = MagicMock()
        tracker._lock.__enter__ = MagicMock()
        tracker._lock.__exit__ = MagicMock(return_value=False)
        tracker._backend = MagicMock()
        tracker._backend.fetchall.side_effect = Exception("DB error")

        # Should not raise
        engine = FeedbackEngine(tracker, MagicMock(), MagicMock())
        assert len(engine._seen_combos) == 0


# ---------------------------------------------------------------------------
# process_step_result
# ---------------------------------------------------------------------------


class TestProcessStepResult:
    """Tests for the process_step_result method."""

    def test_first_execution_is_notable(self):
        engine = _make_engine()
        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True, "quality_score": 0.9, "latency_ms": 1000},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.notable is True
        assert result.learning_generated is not None
        assert "First execution" in result.learning_generated

    def test_second_execution_not_notable(self):
        engine = _make_engine()
        # First: marks as seen
        engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True, "quality_score": 0.9, "latency_ms": 1000},
            cost_usd=0.02,
            tokens_used=500,
        )

        # Mock provider stats for the second call
        mock_stats = MagicMock()
        mock_stats.total_calls = 10
        mock_stats.avg_latency_ms = 1000
        mock_stats.avg_cost_usd = 0.02
        engine._tracker.get_provider_stats.return_value = mock_stats

        # Second: normal execution, not notable
        result = engine.process_step_result(
            step_id="s2",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True, "quality_score": 0.9, "latency_ms": 1000},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.notable is False

    def test_failure_is_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": False},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.notable is True
        assert "failed" in result.learning_generated

    def test_outcome_recorded(self):
        engine = _make_engine()
        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.outcome_recorded is True
        engine._tracker.record.assert_called_once()

    def test_outcome_record_failure(self):
        engine = _make_engine()
        engine._tracker.record.side_effect = Exception("DB error")

        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.outcome_recorded is False

    def test_provider_updated(self):
        engine = _make_engine()
        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.provider_updated is True
        engine._router.update_after_execution.assert_called_once()

    def test_provider_update_failure(self):
        engine = _make_engine()
        engine._router.update_after_execution.side_effect = Exception("Router error")

        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.provider_updated is False

    def test_learning_stored_on_notable(self):
        engine = _make_engine()
        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.notable is True
        engine._memory.record_learning.assert_called_once()

    def test_learning_store_failure(self):
        engine = _make_engine()
        engine._memory.record_learning.side_effect = Exception("Memory error")

        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        # Should not crash, learning still generated
        assert result.notable is True
        assert result.learning_generated is not None

    def test_default_quality_score_on_success(self):
        engine = _make_engine()
        # No quality_score in step_result -> defaults to 1.0 on success
        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": True},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.outcome_recorded is True

    def test_default_quality_score_on_failure(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))
        # No quality_score in step_result -> defaults to 0.0 on failure
        result = engine.process_step_result(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            step_result={"success": False},
            cost_usd=0.02,
            tokens_used=500,
        )
        assert result.notable is True


# ---------------------------------------------------------------------------
# Notability detection
# ---------------------------------------------------------------------------


class TestNotabilityDetection:
    """Tests for the _evaluate_notability method."""

    def test_first_combo_notable(self):
        engine = _make_engine()
        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.9,
            cost_usd=0.02,
            latency_ms=1000,
        )
        assert notable is True
        assert "First execution" in learning

    def test_first_combo_failure(self):
        engine = _make_engine()
        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=False,
            quality_score=0.0,
            cost_usd=0.02,
            latency_ms=1000,
        )
        assert notable is True
        assert "failed" in learning

    def test_failure_after_seen_is_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=False,
            quality_score=0.0,
            cost_usd=0.02,
            latency_ms=1000,
        )
        assert notable is True
        assert "failed" in learning

    def test_slow_execution_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        mock_stats = ProviderStats(
            success_rate=0.95,
            avg_latency_ms=1000.0,
            avg_cost_usd=0.02,
            total_calls=50,
        )
        engine._tracker.get_provider_stats.return_value = mock_stats

        # 3x average latency -> notable
        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.9,
            cost_usd=0.02,
            latency_ms=3000.0,
        )
        assert notable is True
        assert "average" in learning.lower()

    def test_cost_anomaly_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        mock_stats = ProviderStats(
            success_rate=0.95,
            avg_latency_ms=1000.0,
            avg_cost_usd=0.02,
            total_calls=50,
        )
        engine._tracker.get_provider_stats.return_value = mock_stats

        # 5x average cost -> notable
        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.9,
            cost_usd=0.10,
            latency_ms=1000.0,
        )
        assert notable is True
        assert "cost" in learning.lower() or "$" in learning

    def test_low_quality_despite_success_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        mock_stats = ProviderStats(
            success_rate=0.95,
            avg_latency_ms=1000.0,
            avg_cost_usd=0.02,
            total_calls=50,
        )
        engine._tracker.get_provider_stats.return_value = mock_stats

        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.1,  # Below _LOW_QUALITY_THRESHOLD
            cost_usd=0.02,
            latency_ms=1000.0,
        )
        assert notable is True
        assert "low quality" in learning.lower()

    def test_normal_execution_not_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        mock_stats = ProviderStats(
            success_rate=0.95,
            avg_latency_ms=1000.0,
            avg_cost_usd=0.02,
            total_calls=50,
        )
        engine._tracker.get_provider_stats.return_value = mock_stats

        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.9,
            cost_usd=0.02,
            latency_ms=1000.0,
        )
        assert notable is False
        assert learning is None

    def test_stats_exception_not_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))
        engine._tracker.get_provider_stats.side_effect = Exception("DB error")

        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.9,
            cost_usd=0.02,
            latency_ms=1000.0,
        )
        assert notable is False

    def test_few_calls_not_notable(self):
        engine = _make_engine()
        engine._seen_combos.add(("builder", "openai", "gpt-4o"))

        mock_stats = ProviderStats(
            success_rate=0.95,
            avg_latency_ms=1000.0,
            avg_cost_usd=0.02,
            total_calls=1,  # Less than 2
        )
        engine._tracker.get_provider_stats.return_value = mock_stats

        notable, learning = engine._evaluate_notability(
            step_id="s1",
            workflow_id="w1",
            agent_role="builder",
            provider="openai",
            model="gpt-4o",
            success=True,
            quality_score=0.9,
            cost_usd=0.02,
            latency_ms=1000.0,
        )
        assert notable is False


# ---------------------------------------------------------------------------
# process_workflow_result
# ---------------------------------------------------------------------------


class TestProcessWorkflowResult:
    """Tests for the process_workflow_result method."""

    def test_no_outcomes(self):
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = []

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Test Workflow",
            execution_result={"steps": []},
        )
        assert result.total_steps == 0
        assert result.successful_steps == 0
        assert len(result.insights) == 1
        assert "no step outcomes" in result.insights[0].lower()

    def test_all_steps_succeed(self):
        outcomes = [
            _make_outcome(step_id="s1", agent_role="builder", success=True, cost_usd=0.01),
            _make_outcome(step_id="s2", agent_role="reviewer", success=True, cost_usd=0.01),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        assert result.total_steps == 2
        assert result.successful_steps == 2
        assert result.total_cost_usd == 0.02

    def test_partial_failure_generates_insight(self):
        outcomes = [
            _make_outcome(step_id="s1", agent_role="builder", success=True, cost_usd=0.01),
            _make_outcome(step_id="s2", agent_role="builder", success=True, cost_usd=0.01),
            _make_outcome(step_id="s3", agent_role="builder", success=False, cost_usd=0.01),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        assert result.successful_steps == 2
        assert result.total_steps == 3
        # Should have role-level insight
        role_insights = [i for i in result.insights if "builder" in i.lower()]
        assert len(role_insights) > 0

    def test_high_cost_step_generates_insight(self):
        outcomes = [
            _make_outcome(step_id="s1", agent_role="builder", cost_usd=0.90),
            _make_outcome(step_id="s2", agent_role="reviewer", cost_usd=0.10),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        # Step s1 takes 90% of cost
        cost_insights = [i for i in result.insights if "cost" in i.lower()]
        assert len(cost_insights) > 0

    def test_consecutive_failures_detected(self):
        outcomes = [
            _make_outcome(step_id="s1", success=True),
            _make_outcome(step_id="s2", success=False),
            _make_outcome(step_id="s3", success=False),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        failure_insights = [i for i in result.insights if "consecutive" in i.lower()]
        assert len(failure_insights) > 0

    def test_learnings_stored(self):
        outcomes = [
            _make_outcome(step_id="s1", agent_role="builder", success=True, cost_usd=0.01),
            _make_outcome(step_id="s2", agent_role="builder", success=True, cost_usd=0.01),
            _make_outcome(step_id="s3", agent_role="builder", success=False, cost_usd=0.01),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        assert result.learnings_stored > 0
        engine._memory.record_learning.assert_called()

    def test_learning_store_failure_handled(self):
        outcomes = [
            _make_outcome(step_id="s1", agent_role="builder", success=True, cost_usd=0.01),
            _make_outcome(step_id="s2", agent_role="builder", success=False, cost_usd=0.01),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes
        engine._memory.record_learning.side_effect = Exception("Memory error")

        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        # Should still complete, learnings_stored will be 0
        assert result.learnings_stored == 0

    def test_memory_decay_called(self):
        outcomes = [_make_outcome(step_id="s1")]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        engine._memory.decay_memories.assert_called_once_with(half_life_days=90)

    def test_memory_decay_failure_handled(self):
        outcomes = [_make_outcome(step_id="s1")]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes
        engine._memory.decay_memories.side_effect = Exception("Decay error")

        # Should not raise
        result = engine.process_workflow_result(
            workflow_id="w1",
            workflow_name="Build",
            execution_result={"steps": []},
        )
        assert result is not None


# ---------------------------------------------------------------------------
# analyze_agent_trajectory
# ---------------------------------------------------------------------------


class TestAnalyzeAgentTrajectory:
    """Tests for the analyze_agent_trajectory method."""

    def test_no_records(self):
        engine = _make_engine()
        engine._tracker._backend.fetchall.return_value = []

        result = engine.analyze_agent_trajectory("builder", days=30)
        assert result.agent_role == "builder"
        assert result.trend == "stable"
        assert result.total_executions == 0
        assert result.avg_quality_score == 0.0

    def test_improving_trend(self):
        engine = _make_engine()
        now = datetime.now(UTC)

        # Earlier half: low success, later half: high success
        earlier_rows = [
            {
                "step_id": f"s{i}",
                "workflow_id": "w1",
                "agent_role": "builder",
                "provider": "openai",
                "model": "gpt-4o",
                "success": 0 if i % 3 == 0 else 1,
                "quality_score": 0.5,
                "cost_usd": 0.02,
                "tokens_used": 1000,
                "latency_ms": 1000,
                "metadata": "{}",
                "timestamp": (now - timedelta(days=25 - i)).isoformat(),
            }
            for i in range(5)
        ]
        later_rows = [
            {
                "step_id": f"s{i + 5}",
                "workflow_id": "w1",
                "agent_role": "builder",
                "provider": "openai",
                "model": "gpt-4o",
                "success": 1,
                "quality_score": 0.9,
                "cost_usd": 0.02,
                "tokens_used": 1000,
                "latency_ms": 1000,
                "metadata": "{}",
                "timestamp": (now - timedelta(days=5 - i)).isoformat(),
            }
            for i in range(5)
        ]
        engine._tracker._backend.fetchall.return_value = earlier_rows + later_rows

        result = engine.analyze_agent_trajectory("builder", days=30)
        assert result.total_executions == 10
        assert result.current_success_rate > result.previous_success_rate
        assert result.trend == "improving"

    def test_declining_trend(self):
        engine = _make_engine()
        now = datetime.now(UTC)

        # Earlier half: all succeed, later half: mostly fail
        earlier_rows = [
            {
                "step_id": f"s{i}",
                "workflow_id": "w1",
                "agent_role": "builder",
                "provider": "openai",
                "model": "gpt-4o",
                "success": 1,
                "quality_score": 0.9,
                "cost_usd": 0.02,
                "tokens_used": 1000,
                "latency_ms": 1000,
                "metadata": "{}",
                "timestamp": (now - timedelta(days=25 - i)).isoformat(),
            }
            for i in range(5)
        ]
        later_rows = [
            {
                "step_id": f"s{i + 5}",
                "workflow_id": "w1",
                "agent_role": "builder",
                "provider": "openai",
                "model": "gpt-4o",
                "success": 0,
                "quality_score": 0.2,
                "cost_usd": 0.02,
                "tokens_used": 1000,
                "latency_ms": 1000,
                "metadata": "{}",
                "timestamp": (now - timedelta(days=5 - i)).isoformat(),
            }
            for i in range(5)
        ]
        engine._tracker._backend.fetchall.return_value = earlier_rows + later_rows

        result = engine.analyze_agent_trajectory("builder", days=30)
        assert result.trend == "declining"

    def test_stable_trend(self):
        engine = _make_engine()
        now = datetime.now(UTC)

        # Spread records evenly across the full 30-day window so both halves
        # have the same success rate (all succeed).
        rows = [
            {
                "step_id": f"s{i}",
                "workflow_id": "w1",
                "agent_role": "builder",
                "provider": "openai",
                "model": "gpt-4o",
                "success": 1,
                "quality_score": 0.9,
                "cost_usd": 0.02,
                "tokens_used": 1000,
                "latency_ms": 1000,
                "metadata": "{}",
                "timestamp": (now - timedelta(days=28 - i * 3)).isoformat(),
            }
            for i in range(10)
        ]
        engine._tracker._backend.fetchall.return_value = rows

        result = engine.analyze_agent_trajectory("builder", days=30)
        assert result.trend == "stable"


# ---------------------------------------------------------------------------
# suggest_workflow_improvements
# ---------------------------------------------------------------------------


class TestSuggestWorkflowImprovements:
    """Tests for the suggest_workflow_improvements method."""

    def test_no_outcomes(self):
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = []
        result = engine.suggest_workflow_improvements("w1")
        assert result == []

    def test_low_success_rate_suggests_upgrade(self):
        outcomes = [
            _make_outcome(step_id=f"s{i}", agent_role="builder", success=i % 2 == 0)
            for i in range(6)
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        suggestions = engine.suggest_workflow_improvements("w1")
        upgrade_suggestions = [s for s in suggestions if s.category == "provider_upgrade"]
        assert len(upgrade_suggestions) > 0

    def test_high_cost_step_suggests_downgrade(self):
        outcomes = [
            _make_outcome(step_id="s1", agent_role="builder", success=True, cost_usd=0.80),
            _make_outcome(step_id="s2", agent_role="builder", success=True, cost_usd=0.80),
            _make_outcome(step_id="s3", agent_role="builder", success=True, cost_usd=0.80),
            _make_outcome(step_id="s4", agent_role="reviewer", success=True, cost_usd=0.05),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        suggestions = engine.suggest_workflow_improvements("w1")
        cost_suggestions = [s for s in suggestions if s.category == "cost_reduction"]
        assert len(cost_suggestions) > 0

    def test_cheaper_model_with_same_quality_suggested(self):
        outcomes = [
            # Expensive model, good success
            _make_outcome(
                step_id="s1",
                agent_role="builder",
                model="gpt-4",
                success=True,
                cost_usd=0.50,
            ),
            _make_outcome(
                step_id="s2",
                agent_role="builder",
                model="gpt-4",
                success=True,
                cost_usd=0.50,
            ),
            _make_outcome(
                step_id="s3",
                agent_role="builder",
                model="gpt-4",
                success=True,
                cost_usd=0.50,
            ),
            # Cheap model, same success
            _make_outcome(
                step_id="s4",
                agent_role="builder",
                model="gpt-4o-mini",
                success=True,
                cost_usd=0.01,
            ),
            _make_outcome(
                step_id="s5",
                agent_role="builder",
                model="gpt-4o-mini",
                success=True,
                cost_usd=0.01,
            ),
            _make_outcome(
                step_id="s6",
                agent_role="builder",
                model="gpt-4o-mini",
                success=True,
                cost_usd=0.01,
            ),
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        suggestions = engine.suggest_workflow_improvements("w1")
        cost_suggestions = [s for s in suggestions if s.category == "cost_reduction"]
        assert len(cost_suggestions) > 0

    def test_parallelization_suggestion(self):
        # Need 3+ records per role for high success rate
        engine = _make_engine()
        # Mock the get_workflow_outcomes to return enough records
        many_outcomes = [
            _make_outcome(step_id=f"sb{i}", agent_role="builder", success=True) for i in range(5)
        ] + [_make_outcome(step_id=f"sr{i}", agent_role="reviewer", success=True) for i in range(5)]
        engine._tracker.get_workflow_outcomes.return_value = many_outcomes

        suggestions = engine.suggest_workflow_improvements("w1")
        reorder_suggestions = [s for s in suggestions if s.category == "reorder"]
        if reorder_suggestions:
            assert "paralleliz" in reorder_suggestions[0].description.lower()

    def test_suggestions_sorted_by_impact(self):
        outcomes = [
            # Low success rate (builder) -> high impact
            _make_outcome(step_id=f"s{i}", agent_role="builder", success=i % 3 == 0)
            for i in range(6)
        ] + [
            # Always succeeds but high cost (reviewer) -> medium impact
            _make_outcome(
                step_id=f"r{i}",
                agent_role="reviewer",
                success=True,
                cost_usd=0.90,
            )
            for i in range(6)
        ]
        engine = _make_engine()
        engine._tracker.get_workflow_outcomes.return_value = outcomes

        suggestions = engine.suggest_workflow_improvements("w1")
        if len(suggestions) >= 2:
            impact_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(suggestions) - 1):
                assert impact_order.get(suggestions[i].estimated_impact, 3) <= impact_order.get(
                    suggestions[i + 1].estimated_impact, 3
                )


# ---------------------------------------------------------------------------
# Learning helpers
# ---------------------------------------------------------------------------


class TestLearningHelpers:
    """Tests for static learning helper methods."""

    def test_learning_importance_failure(self):
        assert FeedbackEngine._learning_importance(False, 0.5) == 0.8

    def test_learning_importance_low_quality(self):
        assert FeedbackEngine._learning_importance(True, 0.1) == 0.7

    def test_learning_importance_normal_success(self):
        assert FeedbackEngine._learning_importance(True, 0.9) == 0.5

    def test_learning_tags_failure(self):
        tags = FeedbackEngine._learning_tags(False, 0.5, 0.01, 500)
        assert "feedback" in tags
        assert "failure" in tags
        assert "success" not in tags

    def test_learning_tags_success(self):
        tags = FeedbackEngine._learning_tags(True, 0.9, 0.01, 500)
        assert "success" in tags
        assert "failure" not in tags

    def test_learning_tags_low_quality(self):
        tags = FeedbackEngine._learning_tags(True, 0.1, 0.01, 500)
        assert "low_quality" in tags

    def test_learning_tags_high_cost(self):
        tags = FeedbackEngine._learning_tags(True, 0.9, 0.50, 500)
        assert "high_cost" in tags

    def test_learning_tags_high_latency(self):
        tags = FeedbackEngine._learning_tags(True, 0.9, 0.01, 15000)
        assert "high_latency" in tags

    def test_learning_tags_all_flags(self):
        tags = FeedbackEngine._learning_tags(False, 0.1, 0.50, 15000)
        assert "failure" in tags
        assert "low_quality" in tags
        assert "high_cost" in tags
        assert "high_latency" in tags

    def test_learning_tags_no_extra_flags(self):
        tags = FeedbackEngine._learning_tags(True, 0.9, 0.01, 500)
        assert tags == ["feedback", "success"]
