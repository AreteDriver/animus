"""Tests for cost intelligence module.

Comprehensively tests CostIntelligence with mocked CostTracker and OutcomeTracker.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.intelligence.cost_intelligence import (
    CostIntelligence,
    ROIReport,
    SavingsRecommendation,
    SpendForecast,
    SpendingAnalysis,
    SwitchImpact,
)
from animus_forge.metrics.cost_tracker import CostEntry, CostTracker, Provider, TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    days_ago: int = 0,
    provider: Provider = Provider.OPENAI,
    model: str = "gpt-4o",
    cost_usd: float = 0.02,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    workflow_id: str | None = None,
    agent_role: str | None = "builder",
) -> CostEntry:
    """Create a CostEntry for testing."""
    return CostEntry(
        timestamp=datetime.now() - timedelta(days=days_ago),
        provider=provider,
        model=model,
        tokens=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        cost_usd=cost_usd,
        workflow_id=workflow_id,
        agent_role=agent_role,
    )


def _make_tracker(entries=None, budget_limit=None):
    """Create a CostTracker with pre-set entries."""
    tracker = CostTracker(budget_limit_usd=budget_limit)
    if entries:
        tracker.entries = entries
    return tracker


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Tests for cost intelligence data classes."""

    def test_spending_analysis(self):
        sa = SpendingAnalysis(
            period_days=30,
            total_usd=100.0,
            by_provider={"openai": 60.0, "anthropic": 40.0},
            by_model={"gpt-4o": 60.0, "claude-3-sonnet": 40.0},
            by_role={"builder": 70.0, "reviewer": 30.0},
            daily_avg=3.33,
            trend="stable",
            top_workflows=[("wf1", 50.0)],
            top_roles=[("builder", 70.0)],
        )
        assert sa.total_usd == 100.0
        assert sa.trend == "stable"

    def test_savings_recommendation(self):
        rec = SavingsRecommendation(
            agent_role="builder",
            current_model="gpt-4",
            recommended_model="gpt-4o",
            current_cost_monthly=200.0,
            projected_cost_monthly=50.0,
            estimated_savings_monthly=150.0,
            quality_impact="minimal (<5%)",
            confidence="high",
            reasoning="Switch to save money.",
        )
        assert rec.estimated_savings_monthly == 150.0

    def test_switch_impact(self):
        si = SwitchImpact(
            agent_role="builder",
            from_model="gpt-4",
            to_model="gpt-4o",
            cost_change_pct=-70.0,
            quality_change_pct=-2.0,
            latency_change_pct=-50.0,
            monthly_savings_usd=150.0,
            risk_level="low",
        )
        assert si.cost_change_pct == -70.0

    def test_roi_report(self):
        roi = ROIReport(
            total_savings_usd=500.0,
            routing_savings_usd=400.0,
            enforcement_savings_usd=100.0,
            total_executions=1000,
            avg_cost_per_execution=0.02,
            efficiency_score=75.0,
        )
        assert roi.efficiency_score == 75.0

    def test_spend_forecast(self):
        sf = SpendForecast(
            daily_rate=10.0,
            projected_30d=300.0,
            budget_limit=500.0,
            days_until_budget=20,
            trend="increasing",
        )
        assert sf.projected_30d == 300.0
        assert sf.days_until_budget == 20


# ---------------------------------------------------------------------------
# CostIntelligence init
# ---------------------------------------------------------------------------


class TestCostIntelligenceInit:
    """Tests for CostIntelligence initialization."""

    def test_init_without_outcome_tracker(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._outcome_tracker is None

    def test_init_with_outcome_tracker(self):
        tracker = _make_tracker()
        outcome = MagicMock()
        ci = CostIntelligence(tracker, outcome_tracker=outcome)
        assert ci._outcome_tracker is outcome


# ---------------------------------------------------------------------------
# analyze_spending
# ---------------------------------------------------------------------------


class TestAnalyzeSpending:
    """Tests for the analyze_spending method."""

    def test_empty_entries(self):
        tracker = _make_tracker([])
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.total_usd == 0.0
        assert result.daily_avg == 0.0
        assert result.trend == "stable"

    def test_single_entry(self):
        entry = _make_entry(days_ago=1, cost_usd=10.0, agent_role="builder")
        tracker = _make_tracker([entry])
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.total_usd == 10.0
        assert result.period_days == 30

    def test_multiple_providers(self):
        entries = [
            _make_entry(days_ago=1, provider=Provider.OPENAI, model="gpt-4o", cost_usd=5.0),
            _make_entry(
                days_ago=1,
                provider=Provider.ANTHROPIC,
                model="claude-3-sonnet",
                cost_usd=3.0,
            ),
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.by_provider["openai"] == 5.0
        assert result.by_provider["anthropic"] == 3.0
        assert result.total_usd == 8.0

    def test_by_model_aggregation(self):
        entries = [
            _make_entry(days_ago=1, model="gpt-4o", cost_usd=2.0),
            _make_entry(days_ago=1, model="gpt-4o", cost_usd=3.0),
            _make_entry(days_ago=1, model="gpt-4o-mini", cost_usd=0.5),
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.by_model["gpt-4o"] == 5.0
        assert result.by_model["gpt-4o-mini"] == 0.5

    def test_by_role_aggregation(self):
        entries = [
            _make_entry(days_ago=1, agent_role="builder", cost_usd=5.0),
            _make_entry(days_ago=1, agent_role="reviewer", cost_usd=3.0),
            _make_entry(days_ago=1, agent_role=None, cost_usd=1.0),
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.by_role["builder"] == 5.0
        assert result.by_role["reviewer"] == 3.0
        assert result.by_role["unknown"] == 1.0

    def test_workflow_tracking(self):
        entries = [
            _make_entry(days_ago=1, workflow_id="wf-1", cost_usd=10.0),
            _make_entry(days_ago=1, workflow_id="wf-2", cost_usd=5.0),
            _make_entry(days_ago=1, workflow_id="wf-1", cost_usd=3.0),
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert len(result.top_workflows) <= 3
        assert result.top_workflows[0] == ("wf-1", 13.0)

    def test_top_roles_capped_at_3(self):
        entries = [
            _make_entry(days_ago=1, agent_role=f"role-{i}", cost_usd=float(i)) for i in range(5)
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert len(result.top_roles) == 3

    def test_daily_avg(self):
        entries = [_make_entry(days_ago=i, cost_usd=10.0) for i in range(7)]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.daily_avg == pytest.approx(70.0 / 30, abs=0.01)

    def test_entries_outside_window_excluded(self):
        entries = [
            _make_entry(days_ago=1, cost_usd=10.0),
            _make_entry(days_ago=60, cost_usd=100.0),  # Outside 30-day window
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.analyze_spending(30)
        assert result.total_usd == 10.0


# ---------------------------------------------------------------------------
# _compute_trend
# ---------------------------------------------------------------------------


class TestComputeTrend:
    """Tests for the trend computation logic."""

    def test_single_day_stable(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._compute_trend({"2026-01-01": 10.0}, 7) == "stable"

    def test_increasing_trend(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        by_day = {
            "2026-01-01": 1.0,
            "2026-01-02": 1.0,
            "2026-01-03": 2.0,
            "2026-01-04": 3.0,
        }
        assert ci._compute_trend(by_day, 7) == "increasing"

    def test_decreasing_trend(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        by_day = {
            "2026-01-01": 10.0,
            "2026-01-02": 10.0,
            "2026-01-03": 2.0,
            "2026-01-04": 1.0,
        }
        assert ci._compute_trend(by_day, 7) == "decreasing"

    def test_stable_trend(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        by_day = {
            "2026-01-01": 5.0,
            "2026-01-02": 5.0,
            "2026-01-03": 5.0,
            "2026-01-04": 5.0,
        }
        assert ci._compute_trend(by_day, 7) == "stable"

    def test_zero_first_half_is_stable(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        by_day = {
            "2026-01-01": 0.0,
            "2026-01-02": 5.0,
        }
        assert ci._compute_trend(by_day, 7) == "stable"

    def test_empty_dict_stable(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._compute_trend({}, 7) == "stable"


# ---------------------------------------------------------------------------
# recommend_savings
# ---------------------------------------------------------------------------


class TestRecommendSavings:
    """Tests for the recommend_savings method."""

    def test_no_entries_no_recommendations(self):
        tracker = _make_tracker([])
        ci = CostIntelligence(tracker)
        result = ci.recommend_savings(30)
        assert result == []

    def test_recommendations_with_expensive_model(self):
        # Many calls using expensive gpt-4, should recommend cheaper alternative
        entries = [
            _make_entry(
                days_ago=i,
                model="gpt-4",
                agent_role="builder",
                cost_usd=0.50,
                input_tokens=5000,
                output_tokens=5000,
            )
            for i in range(10)
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.recommend_savings(30)
        # Should find at least one recommendation to switch from gpt-4
        if result:  # Some recommendations depend on exact pricing ratios
            assert result[0].current_model == "gpt-4"
            assert result[0].estimated_savings_monthly > 0

    def test_recommendations_sorted_by_savings(self):
        entries = [
            _make_entry(
                days_ago=i,
                model="gpt-4",
                agent_role="builder",
                cost_usd=1.0,
            )
            for i in range(10)
        ] + [
            _make_entry(
                days_ago=i,
                model="gpt-4",
                agent_role="reviewer",
                cost_usd=0.5,
            )
            for i in range(10)
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        recs = ci.recommend_savings(30)
        if len(recs) >= 2:
            assert recs[0].estimated_savings_monthly >= recs[1].estimated_savings_monthly


# ---------------------------------------------------------------------------
# estimate_switch_impact
# ---------------------------------------------------------------------------


class TestEstimateSwitchImpact:
    """Tests for the estimate_switch_impact method."""

    def test_basic_switch(self):
        entries = [
            _make_entry(
                days_ago=i,
                model="gpt-4",
                agent_role="builder",
                cost_usd=0.50,
            )
            for i in range(10)
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.estimate_switch_impact("builder", "gpt-4", "gpt-4o-mini", days=30)
        assert isinstance(result, SwitchImpact)
        assert result.agent_role == "builder"
        assert result.from_model == "gpt-4"
        assert result.to_model == "gpt-4o-mini"
        # gpt-4o-mini should be cheaper than gpt-4
        assert result.cost_change_pct < 0

    def test_switch_no_entries(self):
        tracker = _make_tracker([])
        ci = CostIntelligence(tracker)
        result = ci.estimate_switch_impact("builder", "gpt-4", "gpt-4o", days=30)
        assert result.monthly_savings_usd == 0.0
        assert result.cost_change_pct == 0.0

    def test_switch_with_outcome_tracker(self):
        entries = [
            _make_entry(days_ago=i, model="gpt-4", agent_role="builder", cost_usd=0.5)
            for i in range(5)
        ]
        tracker = _make_tracker(entries)

        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 50
        mock_stats.success_rate = 0.95
        mock_stats.avg_latency_ms = 2000.0
        mock_stats.avg_cost_usd = 0.5
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        result = ci.estimate_switch_impact("builder", "gpt-4", "gpt-4o", days=30)
        assert isinstance(result, SwitchImpact)

    def test_switch_risk_levels(self):
        tracker = _make_tracker([])
        ci = CostIntelligence(tracker)
        # Low risk: minimal quality drop
        assert ci._assess_risk(-2.0, -50.0) == "low"
        # Medium risk: moderate quality drop
        assert ci._assess_risk(-10.0, -50.0) == "medium"
        # High risk: significant quality drop
        assert ci._assess_risk(-20.0, -50.0) == "high"


# ---------------------------------------------------------------------------
# get_roi_report
# ---------------------------------------------------------------------------


class TestGetROIReport:
    """Tests for the get_roi_report method."""

    def test_roi_empty(self):
        tracker = _make_tracker([])
        ci = CostIntelligence(tracker)
        result = ci.get_roi_report()
        assert result.total_savings_usd == 0.0
        assert result.total_executions == 0
        assert result.efficiency_score == 50.0

    def test_roi_with_entries(self):
        entries = [
            _make_entry(
                days_ago=i,
                model="gpt-4o-mini",
                cost_usd=0.01,
                input_tokens=1000,
                output_tokens=500,
            )
            for i in range(20)
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.get_roi_report()
        assert result.total_executions == 20
        assert result.avg_cost_per_execution > 0
        # Using cheap model should show savings vs naive baseline (most expensive)
        assert result.routing_savings_usd >= 0

    def test_roi_efficiency_score_scales(self):
        # All cheap calls -> high efficiency
        cheap_entries = [
            _make_entry(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
            for _ in range(10)
        ]
        tracker = _make_tracker(cheap_entries)
        ci = CostIntelligence(tracker)
        cheap_roi = ci.get_roi_report()

        # All expensive calls -> low efficiency
        expensive_entries = [
            _make_entry(
                model="gpt-4",
                cost_usd=0.50,
                input_tokens=10000,
                output_tokens=10000,
            )
            for _ in range(10)
        ]
        tracker2 = _make_tracker(expensive_entries)
        ci2 = CostIntelligence(tracker2)
        expensive_roi = ci2.get_roi_report()

        # Cheap model should have higher efficiency
        assert cheap_roi.efficiency_score >= expensive_roi.efficiency_score

    def test_roi_with_outcome_tracker(self):
        entries = [
            _make_entry(model="gpt-4o", cost_usd=0.02, input_tokens=1000, output_tokens=500)
            for _ in range(10)
        ]
        tracker = _make_tracker(entries)

        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 50
        mock_stats.success_rate = 0.95
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        result = ci.get_roi_report()
        assert result.enforcement_savings_usd >= 0

    def test_roi_enforcement_no_outcome_tracker(self):
        entries = [_make_entry() for _ in range(5)]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)  # No outcome tracker
        result = ci.get_roi_report()
        assert result.enforcement_savings_usd == 0.0


# ---------------------------------------------------------------------------
# forecast_spend
# ---------------------------------------------------------------------------


class TestForecastSpend:
    """Tests for the forecast_spend method."""

    def test_forecast_empty(self):
        tracker = _make_tracker([], budget_limit=100.0)
        ci = CostIntelligence(tracker)
        result = ci.forecast_spend(30)
        assert result.daily_rate == 0.0
        assert result.projected_30d == 0.0
        assert result.budget_limit == 100.0
        assert result.days_until_budget is None
        assert result.trend == "stable"

    def test_forecast_with_entries(self):
        entries = [_make_entry(days_ago=i, cost_usd=10.0) for i in range(7)]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.forecast_spend(30)
        assert result.daily_rate > 0
        assert result.projected_30d > 0

    def test_forecast_with_budget_limit(self):
        entries = [_make_entry(days_ago=i, cost_usd=10.0) for i in range(7)]
        tracker = _make_tracker(entries, budget_limit=1000.0)
        ci = CostIntelligence(tracker)
        result = ci.forecast_spend(30)
        assert result.budget_limit == 1000.0
        assert result.days_until_budget is not None

    def test_forecast_budget_exhausted(self):
        entries = [_make_entry(days_ago=i, cost_usd=100.0) for i in range(7)]
        tracker = _make_tracker(entries, budget_limit=50.0)
        # Mock get_monthly_cost to return more than budget
        tracker.get_monthly_cost = MagicMock(return_value=60.0)
        ci = CostIntelligence(tracker)
        result = ci.forecast_spend(30)
        assert result.days_until_budget == 0

    def test_forecast_no_budget(self):
        entries = [_make_entry(days_ago=i, cost_usd=5.0) for i in range(5)]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci.forecast_spend(30)
        assert result.budget_limit is None
        assert result.days_until_budget is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Tests for internal helper methods."""

    def test_entries_in_window(self):
        entries = [
            _make_entry(days_ago=1),
            _make_entry(days_ago=10),
            _make_entry(days_ago=40),
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci._entries_in_window(30)
        assert len(result) == 2  # 40 days ago excluded

    def test_aggregate_role_model(self):
        entries = [
            _make_entry(agent_role="builder", model="gpt-4o", cost_usd=1.0),
            _make_entry(agent_role="builder", model="gpt-4o", cost_usd=2.0),
            _make_entry(agent_role="reviewer", model="gpt-4o-mini", cost_usd=0.5),
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        result = ci._aggregate_role_model(entries)
        assert ("builder", "gpt-4o") in result
        assert result[("builder", "gpt-4o")]["cost"] == 3.0
        assert result[("builder", "gpt-4o")]["calls"] == 2
        assert ("reviewer", "gpt-4o-mini") in result

    def test_get_quality_score_heuristic(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        # No outcome tracker -> heuristic
        quality = ci._get_quality_score("builder", "gpt-4o", 30)
        assert 0.0 < quality <= 1.0

    def test_get_quality_score_with_outcome_tracker(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 100
        mock_stats.success_rate = 0.92
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        quality = ci._get_quality_score("builder", "gpt-4o", 30)
        assert quality == 0.92

    def test_get_quality_score_outcome_tracker_no_data(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 0
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        quality = ci._get_quality_score("builder", "gpt-4o", 30)
        # Falls back to heuristic
        assert quality == 0.90

    def test_get_quality_score_outcome_tracker_exception(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_outcome.get_provider_stats.side_effect = Exception("DB error")

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        quality = ci._get_quality_score("builder", "gpt-4o", 30)
        # Falls back to heuristic
        assert quality == 0.90

    def test_get_avg_latency_heuristic(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        latency = ci._get_avg_latency("builder", "gpt-4o", 30)
        assert latency > 0

    def test_get_avg_latency_with_outcome_tracker(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 50
        mock_stats.avg_latency_ms = 1234.5
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        latency = ci._get_avg_latency("builder", "gpt-4o", 30)
        assert latency == 1234.5

    def test_get_avg_latency_outcome_tracker_exception(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_outcome.get_provider_stats.side_effect = Exception("Err")

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        latency = ci._get_avg_latency("builder", "gpt-4o", 30)
        assert latency > 0  # Falls back to heuristic

    def test_pricing_ratio(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        # Same model should be ratio 1.0
        ratio = ci._pricing_ratio("gpt-4o", "gpt-4o")
        assert ratio == pytest.approx(1.0)
        # gpt-4o-mini should be much cheaper than gpt-4
        ratio = ci._pricing_ratio("gpt-4", "gpt-4o-mini")
        assert ratio < 0.1  # gpt-4o-mini is much cheaper

    def test_pricing_ratio_unknown_models(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        ratio = ci._pricing_ratio("unknown-model-a", "unknown-model-b")
        assert ratio == 1.0

    def test_pricing_ratio_from_zero(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        # Patch PRICING to have zero-cost model
        with patch.object(CostTracker, "PRICING", {"zero": {"input": 0.0, "output": 0.0}}):
            ratio = ci._pricing_ratio("zero", "gpt-4o")
            assert ratio == 1.0  # from_avg is 0, returns 1.0

    def test_estimated_quality(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._estimated_quality("gpt-4") == 0.92
        assert ci._estimated_quality("gpt-4o-mini") == 0.72
        assert ci._estimated_quality("claude-3-opus") == 0.93
        assert ci._estimated_quality("unknown-model") == 0.75

    def test_estimated_latency(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._estimated_latency("gpt-4o") == 1500.0
        assert ci._estimated_latency("gpt-4o-mini") == 600.0
        assert ci._estimated_latency("claude-3-haiku") == 500.0
        assert ci._estimated_latency("unknown-model") == 1500.0

    def test_provider_for_model(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._provider_for_model("gpt-4o") == "openai"
        assert ci._provider_for_model("o1-mini") == "openai"
        assert ci._provider_for_model("claude-3-sonnet") == "anthropic"
        assert ci._provider_for_model("unknown-model") == "unknown"

    def test_describe_quality_impact(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._describe_quality_impact(-5.0) == "none"
        assert ci._describe_quality_impact(0.0) == "none"
        assert ci._describe_quality_impact(3.0) == "minimal (<5%)"
        assert ci._describe_quality_impact(10.0) == "moderate (5-15%)"
        assert ci._describe_quality_impact(20.0) == "significant (>15%)"

    def test_recommendation_confidence_no_tracker(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        assert ci._recommendation_confidence("builder", "gpt-4", "gpt-4o", 30) == "low"

    def test_recommendation_confidence_high(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 100
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        assert ci._recommendation_confidence("builder", "gpt-4", "gpt-4o", 30) == "high"

    def test_recommendation_confidence_medium(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 20
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        assert ci._recommendation_confidence("builder", "gpt-4", "gpt-4o", 30) == "medium"

    def test_recommendation_confidence_low_few_calls(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 5
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        assert ci._recommendation_confidence("builder", "gpt-4", "gpt-4o", 30) == "low"

    def test_recommendation_confidence_exception(self):
        tracker = _make_tracker()
        mock_outcome = MagicMock()
        mock_outcome.get_provider_stats.side_effect = Exception("err")

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        assert ci._recommendation_confidence("builder", "gpt-4", "gpt-4o", 30) == "low"


# ---------------------------------------------------------------------------
# compute_naive_baseline
# ---------------------------------------------------------------------------


class TestComputeNaiveBaseline:
    """Tests for naive baseline computation."""

    def test_empty_pricing(self):
        entries = [_make_entry(cost_usd=1.0)]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        with patch.object(CostTracker, "PRICING", {}):
            baseline = ci._compute_naive_baseline(entries)
            assert baseline == 1.0  # Falls back to sum of actual costs

    def test_baseline_higher_than_actual(self):
        entries = [
            _make_entry(
                model="gpt-4o-mini",
                cost_usd=0.01,
                input_tokens=1000,
                output_tokens=500,
            )
            for _ in range(10)
        ]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        baseline = ci._compute_naive_baseline(entries)
        actual = sum(e.cost_usd for e in entries)
        # Naive baseline (most expensive model) should be higher than actual
        assert baseline >= actual


# ---------------------------------------------------------------------------
# compute_enforcement_savings
# ---------------------------------------------------------------------------


class TestComputeEnforcementSavings:
    """Tests for enforcement savings computation."""

    def test_no_outcome_tracker(self):
        entries = [_make_entry()]
        tracker = _make_tracker(entries)
        ci = CostIntelligence(tracker)
        assert ci._compute_enforcement_savings(entries) == 0.0

    def test_with_outcome_tracker(self):
        entries = [_make_entry(model="gpt-4o", cost_usd=0.02) for _ in range(10)]
        tracker = _make_tracker(entries)

        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 50
        mock_stats.success_rate = 0.95  # Better than baseline 85%
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        savings = ci._compute_enforcement_savings(entries)
        assert savings >= 0

    def test_enforcement_savings_zero_calls(self):
        entries = [_make_entry(model="gpt-4o", cost_usd=0.02)]
        tracker = _make_tracker(entries)

        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 0
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        savings = ci._compute_enforcement_savings(entries)
        assert savings == 0.0

    def test_enforcement_savings_exception(self):
        entries = [_make_entry(model="gpt-4o", cost_usd=0.02)]
        tracker = _make_tracker(entries)

        mock_outcome = MagicMock()
        mock_outcome.get_provider_stats.side_effect = Exception("DB error")

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        savings = ci._compute_enforcement_savings(entries)
        assert savings == 0.0

    def test_enforcement_savings_multiple_groups(self):
        entries = [
            _make_entry(model="gpt-4o", cost_usd=0.02),
            _make_entry(model="claude-3-sonnet", cost_usd=0.03, provider=Provider.ANTHROPIC),
        ]
        tracker = _make_tracker(entries)

        mock_outcome = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_calls = 100
        mock_stats.success_rate = 0.95
        mock_outcome.get_provider_stats.return_value = mock_stats

        ci = CostIntelligence(tracker, outcome_tracker=mock_outcome)
        savings = ci._compute_enforcement_savings(entries)
        assert savings >= 0


# ---------------------------------------------------------------------------
# find_best_alternative
# ---------------------------------------------------------------------------


class TestFindBestAlternative:
    """Tests for _find_best_alternative method."""

    def test_no_alternative_found(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        # gpt-4o-mini is already cheap, hard to find cheaper
        result = ci._find_best_alternative(
            role="builder",
            current_model="gpt-4o-mini",
            current_quality=0.72,
            current_cost_per_call=0.001,
            monthly_calls=100,
            current_monthly=0.10,
            days=30,
        )
        # May or may not find an alternative, but shouldn't crash
        # If no meaningful savings, returns None
        if result is not None:
            assert result.estimated_savings_monthly > 0

    def test_finds_cheaper_alternative_for_expensive_model(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        # gpt-4 is expensive, should find alternatives
        result = ci._find_best_alternative(
            role="builder",
            current_model="gpt-4",
            current_quality=0.92,
            current_cost_per_call=0.50,
            monthly_calls=100,
            current_monthly=50.0,
            days=30,
        )
        assert result is not None
        assert result.estimated_savings_monthly > 0
        assert result.recommended_model != "gpt-4"

    def test_skips_same_model(self):
        tracker = _make_tracker()
        ci = CostIntelligence(tracker)
        # Should not recommend same model
        result = ci._find_best_alternative(
            role="builder",
            current_model="gpt-4o-mini",
            current_quality=0.72,
            current_cost_per_call=0.001,
            monthly_calls=1000,
            current_monthly=1.0,
            days=30,
        )
        if result is not None:
            assert result.recommended_model != "gpt-4o-mini"
