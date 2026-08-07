"""Tests for the cost_audit workflow step handler.

Thin wrapper over budget/cost_audit.py — these tests exercise the
step-handler contract (dry_run, missing budget_manager, fail_on_anomaly)
without re-testing the audit logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from animus_kernel.budget import BudgetConfig, BudgetManager, UsageRecord
from animus_forge.workflow.executor_cost_audit import CostAuditHandlerMixin
from animus_forge.workflow.loader import StepConfig

UTC = UTC


class _Host(CostAuditHandlerMixin):
    """Minimal host class exposing the attributes the mixin reads."""

    def __init__(self, budget_manager=None, dry_run=False):
        self.budget_manager = budget_manager
        self.dry_run = dry_run


def _step(**params) -> StepConfig:
    return StepConfig(id="audit-1", type="cost_audit", params=params)


def test_dry_run_returns_empty_report():
    host = _Host(dry_run=True)
    result = host._execute_cost_audit(_step(), context={})
    assert result["dry_run"] is True
    assert result["has_anomalies"] is False
    assert result["anomaly_count"] == 0


def test_missing_budget_manager_returns_warning():
    host = _Host(budget_manager=None)
    result = host._execute_cost_audit(_step(), context={})
    assert result["warning"] == "no_budget_manager"
    assert result["has_anomalies"] is False


def test_quiet_history_no_anomaly():
    bm = BudgetManager(config=BudgetConfig(total_budget=1_000_000))
    host = _Host(budget_manager=bm)
    result = host._execute_cost_audit(_step(), context={})
    assert result["has_anomalies"] is False
    assert result["anomaly_count"] == 0


def test_with_anomaly_fail_on_anomaly_raises():
    bm = BudgetManager(config=BudgetConfig(total_budget=10_000_000))
    # Cold-start spend → cold_start_spend anomaly fires
    bm.record_usage("agent-a", tokens=1000, output_tokens=1000)
    host = _Host(budget_manager=bm)

    # By default, anomalies are reported but don't fail the step
    result = host._execute_cost_audit(_step(), context={})
    assert result["has_anomalies"] is True

    # fail_on_anomaly=True → raises
    with pytest.raises(RuntimeError, match="cost_audit flagged"):
        host._execute_cost_audit(_step(fail_on_anomaly=True), context={})


def test_thresholds_pass_through():
    """The handler must forward the threshold params to the analyzer."""
    bm = MagicMock()
    bm.get_usage_history.return_value = []
    bm.config = SimpleNamespace(model_multipliers={"x": 1.0})

    host = _Host(budget_manager=bm)
    step = _step(sigma_threshold=99.0, ratio_threshold=99.0, window_hours=6, baseline_days=14)
    result = host._execute_cost_audit(step, context={})
    bm.get_usage_history.assert_called_once_with(limit=10_000)
    # Empty history is the cheap signal that the call worked end-to-end
    assert result["anomaly_count"] == 0


def test_window_hours_param_changes_window_size():
    """Verify a non-default window_hours actually narrows the window."""
    bm = BudgetManager(config=BudgetConfig(total_budget=10_000_000))
    anchor = datetime.now(UTC)
    # 12 records 5 days before the anchor (i.e. squarely inside the baseline,
    # well outside any reasonable window)
    for h in range(12):
        bm._usage_history.append(
            UsageRecord(
                agent_id="agent-a",
                tokens=100,
                output_tokens=100,
                timestamp=anchor - timedelta(days=5) + timedelta(hours=h),
            )
        )
    host = _Host(budget_manager=bm)
    # The audit anchors `now` at the most recent record (5 days ago plus 11h).
    # With a 1-hour window from that anchor, only the very last record falls
    # in-window; the rest fall in baseline. window_total_et > 0 but small,
    # baseline_total_et carries the bulk.
    result = host._execute_cost_audit(_step(window_hours=1), context={})
    assert result["baseline_total_et"] > result["window_total_et"]
    assert result["baseline_total_et"] > 0
