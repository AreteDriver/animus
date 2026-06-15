"""Cost-audit step handler mixin for WorkflowExecutor.

Provides the ``cost_audit`` step type. The step queries the executor's
attached ``BudgetManager``, runs the trailing-window anomaly detector
from ``budget/cost_audit.py``, and emits a structured report into the
workflow context.

Params (from ``step.params``):
    window_hours: Size of the current-window bucket (default 24).
    baseline_days: Size of the trailing baseline window (default 7).
    sigma_threshold: Flag if window > baseline_mean + N·stdev (default 2.0).
    ratio_threshold: Flag if window / baseline_mean ≥ N (default 1.5).
    fail_on_anomaly: When True, the step raises if any anomaly fires.
        Default False — emit-and-continue is the common case (the next
        workflow step decides what to do with the report).

Outputs (returned dict):
    has_anomalies: bool
    anomaly_count: int
    anomalies: list of {agent_id, window_et, baseline_mean, ratio, sigma, reason}
    window_total_et, baseline_total_et: floats
    window_by_agent, baseline_by_agent: dict[str, float]
"""

from __future__ import annotations

import logging

from animus_kernel.budget.cost_audit import audit_cost

from .loader import StepConfig

logger = logging.getLogger(__name__)


class CostAuditHandlerMixin:
    """Mixin providing the cost_audit step handler.

    Expects the following attributes from the host class:
    - dry_run: bool
    - budget_manager: BudgetManager | None
    """

    def _execute_cost_audit(self, step: StepConfig, context: dict) -> dict:
        """Execute a cost_audit anomaly-detection step."""
        params = step.params or {}
        window_hours = int(params.get("window_hours", 24))
        baseline_days = int(params.get("baseline_days", 7))
        sigma_threshold = float(params.get("sigma_threshold", 2.0))
        ratio_threshold = float(params.get("ratio_threshold", 1.5))
        fail_on_anomaly = bool(params.get("fail_on_anomaly", False))

        if getattr(self, "dry_run", False):
            return {
                "has_anomalies": False,
                "anomaly_count": 0,
                "anomalies": [],
                "window_total_et": 0.0,
                "baseline_total_et": 0.0,
                "window_by_agent": {},
                "baseline_by_agent": {},
                "dry_run": True,
            }

        budget_manager = getattr(self, "budget_manager", None)
        if budget_manager is None:
            logger.info(
                "cost_audit step %s ran with no budget_manager attached; emitting empty report",
                step.id,
            )
            return {
                "has_anomalies": False,
                "anomaly_count": 0,
                "anomalies": [],
                "window_total_et": 0.0,
                "baseline_total_et": 0.0,
                "window_by_agent": {},
                "baseline_by_agent": {},
                "warning": "no_budget_manager",
            }

        # Pull a large history slice — audit_cost slices it itself
        history = budget_manager.get_usage_history(limit=10_000)
        multipliers = getattr(budget_manager.config, "model_multipliers", None)

        report = audit_cost(
            history=history,
            window_hours=window_hours,
            baseline_days=baseline_days,
            sigma_threshold=sigma_threshold,
            ratio_threshold=ratio_threshold,
            model_multipliers=multipliers,
        )

        result = report.to_dict()
        result["has_anomalies"] = report.has_anomalies

        if fail_on_anomaly and report.has_anomalies:
            agents = ", ".join(a.agent_id for a in report.anomalies)
            raise RuntimeError(f"cost_audit flagged {len(report.anomalies)} agent(s): {agents}")

        return result
