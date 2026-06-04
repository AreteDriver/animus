"""Tests for auto-promotion (roadmap B5)."""

from __future__ import annotations

from unittest.mock import MagicMock

from animus_forge.coordination.auto_promote import auto_promote_on_improvement
from animus_forge.evaluation.compare import ComparisonReport


def _summary(n=200):
    from animus_forge.evaluation.compare import RunSummary

    return RunSummary(
        run_id="x",
        suite_name="s",
        model=None,
        rubric_name=None,
        rubric_version=None,
        prompt_version=None,
        avg_score=0.5,
        pass_rate=0.5,
        total_cost_usd=0.0,
        duration_ms=0.0,
        total_cases=n,
        case_scores=[],
        failure_buckets={},
    )


def _report(ci, delta):
    s = _summary()
    return ComparisonReport(
        baseline=s,
        candidate=s,
        score_delta=delta,
        score_ci_95=ci,
        pass_rate_delta=0.0,
        cost_delta_usd=0.0,
        latency_delta_ms=0.0,
        failure_delta={},
    )


def _call(report):
    evo = MagicMock()
    evo.propose_patch.return_value = "RESULT"
    out = auto_promote_on_improvement(
        report,
        workflow_id="wf",
        prompt_version="v3",
        new_yaml_content="name: wf\nversion: 3\n",
        evolution=evo,
    )
    return out, evo


class TestAutoPromote:
    def test_significant_improvement_proposes_patch(self):
        out, evo = _call(_report((0.02, 0.18), delta=0.10))  # excludes 0, positive
        assert out == "RESULT"
        evo.propose_patch.assert_called_once()
        patch = evo.propose_patch.call_args[0][0]
        assert patch.workflow_id == "wf"
        assert patch.proposed_by == "auto-promotion"
        assert "v3" in patch.performance_evidence

    def test_not_significant_does_not_promote(self):
        out, evo = _call(_report((-0.05, 0.15), delta=0.05))  # CI includes 0
        assert out is None
        evo.propose_patch.assert_not_called()

    def test_significant_regression_does_not_promote(self):
        out, evo = _call(_report((-0.20, -0.04), delta=-0.12))  # significant but worse
        assert out is None
        evo.propose_patch.assert_not_called()

    def test_underpowered_does_not_promote(self):
        # Wide CI, not significant → underpowered, must not promote.
        report = _report((-0.10, 0.14), delta=0.02)
        assert report.underpowered is True
        out, evo = _call(report)
        assert out is None
        evo.propose_patch.assert_not_called()
