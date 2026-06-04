"""Tests for evaluation/compare.py — A/B comparison + bootstrap CI."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from animus_forge.evaluation.compare import (
    ComparisonReport,
    RunSummary,
    bootstrap_ci_delta,
    compare_runs,
    load_run_summary,
    resolve_run_id,
)


def test_bootstrap_ci_identical_scores_near_zero():
    scores = [0.8, 0.9, 0.7, 0.85, 0.75]
    lo, hi = bootstrap_ci_delta(scores, scores, n=500, seed=42)
    assert abs(lo) < 0.1
    assert abs(hi) < 0.1


def test_bootstrap_ci_clear_improvement_positive():
    baseline = [0.3, 0.4, 0.35, 0.32, 0.38] * 4
    candidate = [0.8, 0.85, 0.9, 0.82, 0.88] * 4
    lo, hi = bootstrap_ci_delta(baseline, candidate, n=500, seed=42)
    assert lo > 0  # CI excludes zero on the positive side
    assert hi > lo


def test_bootstrap_ci_clear_regression_negative():
    baseline = [0.9, 0.85, 0.88, 0.92, 0.87] * 4
    candidate = [0.3, 0.25, 0.32, 0.28, 0.30] * 4
    lo, hi = bootstrap_ci_delta(baseline, candidate, n=500, seed=42)
    assert hi < 0
    assert lo < hi


def test_bootstrap_ci_empty_returns_zero():
    assert bootstrap_ci_delta([], [1.0], n=100) == (0.0, 0.0)
    assert bootstrap_ci_delta([1.0], [], n=100) == (0.0, 0.0)


def test_bootstrap_ci_deterministic_with_seed():
    scores_a = [0.5, 0.6, 0.7]
    scores_b = [0.7, 0.8, 0.9]
    ci1 = bootstrap_ci_delta(scores_a, scores_b, n=200, seed=123)
    ci2 = bootstrap_ci_delta(scores_a, scores_b, n=200, seed=123)
    assert ci1 == ci2


# =========================================================================
# load_run_summary
# =========================================================================


def _fake_run(
    run_id: str = "abc123",
    suite: str = "planner",
    avg_score: float = 0.8,
    case_results: list[dict] | None = None,
) -> dict:
    return {
        "id": run_id,
        "suite_name": suite,
        "model": "opus-4.7",
        "rubric_name": "code-edit",
        "rubric_version": "1.0.0",
        "prompt_version": None,
        "avg_score": avg_score,
        "pass_rate": 0.8,
        "total_cost_usd": 0.0,
        "duration_ms": 1000.0,
        "total_cases": len(case_results or []),
        "case_results": case_results or [],
    }


def test_load_run_summary_builds_case_scores_and_buckets():
    store = MagicMock()
    store.get_run.return_value = _fake_run(
        case_results=[
            {"case_name": "a", "status": "passed", "score": 1.0, "failure_mode": None},
            {"case_name": "b", "status": "failed", "score": 0.0, "failure_mode": "refused"},
            {"case_name": "c", "status": "failed", "score": 0.3, "failure_mode": "wrong_answer"},
        ],
    )
    summary = load_run_summary(store, "abc123")
    assert summary.case_scores == [1.0, 0.0, 0.3]
    assert summary.failure_buckets == {"passed": 1, "refused": 1, "wrong_answer": 1}


def test_load_run_summary_counts_content_failures():
    store = MagicMock()
    store.get_run.return_value = _fake_run(
        case_results=[
            {
                "case_name": "a",
                "status": "passed",
                "score": 1.0,
                "failure_mode": None,
                "content_failure_modes": [],
            },
            {
                "case_name": "b",
                "status": "failed",
                "score": 0.3,
                "failure_mode": "wrong_answer",
                "content_failure_modes": ["F4_too_generic", "F5_overlong"],
            },
            {
                "case_name": "c",
                "status": "failed",
                "score": 0.2,
                "failure_mode": "wrong_answer",
                "content_failure_modes": ["F4_too_generic"],
            },
        ],
    )
    summary = load_run_summary(store, "x")
    # F4 appears on two cases, F5 on one
    assert summary.content_failure_buckets == {
        "F4_too_generic": 2,
        "F5_overlong": 1,
    }


def test_load_run_summary_missing_run_raises():
    store = MagicMock()
    store.get_run.return_value = None
    with pytest.raises(ValueError, match="not found"):
        load_run_summary(store, "missing")


# =========================================================================
# compare_runs
# =========================================================================


def test_compare_runs_improvement_flags_ci_and_direction():
    baseline_cases = [
        {"case_name": f"c{i}", "status": "failed", "score": 0.3, "failure_mode": "wrong_answer"}
        for i in range(20)
    ]
    candidate_cases = [
        {"case_name": f"c{i}", "status": "passed", "score": 0.9, "failure_mode": None}
        for i in range(20)
    ]
    store = MagicMock()
    store.get_run.side_effect = [
        _fake_run("base", avg_score=0.3, case_results=baseline_cases),
        _fake_run("cand", avg_score=0.9, case_results=candidate_cases),
    ]

    report = compare_runs("base", "cand", store=store, bootstrap_n=300, seed=42)

    assert isinstance(report, ComparisonReport)
    assert report.direction == "improvement"
    assert report.significant
    assert report.score_ci_95[0] > 0
    assert any("ci_excludes_zero:improvement" in f for f in report.regression_flags)


def test_compare_runs_regression_flags():
    baseline_cases = [
        {"case_name": f"c{i}", "status": "passed", "score": 0.9, "failure_mode": None}
        for i in range(20)
    ]
    candidate_cases = [
        {"case_name": f"c{i}", "status": "failed", "score": 0.2, "failure_mode": "schema_drift"}
        for i in range(20)
    ]
    store = MagicMock()
    store.get_run.side_effect = [
        _fake_run("base", avg_score=0.9, case_results=baseline_cases),
        _fake_run("cand", avg_score=0.2, case_results=candidate_cases),
    ]

    report = compare_runs("base", "cand", store=store, bootstrap_n=300, seed=42)

    assert report.direction == "regression"
    assert report.significant
    assert any("composite_regression" in f for f in report.regression_flags)
    assert any("bucket_spike:schema_drift" in f for f in report.regression_flags)


def test_compare_runs_content_failure_delta_surfaced():
    baseline_cases = [
        {
            "case_name": f"c{i}",
            "status": "failed",
            "score": 0.3,
            "failure_mode": "wrong_answer",
            "content_failure_modes": ["F4_too_generic"],
        }
        for i in range(20)
    ]
    candidate_cases = [
        {
            "case_name": f"c{i}",
            "status": "failed",
            "score": 0.2,
            "failure_mode": "wrong_answer",
            "content_failure_modes": ["F3_weak_evidence", "F4_too_generic"],
        }
        for i in range(20)
    ]
    store = MagicMock()
    store.get_run.side_effect = [
        _fake_run("base", avg_score=0.3, case_results=baseline_cases),
        _fake_run("cand", avg_score=0.2, case_results=candidate_cases),
    ]

    report = compare_runs("base", "cand", store=store, bootstrap_n=200, seed=42)

    # F3 went from 0→20, F4 stayed flat
    assert report.content_failure_delta.get("F3_weak_evidence") == 20
    assert "F4_too_generic" not in report.content_failure_delta  # unchanged
    assert any("content_spike:F3_weak_evidence" in f for f in report.regression_flags)


def test_compare_runs_suite_mismatch_raises():
    store = MagicMock()
    store.get_run.side_effect = [
        _fake_run("a", suite="planner"),
        _fake_run("b", suite="briefing"),
    ]
    with pytest.raises(ValueError, match="Suite mismatch"):
        compare_runs("a", "b", store=store, bootstrap_n=50)


# =========================================================================
# resolve_run_id
# =========================================================================


def test_resolve_run_id_last():
    store = MagicMock()
    store.query_runs.return_value = [{"id": "full-uuid-aaaa-bbbb"}]
    assert resolve_run_id(store, "last") == "full-uuid-aaaa-bbbb"
    assert resolve_run_id(store, "latest") == "full-uuid-aaaa-bbbb"


def test_resolve_run_id_prev():
    store = MagicMock()
    store.query_runs.return_value = [{"id": "r1"}, {"id": "r2"}]
    assert resolve_run_id(store, "prev") == "r2"


def test_resolve_run_id_prev_insufficient_raises():
    store = MagicMock()
    store.query_runs.return_value = [{"id": "only"}]
    with pytest.raises(ValueError, match="Fewer than 2"):
        resolve_run_id(store, "prev")


def test_resolve_run_id_prefix_match():
    store = MagicMock()
    store.query_runs.return_value = [
        {"id": "7b3a2c1d-aaaa-bbbb-cccc-dddddddddddd"},
        {"id": "9fff1111-aaaa-bbbb-cccc-dddddddddddd"},
    ]
    assert resolve_run_id(store, "7b3a2c1d").startswith("7b3a2c1d")


def test_resolve_run_id_ambiguous_prefix_raises():
    store = MagicMock()
    store.query_runs.return_value = [
        {"id": "7b3a2c1d-1"},
        {"id": "7b3a2c1d-2"},
    ]
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_run_id(store, "7b3a2c1d")


def test_resolve_run_id_full_uuid_passthrough():
    full = "12345678-1234-1234-1234-123456789abc"
    store = MagicMock()
    # Full uuid (>=32 chars) should not hit query_runs
    assert resolve_run_id(store, full) == full


def test_comparison_report_significant_property():
    summary = RunSummary(
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
        total_cases=0,
        case_scores=[],
        failure_buckets={},
    )
    sig = ComparisonReport(
        baseline=summary,
        candidate=summary,
        score_delta=0.1,
        score_ci_95=(0.02, 0.18),
        pass_rate_delta=0.0,
        cost_delta_usd=0.0,
        latency_delta_ms=0.0,
        failure_delta={},
    )
    assert sig.significant is True
    assert sig.direction == "improvement"

    not_sig = ComparisonReport(
        baseline=summary,
        candidate=summary,
        score_delta=0.05,
        score_ci_95=(-0.03, 0.13),
        pass_rate_delta=0.0,
        cost_delta_usd=0.0,
        latency_delta_ms=0.0,
        failure_delta={},
    )
    assert not_sig.significant is False


def _report(ci, n=10, delta=0.0):
    s = RunSummary(
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


class TestPowerAdvisor:
    """B6: 'not significant' on a small suite must not read as 'no difference'."""

    def test_significant_has_no_power_note(self):
        r = _report((0.02, 0.18), n=200, delta=0.1)
        assert r.significant is True
        assert r.underpowered is False
        assert r.power_note == ""

    def test_not_significant_wide_ci_is_underpowered(self):
        # Half-width 0.11 > 0.05 meaningful effect → underpowered.
        r = _report((-0.10, 0.12), n=8)
        assert r.significant is False
        assert r.underpowered is True
        assert "Underpowered" in r.power_note
        assert "n=8" in r.power_note

    def test_not_significant_tight_ci_is_equivalence(self):
        # Half-width 0.02 < 0.05 → genuinely equivalent, not underpowered.
        r = _report((-0.02, 0.02), n=500)
        assert r.significant is False
        assert r.equivalent is True
        assert r.underpowered is False
        assert "equivalent" in r.power_note.lower()

    def test_offcenter_ci_one_side_wide_is_underpowered_not_equivalent(self):
        # C10: bounds-based vs half-width. CI [-0.005, 0.085] has half-width
        # 0.045 < 0.05 — the old centered test wrongly called this "equivalent"
        # — yet the upper bound 0.085 can't rule out a meaningful +effect.
        r = _report((-0.005, 0.085), n=30)
        assert r.significant is False
        assert r.ci_halfwidth < r.MEANINGFUL_EFFECT  # would fool half-width logic
        assert r.equivalent is False  # bounds-based catches the off-center reach
        assert r.underpowered is True
        assert "Underpowered" in r.power_note

    def test_equivalence_note_reports_actual_bound_envelope(self):
        # C10: the "within ±x" wording is derived from the real CI bounds, not
        # the fixed threshold. CI [-0.03, 0.02] → envelope max|bound| = 0.030.
        r = _report((-0.03, 0.02), n=200)
        assert r.equivalent is True
        assert "±0.030" in r.power_note
