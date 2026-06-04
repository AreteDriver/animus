"""Tests for judge calibration / meta-eval (roadmap B2)."""

from __future__ import annotations

import pytest

from animus_forge.evaluation.base import EvalMetric
from animus_forge.evaluation.judge_calibration import (
    CalibrationReport,
    GoldenCase,
    calibrate_judge,
    judge_drift,
)
from animus_forge.evaluation.metrics import JudgeError


class _ScriptedJudge(EvalMetric):
    """A judge whose score IS the output string parsed as a float."""

    @property
    def name(self) -> str:
        return "scripted_judge"

    def score(self, output, expected, case) -> float:
        return float(output)


class _RaisingJudge(EvalMetric):
    @property
    def name(self) -> str:
        return "raising_judge"

    def score(self, output, expected, case) -> float:
        raise JudgeError("judge down")


def _golden(pairs):
    # pairs: list of (judge_output_str, human_score)
    return [GoldenCase(input="q", output=o, human_score=h) for o, h in pairs]


class TestCalibrateJudge:
    def test_perfect_agreement(self):
        golden = _golden([("0.9", 0.9), ("0.2", 0.2), ("0.7", 0.7)])
        rep = calibrate_judge(_ScriptedJudge(), golden)
        assert rep.mean_abs_error == 0.0
        assert rep.agreement_rate == 1.0
        assert rep.correlation > 0.99
        assert rep.calibrated is True
        assert rep.errors == 0

    def test_poorly_calibrated_judge(self):
        # Judge says 0.5 everywhere; humans vary → high error, not calibrated.
        golden = _golden([("0.5", 0.9), ("0.5", 0.1), ("0.5", 0.8)])
        rep = calibrate_judge(_ScriptedJudge(), golden)
        assert rep.mean_abs_error > 0.15
        assert rep.calibrated is False

    def test_judge_errors_counted_not_silent(self):
        golden = _golden([("0.9", 0.9), ("0.9", 0.9)])
        rep = calibrate_judge(_RaisingJudge(), golden)
        assert rep.errors == 2
        assert rep.calibrated is False  # a judge that can't score isn't calibrated

    def test_drift_is_change_in_error(self):
        base = CalibrationReport(
            judge_model="m",
            n=3,
            mean_abs_error=0.05,
            agreement_rate=1.0,
            correlation=1.0,
            errors=0,
            tolerance=0.15,
        )
        worse = CalibrationReport(
            judge_model="m",
            n=3,
            mean_abs_error=0.20,
            agreement_rate=0.6,
            correlation=0.5,
            errors=0,
            tolerance=0.15,
        )
        assert judge_drift(base, worse) == pytest.approx(0.15)  # got worse
        assert judge_drift(worse, base) == pytest.approx(-0.15)  # improved
