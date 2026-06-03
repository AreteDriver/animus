"""Judge calibration / meta-evaluation (roadmap B2).

All quality scoring rests on LLM judges whose reliability is otherwise assumed,
not measured. This harness scores a judge against a human-labeled golden set and
reports how well it agrees, so judge quality becomes a tracked, first-class
metric instead of a blind spot. Drift is the change in agreement between a
baseline run and a later one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import EvalCase, EvalMetric


@dataclass
class GoldenCase:
    """One human-labeled example for calibrating a judge.

    ``human_score`` is the trusted 0..1 label the judge is measured against.
    """

    input: str
    output: str
    human_score: float
    expected: str | None = None


@dataclass
class CalibrationReport:
    """How well a judge agrees with the human golden set."""

    judge_model: str | None
    n: int
    mean_abs_error: float
    agreement_rate: float  # fraction within ``tolerance`` of the human label
    correlation: float  # Pearson r between judge and human scores (0 if flat)
    errors: int  # cases where the judge failed to produce a score
    tolerance: float
    judge_scores: list[float] = field(default_factory=list)
    human_scores: list[float] = field(default_factory=list)

    #: A judge is "calibrated" when it tracks humans closely on average.
    MAX_CALIBRATED_MAE: float = 0.15

    @property
    def calibrated(self) -> bool:
        return self.errors == 0 and self.mean_abs_error <= self.MAX_CALIBRATED_MAE


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def calibrate_judge(
    judge: EvalMetric,
    golden: list[GoldenCase],
    *,
    tolerance: float = 0.15,
    judge_model: str | None = None,
) -> CalibrationReport:
    """Score ``judge`` against a human-labeled ``golden`` set.

    Each golden case is scored by the judge and compared to its human label.
    A judge that raises (e.g. ``JudgeError`` from B1) counts as an error, not a
    silent agreement — a broken judge can't be "calibrated".
    """
    judge_scores: list[float] = []
    human_scores: list[float] = []
    abs_errors: list[float] = []
    within = 0
    errors = 0

    for g in golden:
        case = EvalCase(input=g.input, expected=g.expected)
        try:
            js = float(judge.score(g.output, g.expected, case))
        except Exception:
            errors += 1
            continue
        judge_scores.append(js)
        human_scores.append(g.human_score)
        err = abs(js - g.human_score)
        abs_errors.append(err)
        if err <= tolerance:
            within += 1

    scored = len(abs_errors)
    mae = sum(abs_errors) / scored if scored else 1.0
    agreement = within / scored if scored else 0.0
    corr = _pearson(judge_scores, human_scores)

    return CalibrationReport(
        judge_model=judge_model,
        n=len(golden),
        mean_abs_error=mae,
        agreement_rate=agreement,
        correlation=corr,
        errors=errors,
        tolerance=tolerance,
        judge_scores=judge_scores,
        human_scores=human_scores,
    )


def judge_drift(baseline: CalibrationReport, current: CalibrationReport) -> float:
    """Change in judge error vs a baseline. Positive = the judge got worse.

    Track this per judge model over time; a rising value means the judge is
    drifting away from human agreement and the scores it produces are less
    trustworthy than they were.
    """
    return current.mean_abs_error - baseline.mean_abs_error
