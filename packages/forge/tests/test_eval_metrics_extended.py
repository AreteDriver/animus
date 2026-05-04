"""Tests for SelfAuditedLengthMetric and NegativeExampleMetric."""

from __future__ import annotations

from animus_forge.evaluation.base import EvalCase
from animus_forge.evaluation.metrics import (
    NegativeExampleMetric,
    SelfAuditedLengthMetric,
)


def _case(**metadata) -> EvalCase:
    return EvalCase(input="x", expected="y", metadata=metadata)


# =========================================================================
# SelfAuditedLengthMetric
# =========================================================================


def test_self_audit_honest_within_budget_scores_one():
    metric = SelfAuditedLengthMetric(max_length=20)
    text = "one two three four five six seven eight <wc>8</wc>"
    assert metric.score(text, None, _case()) == 1.0


def test_self_audit_tag_absent_returns_mid_score():
    metric = SelfAuditedLengthMetric(max_length=20, tag_absent_score=0.5)
    text = "one two three four five six seven eight"
    assert metric.score(text, None, _case()) == 0.5


def test_self_audit_claim_mismatch_low_score():
    metric = SelfAuditedLengthMetric(max_length=200, tolerance=0.05, mismatch_score=0.3)
    # 8 actual words, claims 50 — huge mismatch
    text = "one two three four five six seven eight <wc>50</wc>"
    assert metric.score(text, None, _case()) == 0.3


def test_self_audit_honest_but_overlong_reduces_score():
    metric = SelfAuditedLengthMetric(max_length=5)
    text = "one two three four five six seven eight nine ten <wc>10</wc>"
    score = metric.score(text, None, _case())
    assert 0.0 < score < 1.0  # reduced by overage
    assert score == 5 / 10  # max/actual


def test_self_audit_claim_tolerance_accepts_close():
    metric = SelfAuditedLengthMetric(max_length=200, tolerance=0.20)
    # 10 actual, claims 11 — within 20% of 11
    text = "one two three four five six seven eight nine ten <wc>11</wc>"
    assert metric.score(text, None, _case()) == 1.0


def test_self_audit_tag_stripped_before_counting():
    metric = SelfAuditedLengthMetric(max_length=20)
    # <wc>...</wc> should NOT count as body words
    text = "a b c <wc>3</wc>"
    assert metric.score(text, None, _case()) == 1.0


def test_self_audit_case_insensitive_tag():
    metric = SelfAuditedLengthMetric(max_length=20)
    text = "a b c <WC>3</WC>"
    assert metric.score(text, None, _case()) == 1.0


def test_self_audit_no_max_length():
    metric = SelfAuditedLengthMetric(max_length=None)
    # 50 words, claims 50, no ceiling → 1.0
    text = " ".join(["word"] * 50) + " <wc>50</wc>"
    assert metric.score(text, None, _case()) == 1.0


# =========================================================================
# NegativeExampleMetric
# =========================================================================


def test_negative_example_no_examples_scores_one():
    metric = NegativeExampleMetric()
    result = metric.score("any output here", None, _case())
    assert result == 1.0


def test_negative_example_identical_to_negative_scores_zero():
    metric = NegativeExampleMetric()
    case = _case(negative_examples=["reject this generic output"])
    result = metric.score("reject this generic output", None, case)
    assert result == 0.0


def test_negative_example_dissimilar_scores_near_one():
    metric = NegativeExampleMetric()
    case = _case(negative_examples=["boilerplate passionate team player"])
    result = metric.score("concrete evidence from the codebase", None, case)
    assert result > 0.9


def test_negative_example_partial_overlap_middle_score():
    metric = NegativeExampleMetric()
    case = _case(negative_examples=["generic boilerplate phrase"])
    # Half the words overlap
    result = metric.score("generic boilerplate specific concrete", None, case)
    assert 0.0 < result < 1.0


def test_negative_example_multiple_examples_takes_worst():
    metric = NegativeExampleMetric()
    case = _case(
        negative_examples=[
            "completely unrelated to output",
            "exact match output test",
        ]
    )
    result = metric.score("exact match output test", None, case)
    assert result == 0.0  # matched second example


def test_negative_example_empty_output_scores_one():
    metric = NegativeExampleMetric()
    case = _case(negative_examples=["any example"])
    result = metric.score("", None, case)
    assert result == 1.0


def test_negative_example_empty_example_skipped():
    metric = NegativeExampleMetric()
    case = _case(negative_examples=["", "real example text"])
    # First example is empty; second should drive the score
    result = metric.score("completely different words", None, case)
    assert 0.9 < result <= 1.0
