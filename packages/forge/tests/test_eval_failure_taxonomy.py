"""Tests for evaluation/failure_taxonomy.py — FailureMode + classifier."""

from __future__ import annotations

from animus_forge.evaluation.base import EvalCase, EvalResult, EvalStatus
from animus_forge.evaluation.failure_taxonomy import (
    FailureClassifier,
    FailureMode,
    tag_results,
)


def _case(**metadata) -> EvalCase:
    return EvalCase(input="x", expected="y", metadata=metadata)


def _result(
    status: EvalStatus = EvalStatus.FAILED,
    output: str | None = "some output",
    error: str | None = None,
    metrics: dict[str, float] | None = None,
    case: EvalCase | None = None,
) -> EvalResult:
    return EvalResult(
        case=case or _case(),
        status=status,
        score=0.0 if status != EvalStatus.PASSED else 1.0,
        output=output,
        metrics=metrics or {},
        error=error,
    )


def test_passed_returns_none():
    classifier = FailureClassifier()
    assert classifier.classify(_result(status=EvalStatus.PASSED)) is None


def test_skipped_returns_none():
    classifier = FailureClassifier()
    assert classifier.classify(_result(status=EvalStatus.SKIPPED)) is None


def test_error_timeout():
    classifier = FailureClassifier()
    result = _result(status=EvalStatus.ERROR, error="Request timed out after 30s")
    assert classifier.classify(result) == FailureMode.TIMEOUT


def test_error_over_budget():
    classifier = FailureClassifier()
    result = _result(status=EvalStatus.ERROR, error="budget exceeded")
    assert classifier.classify(result) == FailureMode.OVER_BUDGET


def test_error_rate_limit_is_provider_error():
    classifier = FailureClassifier()
    result = _result(status=EvalStatus.ERROR, error="HTTP 429 rate limit")
    assert classifier.classify(result) == FailureMode.PROVIDER_ERROR


def test_error_unknown_is_provider_error():
    classifier = FailureClassifier()
    result = _result(status=EvalStatus.ERROR, error="unexpected upstream failure")
    assert classifier.classify(result) == FailureMode.PROVIDER_ERROR


def test_refused_from_refusal_pattern():
    classifier = FailureClassifier()
    result = _result(output="I can't help with that request.")
    assert classifier.classify(result) == FailureMode.REFUSED


def test_contract_violation_from_schema_valid_zero():
    classifier = FailureClassifier()
    result = _result(metrics={"schema_valid": 0.0, "correctness": 0.8})
    assert classifier.classify(result) == FailureMode.CONTRACT_VIOLATION


def test_schema_drift_when_expects_json():
    classifier = FailureClassifier()
    case = _case(expects_json=True)
    result = _result(metrics={"schema_valid": 0.2}, case=case)
    assert classifier.classify(result) == FailureMode.SCHEMA_DRIFT


def test_judge_disagreement_detected():
    classifier = FailureClassifier(judge_variance_threshold=0.1)
    case = _case(judge_samples=[0.9, 0.3, 0.8])
    result = _result(case=case)
    assert classifier.classify(result) == FailureMode.JUDGE_DISAGREEMENT


def test_judge_agreement_not_flagged():
    classifier = FailureClassifier(judge_variance_threshold=0.2)
    case = _case(judge_samples=[0.85, 0.82, 0.80])
    result = _result(case=case)
    assert classifier.classify(result) != FailureMode.JUDGE_DISAGREEMENT


def test_hallucination_from_low_factuality():
    classifier = FailureClassifier()
    result = _result(metrics={"factuality": 0.2})
    assert classifier.classify(result) == FailureMode.HALLUCINATION


def test_flaky_from_metadata():
    classifier = FailureClassifier()
    case = _case(flaky=True)
    result = _result(case=case)
    assert classifier.classify(result) == FailureMode.FLAKY


def test_wrong_answer_default():
    classifier = FailureClassifier()
    result = _result(output="totally wrong", metrics={"correctness": 0.1})
    assert classifier.classify(result) == FailureMode.WRONG_ANSWER


def test_expected_failure_override():
    classifier = FailureClassifier()
    case = _case(expected_failure_mode="hallucination")
    result = _result(case=case, metrics={"schema_valid": 0.0})
    # Override beats contract_violation
    assert classifier.classify(result) == FailureMode.HALLUCINATION


def test_tag_results_mutates_and_returns_counts():
    results = [
        _result(status=EvalStatus.PASSED),
        _result(status=EvalStatus.ERROR, error="timed out"),
        _result(output="I cannot help"),
    ]
    counts = tag_results(results)
    assert counts["passed"] == 1
    assert counts.get("timeout") == 1
    assert counts.get("refused") == 1
    assert results[0].failure_mode is None
    assert results[1].failure_mode == "timeout"
    assert results[2].failure_mode == "refused"
