"""Tests for evaluation/failure_taxonomy_content.py — F1-F8 classifier."""

from __future__ import annotations

from animus_forge.evaluation.base import EvalCase, EvalResult, EvalStatus
from animus_forge.evaluation.failure_taxonomy_content import (
    ContentFailureClassifier,
    ContentFailureMode,
    tag_content_failures,
)


def _case(**metadata) -> EvalCase:
    return EvalCase(input="x", expected="y", metadata=metadata)


def _result(
    output: str = "output",
    metrics: dict[str, float] | None = None,
    case: EvalCase | None = None,
) -> EvalResult:
    return EvalResult(
        case=case or _case(),
        status=EvalStatus.FAILED,
        score=0.0,
        output=output,
        metrics=metrics or {},
    )


# =========================================================================
# F1 missing constraint
# =========================================================================


def test_f1_missing_constraint_triggers():
    classifier = ContentFailureClassifier()
    case = _case(required_phrases=["must include X", "regulatory"])
    result = _result(output="generic output without markers", case=case)
    modes = classifier.classify(result)
    assert ContentFailureMode.F1_MISSING_CONSTRAINT in modes


def test_f1_all_constraints_present_clean():
    classifier = ContentFailureClassifier()
    case = _case(required_phrases=["alpha", "beta"])
    result = _result(output="contains alpha and beta words", case=case)
    modes = classifier.classify(result)
    assert ContentFailureMode.F1_MISSING_CONSTRAINT not in modes


def test_f1_no_constraint_no_trigger():
    classifier = ContentFailureClassifier()
    result = _result(output="anything")
    modes = classifier.classify(result)
    assert ContentFailureMode.F1_MISSING_CONSTRAINT not in modes


# =========================================================================
# F2 wrong assumption
# =========================================================================


def test_f2_wrong_assumption_triggers():
    classifier = ContentFailureClassifier()
    case = _case(assumption_errors=["python 2 is fine", "uuid4 is sequential"])
    result = _result(
        output="Python 2 is fine for this project; use uuid4 which is sequential.",
        case=case,
    )
    modes = classifier.classify(result)
    assert ContentFailureMode.F2_WRONG_ASSUMPTION in modes


def test_f2_correct_assumption_clean():
    classifier = ContentFailureClassifier()
    case = _case(assumption_errors=["python 2 is fine"])
    result = _result(output="Python 3.12 is the target.", case=case)
    modes = classifier.classify(result)
    assert ContentFailureMode.F2_WRONG_ASSUMPTION not in modes


# =========================================================================
# F3 weak evidence
# =========================================================================


def test_f3_weak_evidence_below_threshold():
    classifier = ContentFailureClassifier()
    result = _result(metrics={"evidence_quality": 0.3})
    modes = classifier.classify(result)
    assert ContentFailureMode.F3_WEAK_EVIDENCE in modes


def test_f3_strong_evidence_clean():
    classifier = ContentFailureClassifier()
    result = _result(metrics={"evidence_quality": 0.9})
    modes = classifier.classify(result)
    assert ContentFailureMode.F3_WEAK_EVIDENCE not in modes


# =========================================================================
# F4 too generic
# =========================================================================


def test_f4_too_generic_below_threshold():
    classifier = ContentFailureClassifier()
    result = _result(metrics={"precision": 0.2})
    modes = classifier.classify(result)
    assert ContentFailureMode.F4_TOO_GENERIC in modes


# =========================================================================
# F5 overlong
# =========================================================================


def test_f5_overlong_exceeds_word_budget():
    classifier = ContentFailureClassifier()
    case = _case(max_length=10, length_unit="words")
    result = _result(output="one two three four five six seven eight nine ten eleven", case=case)
    modes = classifier.classify(result)
    assert ContentFailureMode.F5_OVERLONG in modes


def test_f5_within_budget_clean():
    classifier = ContentFailureClassifier()
    case = _case(max_length=10, length_unit="words")
    result = _result(output="one two three", case=case)
    modes = classifier.classify(result)
    assert ContentFailureMode.F5_OVERLONG not in modes


def test_f5_char_unit():
    classifier = ContentFailureClassifier()
    case = _case(max_length=5, length_unit="chars")
    result = _result(output="toolongstring", case=case)
    modes = classifier.classify(result)
    assert ContentFailureMode.F5_OVERLONG in modes


# =========================================================================
# F6 poor structure
# =========================================================================


def test_f6_poor_structure_below_threshold():
    classifier = ContentFailureClassifier()
    result = _result(metrics={"format_compliance": 0.1})
    modes = classifier.classify(result)
    assert ContentFailureMode.F6_POOR_STRUCTURE in modes


# =========================================================================
# F7 false precision
# =========================================================================


def test_f7_false_precision_multiple_unsourced_precise_numbers():
    classifier = ContentFailureClassifier()
    result = _result(
        output="Our model achieves 94.37% accuracy, a 3.142x speedup, and 0.0023 latency."
    )
    modes = classifier.classify(result)
    assert ContentFailureMode.F7_FALSE_PRECISION in modes


def test_f7_suppressed_by_source_marker():
    classifier = ContentFailureClassifier()
    result = _result(
        output="Our model achieves 94.37% accuracy (source: benchmark-2026.md) and 3.142x speedup."
    )
    modes = classifier.classify(result)
    assert ContentFailureMode.F7_FALSE_PRECISION not in modes


def test_f7_single_hit_below_threshold():
    classifier = ContentFailureClassifier()
    result = _result(output="Latency was 2.34ms in this run.")
    modes = classifier.classify(result)
    assert ContentFailureMode.F7_FALSE_PRECISION not in modes


# =========================================================================
# F8 insufficient adversarial review
# =========================================================================


def test_f8_no_counter_markers_when_expected():
    classifier = ContentFailureClassifier()
    case = _case(expects_counterarguments=True)
    result = _result(
        output="This approach is excellent and will deliver great results across the board.",
        case=case,
    )
    modes = classifier.classify(result)
    assert ContentFailureMode.F8_INSUFFICIENT_ADVERSARIAL_REVIEW in modes


def test_f8_counter_markers_present_clean():
    classifier = ContentFailureClassifier()
    case = _case(expects_counterarguments=True)
    result = _result(
        output="The approach works well, however one key risk is rate limiting.",
        case=case,
    )
    modes = classifier.classify(result)
    assert ContentFailureMode.F8_INSUFFICIENT_ADVERSARIAL_REVIEW not in modes


def test_f8_not_expected_no_trigger():
    classifier = ContentFailureClassifier()
    result = _result(output="just a confident statement")
    modes = classifier.classify(result)
    assert ContentFailureMode.F8_INSUFFICIENT_ADVERSARIAL_REVIEW not in modes


# =========================================================================
# Overrides + composition
# =========================================================================


def test_explicit_override_wins():
    classifier = ContentFailureClassifier()
    case = _case(
        expected_content_failure_modes=["F1_missing_constraint", "F4_too_generic"],
        # Would otherwise trigger F3 too:
        required_phrases=[],
    )
    result = _result(metrics={"evidence_quality": 0.0}, case=case)
    modes = classifier.classify(result)
    assert modes == [
        ContentFailureMode.F1_MISSING_CONSTRAINT,
        ContentFailureMode.F4_TOO_GENERIC,
    ]


def test_override_invalid_value_dropped():
    classifier = ContentFailureClassifier()
    case = _case(expected_content_failure_modes=["F99_nonsense", "F4_too_generic"])
    result = _result(case=case)
    modes = classifier.classify(result)
    assert modes == [ContentFailureMode.F4_TOO_GENERIC]


def test_multiple_modes_on_single_result():
    classifier = ContentFailureClassifier()
    case = _case(
        required_phrases=["benchmark"],
        max_length=5,
        length_unit="words",
        expects_counterarguments=True,
    )
    result = _result(
        output="generic output with many words and no markers at all ever",
        metrics={"precision": 0.2, "evidence_quality": 0.3},
        case=case,
    )
    modes = classifier.classify(result)
    assert ContentFailureMode.F1_MISSING_CONSTRAINT in modes
    assert ContentFailureMode.F3_WEAK_EVIDENCE in modes
    assert ContentFailureMode.F4_TOO_GENERIC in modes
    assert ContentFailureMode.F5_OVERLONG in modes
    assert ContentFailureMode.F8_INSUFFICIENT_ADVERSARIAL_REVIEW in modes


# =========================================================================
# tag_content_failures
# =========================================================================


def test_tag_content_failures_mutates_metadata_and_counts():
    results = [
        _result(
            output="generic",
            metrics={"precision": 0.2},
        ),
        _result(
            output="another generic",
            metrics={"precision": 0.1, "evidence_quality": 0.2},
        ),
        _result(output="fine", metrics={"precision": 0.9, "evidence_quality": 0.9}),
    ]
    counts = tag_content_failures(results)
    assert counts.get("F4_too_generic") == 2
    assert counts.get("F3_weak_evidence") == 1
    assert results[0].metadata["content_failure_modes"] == ["F4_too_generic"]
    assert set(results[1].metadata["content_failure_modes"]) == {
        "F3_weak_evidence",
        "F4_too_generic",
    }
    assert results[2].metadata["content_failure_modes"] == []
