"""Failure taxonomy: classify eval failures into structured buckets.

PASSED/FAILED/ERROR/SKIPPED is enough for a test runner but useless for
triage. When a suite regresses, the first question is "what *kind* of
failures spiked" — schema drift? budget exhaustion? judge disagreement?

This module adds an orthogonal axis on top of EvalStatus: a FailureMode
enum attached to each EvalResult, populated by rule-based classification
over output content and error strings.
"""

from __future__ import annotations

import re
from enum import Enum

from .base import EvalResult, EvalStatus


class FailureMode(str, Enum):
    """Structured failure buckets for eval triage.

    Values chosen so regressions tell you *where* to look:
    - SCHEMA_DRIFT: output shape changed (e.g. JSON parse failed, missing field)
    - HALLUCINATION: factually wrong despite producing confident output
    - WRONG_ANSWER: plain incorrect, no other bucket fits
    - REFUSED: model declined to answer
    - OVER_BUDGET: BudgetManager or token ceiling blocked the call
    - TIMEOUT: wall-clock exceeded
    - CONTRACT_VIOLATION: a required output contract (schema / citation) failed
    - JUDGE_DISAGREEMENT: multi-sample LLM judge returned inconsistent scores
    - FLAKY: passed on retry, failed on first attempt
    - PROVIDER_ERROR: upstream API failure (rate limit, 5xx, auth)
    - UNKNOWN: none of the above — investigate, then add a rule
    """

    SCHEMA_DRIFT = "schema_drift"
    HALLUCINATION = "hallucination"
    WRONG_ANSWER = "wrong_answer"
    REFUSED = "refused"
    OVER_BUDGET = "over_budget"
    TIMEOUT = "timeout"
    CONTRACT_VIOLATION = "contract_violation"
    JUDGE_DISAGREEMENT = "judge_disagreement"
    FLAKY = "flaky"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


REFUSAL_PATTERNS = (
    r"\bi (can(?:'|no)t|cannot|won't|will not|am (?:not able|unable))\b",
    r"\bi'?m (?:not able|unable|sorry)\b",
    r"\b(?:as an ai|i (?:do not|don'?t) have (?:the )?(?:ability|access|information))\b",
)
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

TIMEOUT_TOKENS = ("timeout", "timed out", "deadlineexceeded", "readtimeout")
BUDGET_TOKENS = ("budget", "budget exceeded", "over budget", "insufficient funds")
RATE_LIMIT_TOKENS = ("rate limit", "429", "too many requests", "quota exceeded")


class FailureClassifier:
    """Rule-based classifier. Subclass to add project-specific rules.

    Usage:
        classifier = FailureClassifier()
        for result in suite_result.results:
            mode = classifier.classify(result)
            if mode:
                result.failure_mode = mode.value
    """

    def __init__(
        self,
        *,
        judge_variance_threshold: float = 0.2,
        contract_dims: tuple[str, ...] = ("schema_valid", "contract"),
    ):
        """Initialize classifier.

        Args:
            judge_variance_threshold: Max acceptable std-dev across
                multi-sample judge scores before flagging JUDGE_DISAGREEMENT.
                Assumes case.metadata["judge_samples"] is a list of floats.
            contract_dims: Metric/dim names that represent hard contracts.
                Any of these scoring 0 triggers CONTRACT_VIOLATION.
        """
        self.judge_variance_threshold = judge_variance_threshold
        self.contract_dims = tuple(contract_dims)

    def classify(self, result: EvalResult) -> FailureMode | None:
        """Assign a failure mode to a result.

        Returns None if the result passed. Otherwise returns the
        highest-priority matching FailureMode.
        """
        if result.status == EvalStatus.PASSED:
            return None
        if result.status == EvalStatus.SKIPPED:
            return None

        # Explicit case metadata override wins — authors can pre-label
        # known failure types to sanity-check the classifier.
        labeled = result.case.metadata.get("expected_failure_mode")
        if labeled:
            try:
                return FailureMode(labeled)
            except ValueError:
                pass

        # ERROR status — inspect the error string
        if result.status == EvalStatus.ERROR and result.error:
            mode = self._classify_error(result.error.lower())
            if mode:
                return mode
            return FailureMode.PROVIDER_ERROR

        # FAILED status — inspect metrics + output
        mode = self._classify_failed(result)
        if mode:
            return mode

        return FailureMode.UNKNOWN

    # -----------------------------------------------------------------
    # Rules
    # -----------------------------------------------------------------

    def _classify_error(self, err: str) -> FailureMode | None:
        if any(t in err for t in TIMEOUT_TOKENS):
            return FailureMode.TIMEOUT
        if any(t in err for t in BUDGET_TOKENS):
            return FailureMode.OVER_BUDGET
        if any(t in err for t in RATE_LIMIT_TOKENS):
            return FailureMode.PROVIDER_ERROR
        return None

    def _classify_failed(self, result: EvalResult) -> FailureMode | None:
        # Contract violation: a named contract dim scored 0
        for dim_name in self.contract_dims:
            if result.metrics.get(dim_name) == 0.0:
                return FailureMode.CONTRACT_VIOLATION

        # Judge disagreement: multi-sample judge variance
        samples = result.case.metadata.get("judge_samples")
        if isinstance(samples, list) and len(samples) >= 2:
            try:
                mean = sum(samples) / len(samples)
                var = sum((s - mean) ** 2 for s in samples) / len(samples)
                if var**0.5 > self.judge_variance_threshold:
                    return FailureMode.JUDGE_DISAGREEMENT
            except (TypeError, ValueError):
                pass

        # Refusal: model declined in text
        output_text = str(result.output or "")
        if output_text and _REFUSAL_RE.search(output_text):
            return FailureMode.REFUSED

        # Schema drift: case expected JSON / structured output but schema_valid scored low
        if result.case.metadata.get("expects_json"):
            schema_score = result.metrics.get("schema_valid")
            if schema_score is not None and schema_score < 0.5:
                return FailureMode.SCHEMA_DRIFT

        # Hallucination: factuality metric flagged it
        factuality_score = result.metrics.get("factuality")
        if factuality_score is not None and factuality_score < 0.5:
            return FailureMode.HALLUCINATION

        # Flaky: case metadata says it passed on retry
        if result.case.metadata.get("flaky") or result.metadata.get("flaky"):
            return FailureMode.FLAKY

        # Plain wrong
        return FailureMode.WRONG_ANSWER


def tag_results(
    results: list[EvalResult],
    classifier: FailureClassifier | None = None,
) -> dict[str, int]:
    """Classify a batch of results in-place, return bucket counts.

    Mutates each ``EvalResult.failure_mode`` to the classified value.
    Returns a dict of {failure_mode_value: count} for passed+failed.
    """
    classifier = classifier or FailureClassifier()
    counts: dict[str, int] = {}
    for result in results:
        mode = classifier.classify(result)
        result.failure_mode = mode.value if mode else None
        key = mode.value if mode else "passed"
        counts[key] = counts.get(key, 0) + 1
    return counts
