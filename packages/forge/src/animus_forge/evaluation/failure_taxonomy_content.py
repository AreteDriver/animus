"""Content-quality failure taxonomy (F1-F8).

Orthogonal to ``failure_taxonomy.py``. The technical classifier tags
*why the pipeline failed* (schema_drift, timeout, refused, etc). This
classifier tags *why the output is weak* even when the pipeline
succeeded — missing constraints, generic advice, false precision,
insufficient adversarial review.

An output can simultaneously carry one technical failure_mode AND
multiple content failure modes. They answer different questions:

    failure_mode (technical):           "what broke"
    content_failure_modes (this file):  "what's wrong with the output"

Tags are stored as a list in ``result.metadata["content_failure_modes"]``
so a single output can be tagged with multiple modes (e.g. F4 too_generic
AND F5 overlong AND F8 insufficient_adversarial_review).
"""

from __future__ import annotations

import re
from enum import Enum

from .base import EvalResult


class ContentFailureMode(str, Enum):
    """F1-F8 content-quality failure taxonomy.

    Inspired by ARETE's prompt-engineering failure catalog — tracks
    content weaknesses that rubric scores alone don't surface.
    """

    F1_MISSING_CONSTRAINT = "F1_missing_constraint"
    F2_WRONG_ASSUMPTION = "F2_wrong_assumption"
    F3_WEAK_EVIDENCE = "F3_weak_evidence"
    F4_TOO_GENERIC = "F4_too_generic"
    F5_OVERLONG = "F5_overlong"
    F6_POOR_STRUCTURE = "F6_poor_structure"
    F7_FALSE_PRECISION = "F7_false_precision"
    F8_INSUFFICIENT_ADVERSARIAL_REVIEW = "F8_insufficient_adversarial_review"


# Markers that signal adversarial/counter-argument presence
COUNTER_MARKERS = (
    "however",
    "on the other hand",
    "counter",
    "risk",
    "caveat",
    "limitation",
    "weakness",
    "tradeoff",
    "downside",
    "but note",
    "one concern",
)

# Markers that indicate citation / sourcing (suppresses F7 false positives)
SOURCE_MARKERS = (
    "source:",
    "per ",
    "via ",
    "according to",
    "[",  # markdown citation
    "(see ",
    "ref:",
    "cf.",
)

# Regex: decimal numbers with 2+ decimal places, optionally followed by
# %, x, × — the "94.37% accuracy" / "3.142x faster" pattern
_FALSE_PRECISION_RE = re.compile(r"\b\d+\.\d{2,}(?:[%x×])?\b")


class ContentFailureClassifier:
    """Rule-based classifier for F1-F8 content-quality failures.

    Many rules depend on rubric dim scores being present in
    ``result.metrics`` — pair with a rubric that includes those dims
    (e.g. ``personal-quality.yaml``). Rules that can't trigger
    silently skip rather than raising.

    Subclass and override individual rule methods to add project-specific
    detection. Rules are composed via ``classify()``; each returns a list
    of modes to append.
    """

    def __init__(
        self,
        *,
        weak_threshold: float = 0.5,
        false_precision_min_hits: int = 2,
        counter_markers: tuple[str, ...] = COUNTER_MARKERS,
        source_markers: tuple[str, ...] = SOURCE_MARKERS,
    ):
        """Initialize classifier.

        Args:
            weak_threshold: Per-dim score below which F3/F4/F6 fire
            false_precision_min_hits: How many suspicious numbers before F7 fires.
                Raising reduces false positives on legitimately-quantified output.
            counter_markers: Phrases whose presence suppresses F8
            source_markers: Phrases whose presence suppresses F7
        """
        self.weak_threshold = weak_threshold
        self.false_precision_min_hits = false_precision_min_hits
        self.counter_markers = tuple(counter_markers)
        self.source_markers = tuple(source_markers)

    def classify(self, result: EvalResult) -> list[ContentFailureMode]:
        """Return all content failure modes that apply to a result.

        Explicit case metadata override: if
        ``case.metadata["expected_content_failure_modes"]`` is a list of
        mode values, those become the answer (bypasses rule evaluation).
        Useful for fixture cases that pre-label expected failures to
        sanity-check the classifier itself.
        """
        overrides = result.case.metadata.get("expected_content_failure_modes")
        if overrides:
            return self._parse_overrides(overrides)

        modes: list[ContentFailureMode] = []
        output_text = str(result.output or "")
        output_lower = output_text.lower()

        for rule in (
            self._rule_f1_missing_constraint,
            self._rule_f2_wrong_assumption,
            self._rule_f3_weak_evidence,
            self._rule_f4_too_generic,
            self._rule_f5_overlong,
            self._rule_f6_poor_structure,
            self._rule_f7_false_precision,
            self._rule_f8_insufficient_adversarial_review,
        ):
            mode = rule(result, output_text, output_lower)
            if mode is not None:
                modes.append(mode)

        return modes

    # -----------------------------------------------------------------
    # Rules
    # -----------------------------------------------------------------

    def _rule_f1_missing_constraint(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        required = result.case.metadata.get("required_phrases") or []
        if not required:
            return None
        missing = [p for p in required if p.lower() not in output_lower]
        return ContentFailureMode.F1_MISSING_CONSTRAINT if missing else None

    def _rule_f2_wrong_assumption(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        errors = result.case.metadata.get("assumption_errors") or []
        if not errors:
            return None
        if any(err.lower() in output_lower for err in errors):
            return ContentFailureMode.F2_WRONG_ASSUMPTION
        return None

    def _rule_f3_weak_evidence(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        score = result.metrics.get("evidence_quality")
        if score is not None and score < self.weak_threshold:
            return ContentFailureMode.F3_WEAK_EVIDENCE
        return None

    def _rule_f4_too_generic(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        score = result.metrics.get("precision")
        if score is not None and score < self.weak_threshold:
            return ContentFailureMode.F4_TOO_GENERIC
        return None

    def _rule_f5_overlong(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        max_len = result.case.metadata.get("max_length")
        if max_len is None:
            return None
        unit = result.case.metadata.get("length_unit", "words")
        length = len(output_text.split()) if unit == "words" else len(output_text)
        return ContentFailureMode.F5_OVERLONG if length > max_len else None

    def _rule_f6_poor_structure(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        score = result.metrics.get("format_compliance")
        if score is not None and score < self.weak_threshold:
            return ContentFailureMode.F6_POOR_STRUCTURE
        return None

    def _rule_f7_false_precision(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        matches = _FALSE_PRECISION_RE.findall(output_text)
        if len(matches) < self.false_precision_min_hits:
            return None
        if any(marker in output_lower for marker in self.source_markers):
            return None
        return ContentFailureMode.F7_FALSE_PRECISION

    def _rule_f8_insufficient_adversarial_review(
        self, result: EvalResult, output_text: str, output_lower: str
    ) -> ContentFailureMode | None:
        if not result.case.metadata.get("expects_counterarguments"):
            return None
        if any(m in output_lower for m in self.counter_markers):
            return None
        return ContentFailureMode.F8_INSUFFICIENT_ADVERSARIAL_REVIEW

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _parse_overrides(self, overrides: list[str]) -> list[ContentFailureMode]:
        valid_values = {m.value for m in ContentFailureMode}
        return [ContentFailureMode(m) for m in overrides if m in valid_values]


def tag_content_failures(
    results: list[EvalResult],
    classifier: ContentFailureClassifier | None = None,
) -> dict[str, int]:
    """Classify content failures across a batch, return bucket counts.

    Mutates ``result.metadata["content_failure_modes"]`` on each result
    to a list of mode values. Returns {mode_value: count} summary.
    A single output contributes to multiple buckets when it carries
    multiple content failures — counts therefore may exceed result count.
    """
    classifier = classifier or ContentFailureClassifier()
    counts: dict[str, int] = {}
    for result in results:
        modes = classifier.classify(result)
        result.metadata["content_failure_modes"] = [m.value for m in modes]
        for mode in modes:
            counts[mode.value] = counts.get(mode.value, 0) + 1
    return counts
