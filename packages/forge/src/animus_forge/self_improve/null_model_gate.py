"""Null-model gate for self-improvement changes.

A self-improvement change should produce a measurable, non-random improvement
on quantitative metrics of the codebase. Without a null-model comparison,
"the tests still pass" is a very weak signal — a no-op change, a whitespace
adjustment, or a random permutation of independent statements all satisfy it.

This module implements a pre-registered null-model gate (Prima Materia
methodology, H1-H19 pattern: see `~/projects/notes/topics/prima-materia/`):

  1. Compute a `MetricSnapshot` before the change.
  2. Compute a `MetricSnapshot` after applying the proposed change.
  3. Generate N "placebo" snapshots by applying trivial structural no-ops
     (whitespace adjustments, comment toggles, import reordering) of similar
     diff size to the same file set.
  4. Compare the observed delta to the null distribution by the L2 norm of
     the delta vector.
  5. Emit a verdict tier:
       - STRUCTURAL   (delta in ≥95th percentile of null) — real improvement
       - HEURISTIC    (delta in ≥75th percentile) — suggestive
       - NEUTRAL      (within noise)                 — no measurable effect
       - REGRESSION   (delta in ≤5th percentile, negative direction)

The gate does NOT reject neutral changes — it reports them. The caller
decides whether to gate on STRUCTURAL-only, STRUCTURAL+HEURISTIC, or just
to reject REGRESSIONs. Sensible default: require ≥HEURISTIC to proceed past
the sandbox stage, log everything to the audit log.

Wiring into `SelfImproveOrchestrator` is deliberately left to a follow-up:
the intended insertion point is between Stage 5 (sandbox testing) and
Stage 6 (apply approval) in `orchestrator.py::run()`. See module-level
function `gate_against_null()` for the canonical call signature.
"""

from __future__ import annotations

import logging
import math
import random
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NullModelVerdict(str, Enum):
    """Verdict tier for a null-model comparison.

    Matches the Prima Materia H1-H19 structural-analogy rubric:
      - STRUCTURAL: observed delta in ≥95th percentile of the null distribution
      - HEURISTIC:  observed delta in ≥75th percentile
      - NEUTRAL:    observed delta is within the null noise band
      - REGRESSION: observed delta is strongly negative vs the null
    """

    STRUCTURAL = "structural"
    HEURISTIC = "heuristic"
    NEUTRAL = "neutral"
    REGRESSION = "regression"


@dataclass(frozen=True)
class MetricSnapshot:
    """Quantitative snapshot of a codebase state.

    All fields are floats (ints coerced) so delta vectors are uniform.
    Fields with `None` values are excluded from delta calculations.

    Attributes:
        tests_total: Total number of test cases collected.
        tests_passing: Number of tests that pass.
        coverage_percent: Test coverage as a percentage (0–100).
        lint_errors: Number of lint errors (ruff check violations).
        type_coverage_percent: Percentage of annotated functions.
        docstring_coverage_percent: Percentage of functions with docstrings.
        lines_of_code: Non-blank, non-comment lines of code.
        cyclomatic_complexity_mean: Mean cyclomatic complexity per function.
        metadata: Free-form provenance data for the snapshot.
    """

    tests_total: float | None = None
    tests_passing: float | None = None
    coverage_percent: float | None = None
    lint_errors: float | None = None
    type_coverage_percent: float | None = None
    docstring_coverage_percent: float | None = None
    lines_of_code: float | None = None
    cyclomatic_complexity_mean: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def metric_keys(self) -> list[str]:
        """Return the names of all non-None quantitative metrics."""
        return [
            name
            for name in (
                "tests_total",
                "tests_passing",
                "coverage_percent",
                "lint_errors",
                "type_coverage_percent",
                "docstring_coverage_percent",
                "lines_of_code",
                "cyclomatic_complexity_mean",
            )
            if getattr(self, name) is not None
        ]

    def delta_to(self, other: MetricSnapshot) -> dict[str, float]:
        """Return component-wise delta `other − self` for shared non-None keys.

        Metrics for which either snapshot is None are omitted from the result.
        """
        shared = set(self.metric_keys()) & set(other.metric_keys())
        return {k: float(getattr(other, k)) - float(getattr(self, k)) for k in shared}


@dataclass
class NullModelResult:
    """Result of a null-model gate comparison.

    Attributes:
        verdict: Categorical verdict tier.
        observed_delta: Component-wise metric deltas from baseline to proposed.
        observed_score: Scalar score of the observed delta (L2 norm with sign).
        null_scores: Scalar scores from each placebo sample.
        percentile: Rank of observed_score in null_scores (0–100).
        null_mean: Mean of the null distribution.
        null_std: Standard deviation of the null distribution.
        n_null_samples: Number of null samples used.
        rationale: Human-readable explanation of the verdict.
    """

    verdict: NullModelVerdict
    observed_delta: dict[str, float]
    observed_score: float
    null_scores: list[float]
    percentile: float
    null_mean: float
    null_std: float
    n_null_samples: int
    rationale: str

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a serializable representation for the audit log."""
        return {
            "verdict": self.verdict.value,
            "observed_score": round(self.observed_score, 6),
            "percentile": round(self.percentile, 2),
            "null_mean": round(self.null_mean, 6),
            "null_std": round(self.null_std, 6),
            "n_null_samples": self.n_null_samples,
            "observed_delta": {k: round(v, 6) for k, v in self.observed_delta.items()},
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Metric direction weighting
# ---------------------------------------------------------------------------

# Sign map: +1 means "larger is better", -1 means "smaller is better".
# Used to convert raw delta components into a directional improvement score.
_METRIC_DIRECTION: dict[str, int] = {
    "tests_total": +1,  # more tests is better
    "tests_passing": +1,  # more passing tests is better
    "coverage_percent": +1,
    "lint_errors": -1,  # fewer lint errors is better
    "type_coverage_percent": +1,
    "docstring_coverage_percent": +1,
    "lines_of_code": 0,  # neutral — don't reward either direction
    "cyclomatic_complexity_mean": -1,
}


def _scale_for(metric: str) -> float:
    """Per-metric scale factor for normalizing delta components.

    Without this, a 1-percentage-point coverage change competes on equal
    footing with a 1-line LOC change, which is wrong. Scales are hand-tuned
    to reflect typical noise in each metric.
    """
    return {
        "tests_total": 5.0,
        "tests_passing": 5.0,
        "coverage_percent": 1.0,
        "lint_errors": 1.0,
        "type_coverage_percent": 1.0,
        "docstring_coverage_percent": 1.0,
        "lines_of_code": 20.0,
        "cyclomatic_complexity_mean": 0.2,
    }.get(metric, 1.0)


def score_delta(delta: dict[str, float]) -> float:
    """Reduce a component-wise delta to a single scalar improvement score.

    The score is the signed L2 norm of the directional, scale-normalized
    delta vector. Positive = improvement, negative = regression.

    Args:
        delta: Mapping metric_name → raw numeric change.

    Returns:
        Scalar improvement score (can be negative).
    """
    signed_squares: list[float] = []
    signed_sum = 0.0
    for metric, raw in delta.items():
        direction = _METRIC_DIRECTION.get(metric, 0)
        if direction == 0:
            continue
        scaled = (raw * direction) / _scale_for(metric)
        signed_sum += scaled
        signed_squares.append(scaled * scaled)

    if not signed_squares:
        return 0.0
    magnitude = math.sqrt(sum(signed_squares))
    # Preserve the sign of the dominant direction (sum over components)
    return magnitude if signed_sum >= 0 else -magnitude


# ---------------------------------------------------------------------------
# Placebo generator
# ---------------------------------------------------------------------------


def _default_placebo_perturbation(source: str, rng: random.Random) -> str:
    """Apply a trivial, semantically neutral perturbation to source text.

    These perturbations should not measurably change any of the tracked
    metrics: the resulting null distribution centers on zero, and the
    observed change is compared against that distribution.

    Perturbation kinds:
      - Insert a trailing-whitespace line
      - Toggle a blank line between two non-adjacent non-blank lines
      - Insert a no-op comment line
      - Reorder two adjacent blank lines (always a no-op)
    """
    lines = source.splitlines()
    if len(lines) < 4:
        return source

    kind = rng.choice(["blank", "comment", "trailing_ws", "noop_pass"])
    idx = rng.randrange(2, max(3, len(lines) - 1))

    if kind == "blank":
        lines.insert(idx, "")
    elif kind == "comment":
        lines.insert(idx, "# null-model placebo perturbation")
    elif kind == "trailing_ws":
        lines[idx] = lines[idx] + " "
    elif kind == "noop_pass":
        # Insert only at a line that's a standalone statement inside a function
        # Fallback: trailing whitespace if we can't find one
        inserted = False
        for pos in range(idx, len(lines) - 1):
            stripped = lines[pos].strip()
            if stripped and not stripped.startswith("#") and re.match(r"^\s+", lines[pos]):
                indent = re.match(r"^(\s+)", lines[pos]).group(1)
                lines.insert(pos + 1, f"{indent}pass  # placebo")
                inserted = True
                break
        if not inserted:
            lines[idx] = lines[idx] + " "
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def generate_placebo_changes(
    original_files: dict[str, str],
    n: int,
    seed: int = 0,
    perturbation: Callable[[str, random.Random], str] | None = None,
) -> list[dict[str, str]]:
    """Generate n placebo change-sets of the same file set.

    Args:
        original_files: Mapping path → original file content.
        n: Number of placebo samples to generate.
        seed: Base random seed for reproducibility.
        perturbation: Override the placebo generator (for testing).

    Returns:
        List of dicts, each mapping path → perturbed file content.
    """
    perturbation = perturbation or _default_placebo_perturbation
    samples: list[dict[str, str]] = []
    for i in range(n):
        rng = random.Random(seed + i)
        sample: dict[str, str] = {}
        for path, content in original_files.items():
            sample[path] = perturbation(content, rng)
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Core gate logic
# ---------------------------------------------------------------------------


def classify_verdict(
    observed_score: float,
    null_scores: list[float],
) -> tuple[NullModelVerdict, float, str]:
    """Classify an observed score against a null distribution.

    Args:
        observed_score: Scalar improvement score of the actual change.
        null_scores: Scalar scores from placebo samples.

    Returns:
        Tuple of (verdict, percentile, rationale_text).
    """
    if not null_scores:
        return (
            NullModelVerdict.NEUTRAL,
            50.0,
            "No null samples available — cannot classify.",
        )

    strictly_below = sum(1 for s in null_scores if s < observed_score)
    equal = sum(1 for s in null_scores if s == observed_score)
    percentile = 100.0 * (strictly_below + 0.5 * equal) / len(null_scores)

    if observed_score < 0 and percentile <= 5.0:
        verdict = NullModelVerdict.REGRESSION
        rationale = (
            f"Observed score {observed_score:.4f} is in the bottom 5% of the "
            f"null distribution (percentile {percentile:.1f}); the change "
            f"appears to measurably worsen the tracked metrics."
        )
    elif percentile >= 95.0:
        verdict = NullModelVerdict.STRUCTURAL
        rationale = (
            f"Observed score {observed_score:.4f} is in the top 5% of the "
            f"null distribution (percentile {percentile:.1f}); the change "
            f"measurably improves the tracked metrics beyond placebo noise."
        )
    elif percentile >= 75.0:
        verdict = NullModelVerdict.HEURISTIC
        rationale = (
            f"Observed score {observed_score:.4f} is in the top 25% of the "
            f"null distribution (percentile {percentile:.1f}); suggestive "
            f"improvement, but not clearly above placebo noise."
        )
    else:
        verdict = NullModelVerdict.NEUTRAL
        rationale = (
            f"Observed score {observed_score:.4f} is within placebo noise "
            f"(percentile {percentile:.1f}); no measurable effect on the "
            f"tracked metrics."
        )
    return verdict, percentile, rationale


def gate_against_null(
    baseline: MetricSnapshot,
    proposed: MetricSnapshot,
    null_samples: list[MetricSnapshot],
) -> NullModelResult:
    """Apply the null-model gate to a proposed self-improvement change.

    Computes the observed delta (baseline → proposed), computes placebo
    deltas (baseline → each null sample), and compares them by signed L2
    score. Returns a structured verdict + rationale suitable for the
    audit log.

    Args:
        baseline: Metric snapshot before the change.
        proposed: Metric snapshot after the change.
        null_samples: Snapshots from the placebo distribution (baseline +
            trivial perturbation). At least 10 recommended, 100+ preferred.

    Returns:
        NullModelResult with verdict tier, scores, and rationale.
    """
    observed_delta = baseline.delta_to(proposed)
    observed_score = score_delta(observed_delta)

    null_scores: list[float] = []
    for sample in null_samples:
        sample_delta = baseline.delta_to(sample)
        null_scores.append(score_delta(sample_delta))

    verdict, percentile, rationale = classify_verdict(observed_score, null_scores)

    if null_scores:
        null_mean = statistics.fmean(null_scores)
        null_std = statistics.pstdev(null_scores) if len(null_scores) > 1 else 0.0
    else:
        null_mean = 0.0
        null_std = 0.0

    return NullModelResult(
        verdict=verdict,
        observed_delta=observed_delta,
        observed_score=observed_score,
        null_scores=null_scores,
        percentile=percentile,
        null_mean=null_mean,
        null_std=null_std,
        n_null_samples=len(null_scores),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# High-level wrapper for orchestrator wiring (future use)
# ---------------------------------------------------------------------------


@dataclass
class NullModelGateConfig:
    """Configuration for the orchestrator-integrated null-model gate.

    Attributes:
        enabled: Master switch for the gate.
        n_null_samples: Number of placebo samples to generate per evaluation.
        min_verdict: Minimum verdict tier required to proceed past the gate.
        seed: Random seed for placebo generation (deterministic runs).
        block_regressions: If True, a REGRESSION verdict short-circuits
            to rejection even in permissive modes.
    """

    enabled: bool = True
    n_null_samples: int = 30
    min_verdict: NullModelVerdict = NullModelVerdict.HEURISTIC
    seed: int = 42
    block_regressions: bool = True

    def allows(self, verdict: NullModelVerdict) -> bool:
        """Return True if the verdict satisfies this config's gate."""
        if not self.enabled:
            return True
        if verdict == NullModelVerdict.REGRESSION:
            return not self.block_regressions
        tier_rank = {
            NullModelVerdict.NEUTRAL: 0,
            NullModelVerdict.HEURISTIC: 1,
            NullModelVerdict.STRUCTURAL: 2,
            NullModelVerdict.REGRESSION: -1,
        }
        return tier_rank[verdict] >= tier_rank[self.min_verdict]


__all__ = [
    "NullModelVerdict",
    "MetricSnapshot",
    "NullModelResult",
    "NullModelGateConfig",
    "score_delta",
    "generate_placebo_changes",
    "classify_verdict",
    "gate_against_null",
]
