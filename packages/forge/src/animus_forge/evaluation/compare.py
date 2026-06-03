"""A/B comparison for eval runs with bootstrap confidence intervals.

Given two stored run IDs, produces a ComparisonReport with:
    - Composite score delta + 95% bootstrap CI on the delta
    - Pass-rate delta
    - Cost and latency delta
    - Failure-bucket delta (which taxonomy buckets moved)
    - Regression flags: human-readable warnings when something
      crosses a configurable threshold

Bootstrap CI uses paired resampling when the two runs have the same
case count (typical A/B: same suite, two configs) and falls back to
independent resampling otherwise. 1000 resamples is the default —
enough to stabilize the ~95% quantile without noticeable latency on
suites up to a few hundred cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .store import EvalStore


@dataclass
class RunSummary:
    """Flattened view of a stored eval run for comparison."""

    run_id: str
    suite_name: str
    model: str | None
    rubric_name: str | None
    rubric_version: str | None
    prompt_version: str | None
    avg_score: float
    pass_rate: float
    total_cost_usd: float
    duration_ms: float
    total_cases: int
    case_scores: list[float]
    failure_buckets: dict[str, int]
    content_failure_buckets: dict[str, int] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Result of comparing two eval runs.

    ``significant`` is True when the 95% bootstrap CI on the score
    delta excludes zero — i.e. the observed difference is unlikely to
    be noise given this suite size.
    """

    baseline: RunSummary
    candidate: RunSummary
    score_delta: float
    score_ci_95: tuple[float, float]
    pass_rate_delta: float
    cost_delta_usd: float
    latency_delta_ms: float
    failure_delta: dict[str, int]
    content_failure_delta: dict[str, int] = field(default_factory=dict)
    regression_flags: list[str] = field(default_factory=list)
    bootstrap_n: int = 1000

    #: Smallest score delta (on the 0-1 scale) considered worth caring about.
    #: A "not significant" verdict whose CI is wider than this can't rule out a
    #: meaningful effect — it is underpowered, not evidence of equivalence.
    MEANINGFUL_EFFECT: ClassVar[float] = 0.05

    @property
    def significant(self) -> bool:
        lo, hi = self.score_ci_95
        return lo > 0 or hi < 0

    @property
    def direction(self) -> str:
        if self.score_delta > 0:
            return "improvement"
        if self.score_delta < 0:
            return "regression"
        return "no-change"

    @property
    def ci_halfwidth(self) -> float:
        lo, hi = self.score_ci_95
        return (hi - lo) / 2

    @property
    def equivalent(self) -> bool:
        """True when the WHOLE 95% CI lies within ±MEANINGFUL_EFFECT (C10).

        This is a bounds-based (TOST-style) equivalence test, not a half-width
        one. Half-width is wrong for an off-center interval: CI [0.01, 0.09]
        has half-width 0.04 < 0.05 yet fails to rule out a +0.09 effect, so
        calling it "equivalent" would be false. Equivalence requires both
        bounds inside the indifference zone.
        """
        lo, hi = self.score_ci_95
        return (
            (not self.significant) and lo > -self.MEANINGFUL_EFFECT and hi < self.MEANINGFUL_EFFECT
        )

    @property
    def underpowered(self) -> bool:
        """True when 'not significant' really means 'too few cases to tell'.

        A non-significant result that is also not equivalent — i.e. the 95% CI
        extends beyond ±``MEANINGFUL_EFFECT`` on at least one side — can't
        exclude a difference worth caring about: absence of evidence, not
        evidence of absence (roadmap B6, corrected bounds-based in C10).
        """
        return (not self.significant) and not self.equivalent

    @property
    def power_note(self) -> str:
        """Human-readable power/sample-size advisory; empty when significant."""
        if self.significant:
            return ""
        lo, hi = self.score_ci_95
        n = min(self.baseline.total_cases, self.candidate.total_cases)
        if self.underpowered:
            suggest = max(n * 4, 4)
            return (
                f"Underpowered: the 95% CI [{lo:+.3f}, {hi:+.3f}] is wider than "
                f"±{self.MEANINGFUL_EFFECT:.2f}, so n={n} cases is too few to "
                f"rule out a meaningful difference — absence of evidence, not "
                f"evidence of equivalence. ~{suggest} cases would roughly halve "
                f"the interval."
            )
        # Equivalent: both bounds inside the indifference zone. Report the
        # ACTUAL tightest symmetric envelope the CI fits in, derived from the
        # bounds — not the fixed threshold (C10).
        envelope = max(abs(lo), abs(hi))
        return (
            f"Not significant and tightly bounded (95% CI [{lo:+.3f}, {hi:+.3f}] "
            f"within ±{envelope:.3f} ≤ ±{self.MEANINGFUL_EFFECT:.2f}): effectively "
            f"equivalent on this suite (n={n})."
        )


# =========================================================================
# Bootstrap
# =========================================================================


def bootstrap_ci_delta(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap CI on mean(b) - mean(a).

    Paired resampling when lengths match (same suite, same order);
    otherwise independent resampling.
    """
    if not scores_a or not scores_b:
        return (0.0, 0.0)

    rng = random.Random(seed)
    paired = len(scores_a) == len(scores_b)
    deltas: list[float] = []

    for _ in range(n):
        if paired:
            idx = [rng.randrange(len(scores_a)) for _ in range(len(scores_a))]
            mean_a = sum(scores_a[i] for i in idx) / len(idx)
            mean_b = sum(scores_b[i] for i in idx) / len(idx)
        else:
            sample_a = [rng.choice(scores_a) for _ in range(len(scores_a))]
            sample_b = [rng.choice(scores_b) for _ in range(len(scores_b))]
            mean_a = sum(sample_a) / len(sample_a)
            mean_b = sum(sample_b) / len(sample_b)
        deltas.append(mean_b - mean_a)

    deltas.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * n))
    hi_idx = min(n - 1, int((1 - alpha) * n) - 1)
    return (deltas[lo_idx], deltas[hi_idx])


# =========================================================================
# Summary
# =========================================================================


def load_run_summary(store: EvalStore, run_id: str) -> RunSummary:
    """Load a stored run and flatten into a RunSummary."""
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    case_rows = run.get("case_results", [])
    case_scores = [float(cr.get("score", 0.0)) for cr in case_rows]

    buckets: dict[str, int] = {}
    content_buckets: dict[str, int] = {}
    for cr in case_rows:
        key = cr.get("failure_mode")
        if not key:
            key = "passed" if cr.get("status") == "passed" else "unclassified"
        buckets[key] = buckets.get(key, 0) + 1

        # Content failures live as a list (a case can carry multiple)
        for content_mode in cr.get("content_failure_modes", []) or []:
            content_buckets[content_mode] = content_buckets.get(content_mode, 0) + 1

    return RunSummary(
        run_id=str(run.get("id")),
        suite_name=str(run.get("suite_name")),
        model=run.get("model"),
        rubric_name=run.get("rubric_name"),
        rubric_version=run.get("rubric_version"),
        prompt_version=run.get("prompt_version"),
        avg_score=float(run.get("avg_score", 0.0) or 0.0),
        pass_rate=float(run.get("pass_rate", 0.0) or 0.0),
        total_cost_usd=float(run.get("total_cost_usd", 0.0) or 0.0),
        duration_ms=float(run.get("duration_ms", 0.0) or 0.0),
        total_cases=int(run.get("total_cases", 0) or 0),
        case_scores=case_scores,
        failure_buckets=buckets,
        content_failure_buckets=content_buckets,
    )


# =========================================================================
# Compare
# =========================================================================


def compare_runs(
    baseline_id: str,
    candidate_id: str,
    *,
    store: EvalStore | None = None,
    bootstrap_n: int = 1000,
    regression_threshold: float = 0.02,
    bucket_spike_threshold: float = 0.20,
    seed: int | None = None,
) -> ComparisonReport:
    """Compare two eval runs and flag regressions.

    Args:
        baseline_id: Run ID of the baseline (the "before")
        candidate_id: Run ID of the candidate (the "after")
        store: EvalStore instance; resolved via get_eval_store() if None
        bootstrap_n: Resamples for the CI — more = tighter, slower
        regression_threshold: Composite score drop that flags regression
        bucket_spike_threshold: Failure-bucket count increase, as fraction
            of baseline total_cases, that flags a bucket spike
        seed: Optional RNG seed for reproducible bootstrap
    """
    if store is None:
        from .store import get_eval_store

        store = get_eval_store()

    base = load_run_summary(store, baseline_id)
    cand = load_run_summary(store, candidate_id)

    if base.suite_name != cand.suite_name:
        raise ValueError(
            f"Suite mismatch: baseline ran '{base.suite_name}', "
            f"candidate ran '{cand.suite_name}'. Comparison requires same suite."
        )

    ci = bootstrap_ci_delta(base.case_scores, cand.case_scores, n=bootstrap_n, seed=seed)

    failure_delta: dict[str, int] = {}
    keys = set(base.failure_buckets) | set(cand.failure_buckets)
    for key in keys:
        delta = cand.failure_buckets.get(key, 0) - base.failure_buckets.get(key, 0)
        if delta != 0:
            failure_delta[key] = delta

    content_failure_delta: dict[str, int] = {}
    content_keys = set(base.content_failure_buckets) | set(cand.content_failure_buckets)
    for key in content_keys:
        delta = cand.content_failure_buckets.get(key, 0) - base.content_failure_buckets.get(key, 0)
        if delta != 0:
            content_failure_delta[key] = delta

    flags: list[str] = []
    score_delta = cand.avg_score - base.avg_score
    if score_delta < -regression_threshold:
        flags.append(f"composite_regression:{score_delta:+.3f}")
    if ci[1] < 0:
        flags.append("ci_excludes_zero:regression")
    elif ci[0] > 0:
        flags.append("ci_excludes_zero:improvement")

    spike_floor = max(1, int(bucket_spike_threshold * max(base.total_cases, 1)))
    for bucket, delta in failure_delta.items():
        if bucket == "passed":
            continue
        if delta >= spike_floor:
            flags.append(f"bucket_spike:{bucket}:+{delta}")

    for bucket, delta in content_failure_delta.items():
        if delta >= spike_floor:
            flags.append(f"content_spike:{bucket}:+{delta}")

    if cand.total_cost_usd > base.total_cost_usd * 1.5 and base.total_cost_usd > 0:
        flags.append(f"cost_spike:{cand.total_cost_usd - base.total_cost_usd:+.4f}usd")

    return ComparisonReport(
        baseline=base,
        candidate=cand,
        score_delta=score_delta,
        score_ci_95=ci,
        pass_rate_delta=cand.pass_rate - base.pass_rate,
        cost_delta_usd=cand.total_cost_usd - base.total_cost_usd,
        latency_delta_ms=cand.duration_ms - base.duration_ms,
        failure_delta=failure_delta,
        content_failure_delta=content_failure_delta,
        regression_flags=flags,
        bootstrap_n=bootstrap_n,
    )


# =========================================================================
# Run ID resolution helpers (for CLI ergonomics)
# =========================================================================


def resolve_run_id(
    store: EvalStore,
    selector: str,
    *,
    suite_name: str | None = None,
) -> str:
    """Resolve a user-friendly selector to a run ID.

    Selectors:
        - "last" / "latest": most recent run (optionally for suite)
        - "prev": second-most-recent
        - <full uuid>: returned as-is
        - <8-char prefix>: resolved via prefix match in recent runs
    """
    if selector in {"last", "latest"}:
        runs = store.query_runs(suite_name=suite_name, limit=1)
        if not runs:
            raise ValueError(f"No runs found for suite={suite_name}")
        return str(runs[0]["id"])

    if selector == "prev":
        runs = store.query_runs(suite_name=suite_name, limit=2)
        if len(runs) < 2:
            raise ValueError("Fewer than 2 runs stored — nothing to compare against")
        return str(runs[1]["id"])

    if len(selector) >= 32:
        return selector

    runs = store.query_runs(suite_name=suite_name, limit=50)
    matches = [r for r in runs if str(r["id"]).startswith(selector)]
    if not matches:
        raise ValueError(f"No run found with id prefix '{selector}'")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous run prefix '{selector}' matches {len(matches)} runs")
    return str(matches[0]["id"])
