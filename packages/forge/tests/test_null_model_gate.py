"""Tests for the null-model gate (Prima Materia methodology).

The gate compares an observed improvement delta against a distribution of
placebo (no-op) perturbations and classifies it into STRUCTURAL, HEURISTIC,
NEUTRAL, or REGRESSION tiers.
"""

from __future__ import annotations

import random

from animus_forge.self_improve.null_model_gate import (
    MetricSnapshot,
    NullModelGateConfig,
    NullModelVerdict,
    _default_placebo_perturbation,
    classify_verdict,
    gate_against_null,
    generate_placebo_changes,
    score_delta,
)

# ---------------------------------------------------------------------------
# MetricSnapshot
# ---------------------------------------------------------------------------


class TestMetricSnapshot:
    def test_empty_snapshot_has_no_metric_keys(self):
        snap = MetricSnapshot()
        assert snap.metric_keys() == []

    def test_metric_keys_exclude_none_fields(self):
        snap = MetricSnapshot(
            tests_total=100,
            tests_passing=95,
            coverage_percent=None,
        )
        keys = snap.metric_keys()
        assert "tests_total" in keys
        assert "tests_passing" in keys
        assert "coverage_percent" not in keys

    def test_delta_to_computes_component_wise_difference(self):
        baseline = MetricSnapshot(tests_total=100, coverage_percent=80.0)
        after = MetricSnapshot(tests_total=110, coverage_percent=85.0)
        delta = baseline.delta_to(after)
        assert delta == {"tests_total": 10.0, "coverage_percent": 5.0}

    def test_delta_to_only_uses_shared_non_none_keys(self):
        a = MetricSnapshot(tests_total=100, coverage_percent=80.0)
        b = MetricSnapshot(tests_total=110, lint_errors=5.0)
        delta = a.delta_to(b)
        assert delta == {"tests_total": 10.0}

    def test_delta_is_directional_not_absolute(self):
        a = MetricSnapshot(coverage_percent=80.0)
        b = MetricSnapshot(coverage_percent=70.0)
        assert a.delta_to(b) == {"coverage_percent": -10.0}


# ---------------------------------------------------------------------------
# score_delta: direction-aware L2 scoring
# ---------------------------------------------------------------------------


class TestScoreDelta:
    def test_empty_delta_scores_zero(self):
        assert score_delta({}) == 0.0

    def test_coverage_increase_scores_positive(self):
        score = score_delta({"coverage_percent": 5.0})
        assert score > 0

    def test_coverage_decrease_scores_negative(self):
        score = score_delta({"coverage_percent": -5.0})
        assert score < 0

    def test_lint_errors_decreasing_is_improvement(self):
        score = score_delta({"lint_errors": -3.0})
        assert score > 0, "Fewer lint errors is an improvement"

    def test_lint_errors_increasing_is_regression(self):
        score = score_delta({"lint_errors": 3.0})
        assert score < 0, "More lint errors is a regression"

    def test_lines_of_code_is_neutral(self):
        pos = score_delta({"lines_of_code": 100.0})
        neg = score_delta({"lines_of_code": -100.0})
        assert pos == 0.0
        assert neg == 0.0

    def test_multiple_improvements_compose(self):
        score = score_delta({"coverage_percent": 5.0, "lint_errors": -3.0})
        single = score_delta({"coverage_percent": 5.0})
        assert score > single, "Multiple improvements > single improvement"

    def test_mixed_delta_takes_sign_of_dominant_direction(self):
        # Small regression on one metric, large improvement on another:
        # signed sum is positive, overall score is positive.
        score = score_delta({"coverage_percent": 10.0, "lint_errors": 1.0})
        assert score > 0

    def test_unknown_metric_is_ignored(self):
        score = score_delta({"made_up_metric": 999.0})
        assert score == 0.0


# ---------------------------------------------------------------------------
# classify_verdict: percentile → verdict tier
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    def test_empty_null_scores_returns_neutral(self):
        verdict, _, _ = classify_verdict(1.0, [])
        assert verdict == NullModelVerdict.NEUTRAL

    def test_observed_above_95th_pct_is_structural(self):
        null = [0.0] * 100
        verdict, pct, _ = classify_verdict(5.0, null)
        assert verdict == NullModelVerdict.STRUCTURAL
        assert pct >= 95.0

    def test_observed_in_75_to_95_pct_is_heuristic(self):
        null = list(range(100))  # 0..99
        verdict, pct, _ = classify_verdict(80.0, null)
        assert verdict == NullModelVerdict.HEURISTIC
        assert 75.0 <= pct < 95.0

    def test_observed_in_noise_is_neutral(self):
        null = list(range(100))
        verdict, pct, _ = classify_verdict(50.0, null)
        assert verdict == NullModelVerdict.NEUTRAL
        assert pct < 75.0

    def test_strong_negative_observed_is_regression(self):
        null = [0.0] * 100
        verdict, pct, _ = classify_verdict(-5.0, null)
        assert verdict == NullModelVerdict.REGRESSION
        assert pct <= 5.0

    def test_small_negative_in_noise_is_neutral_not_regression(self):
        # Symmetric null around zero — a tiny negative shouldn't regress
        null = [float(i - 50) for i in range(100)]
        verdict, _, _ = classify_verdict(-1.0, null)
        assert verdict == NullModelVerdict.NEUTRAL

    def test_rationale_mentions_percentile(self):
        _, _, rationale = classify_verdict(5.0, [0.0] * 100)
        assert "percentile" in rationale.lower()


# ---------------------------------------------------------------------------
# gate_against_null: end-to-end
# ---------------------------------------------------------------------------


class TestGateAgainstNull:
    def _make_baseline(self) -> MetricSnapshot:
        return MetricSnapshot(
            tests_total=100,
            tests_passing=95,
            coverage_percent=80.0,
            lint_errors=10.0,
            type_coverage_percent=60.0,
            docstring_coverage_percent=50.0,
        )

    def test_real_improvement_vs_zero_null_is_structural(self):
        baseline = self._make_baseline()
        proposed = MetricSnapshot(
            tests_total=100,
            tests_passing=95,
            coverage_percent=90.0,  # +10 coverage
            lint_errors=2.0,  # -8 lint errors
            type_coverage_percent=60.0,
            docstring_coverage_percent=50.0,
        )
        null_samples = [self._make_baseline() for _ in range(30)]
        result = gate_against_null(baseline, proposed, null_samples)
        assert result.verdict == NullModelVerdict.STRUCTURAL
        assert result.observed_score > 0
        assert result.percentile >= 95.0

    def test_regression_detected_against_zero_null(self):
        baseline = self._make_baseline()
        proposed = MetricSnapshot(
            tests_total=100,
            tests_passing=90,  # -5 passing
            coverage_percent=70.0,  # -10 coverage
            lint_errors=25.0,  # +15 lint errors
            type_coverage_percent=60.0,
            docstring_coverage_percent=50.0,
        )
        null_samples = [self._make_baseline() for _ in range(30)]
        result = gate_against_null(baseline, proposed, null_samples)
        assert result.verdict == NullModelVerdict.REGRESSION
        assert result.observed_score < 0

    def test_no_change_vs_noisy_null_is_neutral(self):
        baseline = self._make_baseline()
        proposed = self._make_baseline()  # identical
        # Null has symmetric noise around zero
        rng = random.Random(0)
        null_samples = [
            MetricSnapshot(
                tests_total=100,
                tests_passing=95,
                coverage_percent=80.0 + rng.uniform(-2, 2),
                lint_errors=10.0 + rng.uniform(-2, 2),
                type_coverage_percent=60.0,
                docstring_coverage_percent=50.0,
            )
            for _ in range(100)
        ]
        result = gate_against_null(baseline, proposed, null_samples)
        assert result.verdict == NullModelVerdict.NEUTRAL

    def test_result_has_audit_dict(self):
        baseline = self._make_baseline()
        proposed = self._make_baseline()
        result = gate_against_null(baseline, proposed, [self._make_baseline()])
        audit = result.to_audit_dict()
        assert "verdict" in audit
        assert "observed_score" in audit
        assert "percentile" in audit
        assert "rationale" in audit
        assert "observed_delta" in audit

    def test_null_summary_stats_are_computed(self):
        baseline = self._make_baseline()
        proposed = self._make_baseline()
        null_samples = [self._make_baseline() for _ in range(10)]
        result = gate_against_null(baseline, proposed, null_samples)
        assert result.n_null_samples == 10
        # With identical nulls, std is 0 and mean is 0
        assert result.null_mean == 0.0
        assert result.null_std == 0.0


# ---------------------------------------------------------------------------
# Placebo generator
# ---------------------------------------------------------------------------


SAMPLE_SOURCE = """\
def example(x):
    y = x + 1
    z = y * 2
    if z > 10:
        return z
    return x


def another(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""


class TestPlaceboGenerator:
    def test_default_perturbation_changes_source(self):
        rng = random.Random(42)
        perturbed = _default_placebo_perturbation(SAMPLE_SOURCE, rng)
        assert perturbed != SAMPLE_SOURCE

    def test_default_perturbation_preserves_original_lines(self):
        rng = random.Random(42)
        perturbed = _default_placebo_perturbation(SAMPLE_SOURCE, rng)
        # Every original non-blank content line should still be present
        for line in SAMPLE_SOURCE.splitlines():
            if line.strip():
                assert line in perturbed or line.rstrip() in perturbed

    def test_perturbation_short_source_returns_unchanged(self):
        short = "x = 1\n"
        rng = random.Random(0)
        assert _default_placebo_perturbation(short, rng) == short

    def test_generate_placebo_changes_returns_correct_count(self):
        originals = {"a.py": SAMPLE_SOURCE, "b.py": SAMPLE_SOURCE}
        samples = generate_placebo_changes(originals, n=5, seed=1)
        assert len(samples) == 5
        for sample in samples:
            assert set(sample.keys()) == {"a.py", "b.py"}

    def test_placebo_samples_are_deterministic_under_same_seed(self):
        originals = {"a.py": SAMPLE_SOURCE}
        s1 = generate_placebo_changes(originals, n=3, seed=7)
        s2 = generate_placebo_changes(originals, n=3, seed=7)
        assert s1 == s2

    def test_different_seeds_give_different_samples(self):
        originals = {"a.py": SAMPLE_SOURCE}
        s1 = generate_placebo_changes(originals, n=10, seed=1)
        s2 = generate_placebo_changes(originals, n=10, seed=2)
        # At least some samples should differ
        assert s1 != s2


# ---------------------------------------------------------------------------
# NullModelGateConfig
# ---------------------------------------------------------------------------


class TestGateConfig:
    def test_disabled_config_allows_anything(self):
        config = NullModelGateConfig(enabled=False)
        assert config.allows(NullModelVerdict.REGRESSION)
        assert config.allows(NullModelVerdict.NEUTRAL)

    def test_default_config_requires_heuristic_or_better(self):
        config = NullModelGateConfig()
        assert config.allows(NullModelVerdict.STRUCTURAL)
        assert config.allows(NullModelVerdict.HEURISTIC)
        assert not config.allows(NullModelVerdict.NEUTRAL)

    def test_regression_always_blocked_when_block_regressions(self):
        config = NullModelGateConfig(
            block_regressions=True,
            min_verdict=NullModelVerdict.NEUTRAL,
        )
        assert not config.allows(NullModelVerdict.REGRESSION)
        assert config.allows(NullModelVerdict.NEUTRAL)

    def test_strict_structural_only_config(self):
        config = NullModelGateConfig(min_verdict=NullModelVerdict.STRUCTURAL)
        assert config.allows(NullModelVerdict.STRUCTURAL)
        assert not config.allows(NullModelVerdict.HEURISTIC)
        assert not config.allows(NullModelVerdict.NEUTRAL)

    def test_regressions_allowed_when_block_false(self):
        config = NullModelGateConfig(
            block_regressions=False,
            min_verdict=NullModelVerdict.NEUTRAL,
        )
        assert config.allows(NullModelVerdict.REGRESSION)
