# Spec: Quorum v2 Week 3-4 — Active-Inference StabilityScorer

**Status:** Ready to build (independent of Weeks 1, 2, 5)
**Mode:** /specification
**Date:** 2026-05-10
**Effort:** 10 days (3 math/scorer, 2 protocol refactor, 2 A/B harness, 3 tuning)
**Roadmap:** `docs/ROADMAP_quorum_v2.md` Week 3-4
**Roadmap correction:** roadmap claimed current scorer is "evidence-counting"; correct is **hardcoded weighted-sum on EvidenceKind tallies** (`intent.py:172-194`). Same failure mode (saturation, no surprise-weighting), different surface.

---

## 0. Premise (do not re-argue)

Current `Intent.compute_stability()` (intent.py:172-194):

```python
score = 0.3                                                    # base
score += min(test_passes * 0.05, 0.3)                          # cap at +0.3
if any(code_committed): score += 0.2                           # binary +0.2
score += min(consumed_by_other * 0.1, 0.2)                     # cap at +0.2
score -= conflicts * 0.15                                      # uncapped
score -= test_fails * 0.15                                     # uncapped
if any(manual_approval): score += 0.3                          # binary +0.3
return clamp(score, 0.0, 1.0)
```

Failure modes:
1. **Saturation** — once `test_passes >= 6`, more passes do nothing; cannot distinguish 6 confirmations from 600.
2. **No surprise weighting** — 100 confirmations of an obvious intent counts the same as 100 confirmations of a contested one that just shifted.
3. **No uncertainty signal** — score is a point estimate; no variance to drive curiosity-directed work.
4. **Evidence-flooding vulnerable** — a misbehaving agent emitting `test_pass` events can saturate the cap and lock stability high.

Active inference fixes all four. Posterior variance gives Forge a natural curiosity gradient (target intents with high variance = "still figuring this out").

---

## 1. Goal

Replace the body of `Intent.compute_stability()` with a registry-dispatched scorer. Ship two scorers behind a feature flag: existing logic (preserved as `WeightedSumStabilityScorer`) and new `ActiveInferenceStabilityScorer`. Default stays existing scorer until A/B harness shows the new one wins.

Drop-in: every existing call site (`compute_stability()` in resolver, governor, health, visualization, versioning, sqlite_backend, constraints — 15+ sites) keeps working without modification.

---

## 2. Out of scope

| Item | Reason |
|---|---|
| Replacing TriumvirateVoter | Vote semantics unchanged |
| Multi-modal posteriors | Single Gaussian per intent is fine for v1 |
| Hyperparameter learning | Per-`EvidenceKind` priors hand-tuned for v1 |
| Rust core scorer changes | Python path only; Rust StabilityScorer (which takes consensus_rate / alignment / separation per Quorum CLAUDE.md Protocol Invariants) stays untouched and divergent |
| Per-agent scorer overrides | Module-level singleton for v1; per-agent config deferred |
| Persistence of posteriors across process restart | In-memory only; restart re-warms from `intent.evidence` list |

The Rust/Python scorer divergence is documented as a known limitation. v1 ships Python-only; if the Rust hot path needs the upgrade later, that is a separate spec.

---

## 3. Current-state baseline

### 3.1 Call sites of `compute_stability()`

15 call sites across 9 files (verified via grep `intent.py:172`):

```
constraints.py:219       resolver.py:51, 61, 73, 168, 175, 256, 309, 355
governor.py:185, 290     versioning.py:251
health.py:153            visualization.py:41, 79, 119, 151
sqlite_backend.py:71
```

All call `intent.compute_stability()` — zero-argument method on Intent. The protocol must preserve this signature.

### 3.2 Quorum protocol invariants (`packages/quorum/CLAUDE.md`)

> StabilityScorer input: recent consensus rate, alignment score, separation distance

This describes the **Rust** scorer. Python `compute_stability()` is a separate algorithm that the docstring says "mirrors" Rust but is structurally different. v1 keeps the divergence; Rust path is not touched.

### 3.3 Evidence kinds (`intent.py:35-41`)

Six kinds: `TEST_PASS`, `TEST_FAIL`, `CODE_COMMITTED`, `CONSUMED_BY_OTHER`, `CONFLICT`, `MANUAL_APPROVAL`. v1 maps each to a `(claim, obs_noise)` prior in the new scorer.

### 3.4 Forge eval framework (`packages/forge/src/animus_forge/evaluation/`)

Available rubrics: `personal-quality`, `code-edit`, `briefing-quality`. None exists for stability scoring. v1 spec adds a new rubric file: `packages/forge/rubrics/stability-scoring.yaml`.

---

## 4. Acceptance criteria (measurable)

- [ ] **AC1** — `StabilityScorer` Protocol in `convergent/scoring/protocol.py`. Single method: `score(intent: Intent) -> float`. Optional method: `curiosity(intent: Intent) -> float` (defaults to `0.0` for stateless scorers).
- [ ] **AC2** — `WeightedSumStabilityScorer` in `convergent/scoring/weighted_sum.py` reproduces current `Intent.compute_stability()` byte-identically. Verified by parameterized test over 100 random evidence lists.
- [ ] **AC3** — `ActiveInferenceStabilityScorer` in `convergent/scoring/active_inference.py` implements conjugate Gaussian posterior update over evidence stream. State is `dict[intent_id, GaussianPosterior]`.
- [ ] **AC4** — Scorer registry in `convergent/scoring/registry.py`: `set_default_scorer(scorer)`, `get_default_scorer() -> StabilityScorer`. Default value is `WeightedSumStabilityScorer()`. Thread-safe (single module-level reference, atomic Python rebind).
- [ ] **AC5** — `Intent.compute_stability()` body becomes `return get_default_scorer().score(self)`. All other Intent fields unchanged.
- [ ] **AC6** — Feature flag selection: `convergent.scoring.configure_from_env()` reads `QUORUM_SCORER` env var (`"weighted_sum"` | `"active_inference"`). Default `"weighted_sum"`. Bootstrap composition calls this at startup.
- [ ] **AC7** — Curiosity readout: `intent.curiosity_score: float` property added to Intent (computed from `get_default_scorer().curiosity(self)`). Returns 0.0 when scorer is stateless.
- [ ] **AC8** — Surprise property: `WeightedSumStabilityScorer` and `ActiveInferenceStabilityScorer` both expose `score()`; only `ActiveInferenceStabilityScorer` produces non-zero `curiosity()`. Verified by test.
- [ ] **AC9** — Evidence-flooding test: 1 `MANUAL_APPROVAL` then 1000 `TEST_PASS` events on the same intent. Weighted-sum scorer score = 1.0 (saturated). Active-inference scorer variance approaches zero, mean stable around 0.95±0.02.
- [ ] **AC10** — Surprise-weighting test: 100 `TEST_PASS` events building stable posterior, then 1 `CONFLICT`. Active-inference scorer registers larger downward shift than weighted-sum scorer (because the conflict is surprising under the prior).
- [ ] **AC11** — A/B harness in `packages/forge/src/animus_forge/evaluation/scorer_ab.py`: takes a captured event stream (CSV or EventLog dump) and an oracle file (intent_id → ground_truth_stability), computes MSE for both scorers, reports winner with bootstrap 95% CI on MSE delta.
- [ ] **AC12** — Win condition: on the captured stream + hand-labeled oracle, active-inference MSE is at least 10% lower than weighted-sum MSE with bootstrap 95% CI excluding zero.
- [ ] **AC13** — Curiosity correlation: post-hoc check on 2 weeks of historical Forge runs — Spearman ρ between active-inference `curiosity` and "did Forge subsequently target this intent for self-improvement work" is ≥0.3 with p<0.05. (Sanity check, not a hard pass-gate.)
- [ ] **AC14** — Reversibility: `set_default_scorer(WeightedSumStabilityScorer())` mid-process restores prior behavior; subsequent `compute_stability()` calls match the old implementation.
- [ ] **AC15** — Zero new production deps. `pyproject.toml` `dependencies = []` unchanged.
- [ ] **AC16** — Existing 926+ Quorum tests pass. New tests pass. Ruff + mypy clean.

---

## 5. Implementation

### 5.1 Module layout

```
packages/quorum/python/convergent/scoring/
├── __init__.py             # re-exports + configure_from_env()
├── protocol.py             # StabilityScorer Protocol
├── weighted_sum.py         # WeightedSumStabilityScorer (current logic)
├── active_inference.py     # ActiveInferenceStabilityScorer (new)
├── registry.py             # set/get default scorer
└── posterior.py            # GaussianPosterior helpers
```

### 5.2 Protocol

```python
# scoring/protocol.py
from typing import Protocol, runtime_checkable
from convergent.intent import Intent

@runtime_checkable
class StabilityScorer(Protocol):
    """Compute stability ∈ [0.0, 1.0] from an intent's evidence."""

    def score(self, intent: Intent) -> float:
        ...

    def curiosity(self, intent: Intent) -> float:
        """Posterior uncertainty ∈ [0.0, 1.0]. Stateless scorers
        return 0.0."""
        ...
```

### 5.3 Registry

```python
# scoring/registry.py
import os
from convergent.scoring.protocol import StabilityScorer

_default: StabilityScorer | None = None

def set_default_scorer(scorer: StabilityScorer) -> None:
    global _default
    _default = scorer

def get_default_scorer() -> StabilityScorer:
    global _default
    if _default is None:
        from convergent.scoring.weighted_sum import (
            WeightedSumStabilityScorer,
        )
        _default = WeightedSumStabilityScorer()
    return _default

def configure_from_env() -> None:
    name = os.environ.get("QUORUM_SCORER", "weighted_sum")
    if name == "active_inference":
        from convergent.scoring.active_inference import (
            ActiveInferenceStabilityScorer,
        )
        set_default_scorer(ActiveInferenceStabilityScorer())
    else:
        from convergent.scoring.weighted_sum import (
            WeightedSumStabilityScorer,
        )
        set_default_scorer(WeightedSumStabilityScorer())
```

### 5.4 Weighted-sum scorer (refactor of current logic)

```python
# scoring/weighted_sum.py
from convergent.intent import EvidenceKind, Intent

class WeightedSumStabilityScorer:
    """Current scorer extracted from Intent.compute_stability().
    Byte-identical behavior preserved for backward compatibility."""

    def score(self, intent: Intent) -> float:
        score = 0.3
        ev = intent.evidence
        test_passes = sum(
            1 for e in ev if e.kind == EvidenceKind.TEST_PASS
        )
        score += min(test_passes * 0.05, 0.3)
        if any(e.kind == EvidenceKind.CODE_COMMITTED for e in ev):
            score += 0.2
        dependents = sum(
            1 for e in ev if e.kind == EvidenceKind.CONSUMED_BY_OTHER
        )
        score += min(dependents * 0.1, 0.2)
        conflicts = sum(
            1 for e in ev if e.kind == EvidenceKind.CONFLICT
        )
        score -= conflicts * 0.15
        test_fails = sum(
            1 for e in ev if e.kind == EvidenceKind.TEST_FAIL
        )
        score -= test_fails * 0.15
        if any(e.kind == EvidenceKind.MANUAL_APPROVAL for e in ev):
            score += 0.3
        return max(0.0, min(1.0, score))

    def curiosity(self, intent: Intent) -> float:
        return 0.0
```

### 5.5 GaussianPosterior

```python
# scoring/posterior.py
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class GaussianPosterior:
    mean: float
    variance: float

    def update(
        self,
        observation: float,
        obs_variance: float,
    ) -> "GaussianPosterior":
        """Conjugate Gaussian update.

        Posterior = Normal with precision-weighted combination
        of prior and observation. Saturated priors (low variance)
        absorb new evidence weakly; uncertain priors absorb strongly.
        """
        prior_prec = 1.0 / max(self.variance, 1e-9)
        obs_prec = 1.0 / max(obs_variance, 1e-9)
        new_prec = prior_prec + obs_prec
        new_var = 1.0 / new_prec
        new_mean = new_var * (
            self.mean * prior_prec + observation * obs_prec
        )
        return GaussianPosterior(mean=new_mean, variance=new_var)

    def surprise(
        self,
        observation: float,
        obs_variance: float,
    ) -> float:
        """Negative log-likelihood of observation under prior.
        Returned as a non-negative scalar; zero = perfectly expected."""
        var = self.variance + obs_variance
        return 0.5 * (
            math.log(2 * math.pi * var)
            + (observation - self.mean) ** 2 / var
        )
```

### 5.6 ActiveInferenceStabilityScorer

```python
# scoring/active_inference.py
from convergent.intent import EvidenceKind, Intent
from convergent.scoring.posterior import GaussianPosterior

# (claim, observation_noise) per evidence kind.
# Claim: where this kind of evidence pulls the posterior toward.
# Obs noise: how informative one such observation is (lower = more).
DEFAULT_PRIORS: dict[EvidenceKind, tuple[float, float]] = {
    EvidenceKind.TEST_PASS: (0.7, 0.3),
    EvidenceKind.TEST_FAIL: (0.1, 0.2),
    EvidenceKind.CODE_COMMITTED: (0.8, 0.15),
    EvidenceKind.CONSUMED_BY_OTHER: (0.75, 0.25),
    EvidenceKind.CONFLICT: (0.05, 0.1),
    EvidenceKind.MANUAL_APPROVAL: (0.95, 0.05),
}

PRIOR_MEAN = 0.3
PRIOR_VARIANCE = 0.25  # broad prior: σ=0.5


class ActiveInferenceStabilityScorer:
    """Bayesian posterior over intent stability.

    Each evidence event is a noisy observation; conjugate Gaussian
    update accumulates evidence with surprise-weighted impact.
    Curiosity = posterior variance. Saturated intents (low variance)
    resist evidence flooding; surprising contradictions move the
    posterior more than confirmatory observations of settled intents.
    """

    def __init__(
        self,
        kind_priors: dict[EvidenceKind, tuple[float, float]] | None = None,
        prior_mean: float = PRIOR_MEAN,
        prior_variance: float = PRIOR_VARIANCE,
    ) -> None:
        self._kind_priors = kind_priors or DEFAULT_PRIORS
        self._prior = GaussianPosterior(
            mean=prior_mean, variance=prior_variance
        )
        self._posteriors: dict[str, GaussianPosterior] = {}
        self._processed: dict[str, int] = {}

    def score(self, intent: Intent) -> float:
        post = self._update(intent)
        return max(0.0, min(1.0, post.mean))

    def curiosity(self, intent: Intent) -> float:
        post = self._update(intent)
        # Normalize variance to [0, 1]; prior variance is the cap.
        return min(1.0, post.variance / self._prior.variance)

    def _update(self, intent: Intent) -> GaussianPosterior:
        post = self._posteriors.get(intent.id, self._prior)
        already = self._processed.get(intent.id, 0)
        new_evidence = intent.evidence[already:]
        for ev in new_evidence:
            claim, noise = self._kind_priors.get(ev.kind, (0.5, 1.0))
            post = post.update(observation=claim, obs_variance=noise)
        self._posteriors[intent.id] = post
        self._processed[intent.id] = len(intent.evidence)
        return post
```

**Key property:** `_update()` is incremental — only new evidence (`intent.evidence[already:]`) is absorbed. Re-scoring an intent is O(new evidence), not O(total evidence). After many updates, variance shrinks; further evidence has diminishing impact unless its claim is far from current posterior mean (surprise-weighted).

### 5.7 Intent integration

```python
# intent.py — replace compute_stability body
def compute_stability(self) -> float:
    """Compute stability from evidence via the registered scorer."""
    from convergent.scoring.registry import get_default_scorer
    return get_default_scorer().score(self)

@property
def curiosity_score(self) -> float:
    """Posterior uncertainty if active scorer is stateful, else 0."""
    from convergent.scoring.registry import get_default_scorer
    return get_default_scorer().curiosity(self)
```

Lazy import inside method body avoids import cycles between `scoring/` and `intent.py`.

### 5.8 Bootstrap wiring

In Bootstrap startup (location TBD by Bootstrap maintainer):

```python
from convergent.scoring import configure_from_env
configure_from_env()
```

Operator flips behavior with `QUORUM_SCORER=active_inference`.

### 5.9 A/B harness

```python
# packages/forge/src/animus_forge/evaluation/scorer_ab.py
from dataclasses import dataclass
from statistics import mean

@dataclass
class ScorerABResult:
    scorer_a_mse: float
    scorer_b_mse: float
    delta: float
    ci_low: float
    ci_high: float
    winner: str

def replay_and_score(
    event_stream: list[dict],
    oracle: dict[str, float],
    scorer_a: "StabilityScorer",
    scorer_b: "StabilityScorer",
    bootstrap_n: int = 1000,
) -> ScorerABResult:
    """Replay an event stream through two scorers, return MSE comparison
    against ground-truth oracle with bootstrap 95% CI on MSE delta."""
    # ... reconstruct Intents from stream, score with both,
    #     compute MSE per intent, bootstrap CI on MSE delta ...
```

Ground-truth oracle source: hand-labeled subset of intents from a captured Animus run. Labels: "did this intent ultimately ship / get adopted / fail"? Map to `{0.0, 0.5, 1.0}`.

---

## 6. Test plan

### 6.1 Protocol contract (new: `tests/test_scoring_protocol.py`)

- `test_weighted_sum_satisfies_protocol`
- `test_active_inference_satisfies_protocol`
- `test_default_registry_returns_weighted_sum`
- `test_set_default_scorer_replaces_default`
- `test_configure_from_env_weighted_sum`
- `test_configure_from_env_active_inference`

### 6.2 Weighted-sum parity (new: `tests/test_weighted_sum_parity.py`)

- `test_byte_identical_to_legacy_compute_stability`
  - Parameterized over 100 randomly generated evidence lists
  - Compare new `WeightedSumStabilityScorer.score(intent)` to a frozen copy of the legacy implementation
  - Assert exact equality

### 6.3 Active inference behavior (new: `tests/test_active_inference.py`)

- `test_empty_evidence_returns_prior_mean`
- `test_first_test_pass_moves_posterior_toward_prior_claim`
- `test_variance_decreases_with_more_evidence`
- `test_evidence_flooding_does_not_saturate` (AC9)
- `test_contradictory_evidence_creates_larger_shift_than_confirming` (AC10)
- `test_curiosity_decays_to_zero_over_time`
- `test_re_score_idempotent_when_no_new_evidence`
- `test_per_intent_state_isolation`
- `test_warm_start_from_existing_evidence`

### 6.4 GaussianPosterior unit (new: `tests/test_posterior.py`)

- `test_update_with_certain_observation_collapses_variance`
- `test_update_with_uncertain_observation_barely_moves_mean`
- `test_surprise_zero_at_prior_mean`
- `test_surprise_grows_quadratically_with_distance`
- `test_repeated_updates_converge_to_observation_when_obs_certain`

### 6.5 Integration with existing Quorum (new: `tests/test_scoring_integration.py`)

- `test_intent_compute_stability_uses_registry`
- `test_switching_scorer_changes_compute_stability`
- `test_curiosity_score_property_returns_scorer_curiosity`
- `test_resolver_uses_registered_scorer` (verify resolver.py:51 path)
- `test_existing_intent_tests_pass_with_weighted_sum_default` — full Quorum test suite green

### 6.6 A/B harness (new: `packages/forge/tests/test_scorer_ab.py`)

- `test_replay_reproduces_scores`
- `test_mse_computed_correctly`
- `test_bootstrap_ci_excludes_zero_when_clear_winner`
- `test_bootstrap_ci_includes_zero_when_no_signal`

### 6.7 Tuning data prep (out of automated test scope)

Manual: capture 2 weeks of EventLog data from real Animus runs, hand-label 50 intents for ground truth, store at `packages/forge/tests/fixtures/scorer_oracle.json`. This is the input to AC12.

---

## 7. File-by-file change list

| File | Change | LOC |
|---|---|---|
| `packages/quorum/python/convergent/scoring/__init__.py` | New | +30 |
| `packages/quorum/python/convergent/scoring/protocol.py` | New | +30 |
| `packages/quorum/python/convergent/scoring/posterior.py` | New | +60 |
| `packages/quorum/python/convergent/scoring/weighted_sum.py` | New (refactor) | +50 |
| `packages/quorum/python/convergent/scoring/active_inference.py` | New | +90 |
| `packages/quorum/python/convergent/scoring/registry.py` | New | +50 |
| `packages/quorum/python/convergent/intent.py` | Replace `compute_stability()` body, add `curiosity_score` property | -25 +20 |
| `packages/quorum/tests/test_scoring_protocol.py` | New | +90 |
| `packages/quorum/tests/test_weighted_sum_parity.py` | New | +60 |
| `packages/quorum/tests/test_active_inference.py` | New | +180 |
| `packages/quorum/tests/test_posterior.py` | New | +80 |
| `packages/quorum/tests/test_scoring_integration.py` | New | +100 |
| `packages/forge/src/animus_forge/evaluation/scorer_ab.py` | New | +120 |
| `packages/forge/tests/test_scorer_ab.py` | New | +90 |
| `packages/forge/rubrics/stability-scoring.yaml` | New | +30 |
| `packages/quorum/CHANGELOG.md` | 1.5.0 entry | +10 |
| `packages/forge/CHANGELOG.md` | Note A/B harness | +5 |
| **Total** | | **~1,070 LOC** |

---

## 8. Rollout

1. PR1: scoring package + WeightedSumStabilityScorer + Intent refactor + parity tests. Merge as 1.5.0-rc1. **Critical:** AC2 (byte-identical parity) must be green before merge.
2. PR2: ActiveInferenceStabilityScorer + posterior + tests. Merge as 1.5.0-rc2 with default still weighted_sum.
3. PR3: A/B harness + rubric + Forge integration. Merge as part of Forge release.
4. Capture 2 weeks of data, hand-label oracle subset.
5. Run A/B harness. If AC12 passes, flip Bootstrap default to `QUORUM_SCORER=active_inference` for one user (ARETE) for 1 week. If no regressions, ship 1.5.0 with active_inference as default.
6. ADL `ADL-202605XX-001` class ARCH — "Quorum stability scoring became Bayesian; weighted-sum preserved as fallback."

---

## 9. Constitutional alignment

| Principle | How |
|---|---|
| **P1 Sovereignty** | Scorer state in-memory and local; no telemetry. |
| **P2 Continuity** | Posterior fits to existing evidence on warm-start; no data migration; reversible via env flip. |
| **P3 Transparency** | Score derivation auditable via re-running the scorer over EventLog evidence stream (Week 1 makes this possible). |
| **P5 Signed writes** | `compute_stability()` reads only signed Intent fields; no write surface change. |
| **Non-negotiable #5** | No change to `convergent.intent` write path; only computation logic swapped. |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Posterior diverges over long evidence streams (numerical drift) | Variance is bounded above by prior, below by `1e-9` clamp. Tested in AC9. |
| Active scorer wrong on real data → flips production score | Default stays weighted_sum until AC12 passes; env flag is per-process and reversible. |
| Per-intent state grows unbounded | Bounded by intent count; at typical Animus scale (<10K intents) memory is negligible (~1KB per posterior). Add size cap + LRU only if a real load problem appears. |
| Rust scorer divergence causes confusion | Documented as known limitation in §2. Re-eval gate decides if Rust upgrade is needed. |
| `_processed` counter wrong if evidence list is mutated out-of-order | Evidence is append-only by convention (no public API for insertion). Add assertion in scorer that `len(evidence) >= already`; raise if violated. |
| Resolver / governor using stale stability cache | Existing code recomputes via `compute_stability()` on each call (no cache). Verified via grep — no `@cached_property` or memo decorators. |
| Forge consumes `curiosity_score` before AC13 sanity check passes | `curiosity_score` property always returns 0.0 under weighted_sum default; safe no-op until active_inference is enabled. |
| Hyperparameters in DEFAULT_PRIORS are wrong for real data | AC12 tunes by replaying historical stream. Tuning notes captured in spec rollout step 5. |

---

## 11. What this unblocks

- **Forge curiosity-directed self-improvement** — Forge subscribes to intents with `curiosity_score > 0.5` and prioritizes investigation/test work on them. Closes the existing self-improvement loop's "what to work on next?" gap.
- **Bootstrap dashboard intent uncertainty tile** — show variance bars alongside mean stability.
- **Future Rust scorer upgrade** — Python active-inference proves out the math before porting to PyO3 (post-re-eval gate).
- **TIAID assessment artifact** — "your AI doesn't know what it doesn't know" becomes a concrete demonstration. Curiosity is the visible signal of epistemic humility.

---

## 12. Open questions (defer to build)

1. Should `DEFAULT_PRIORS` be loaded from a YAML config so operator can tune without code change? Defer until tuning data exists; constants are fine for v1.
2. Should `curiosity()` be exposed as a Quorum query (e.g., `resolver.list_curious_intents(min_curiosity=0.5)`)? Likely yes once Forge consumes it. Add to v1 if Forge integration is in scope; otherwise add when Forge needs it.
3. Should the A/B harness support more than 2 scorers? YAGNI — keep 2 for v1.
4. Should warm-start be opt-out (cold-start everywhere) for stricter Bayesian semantics? No — warm-start is the practical default; cold-start can be configured per-intent if a use case appears.

---

## 13. Ground-truth oracle creation (manual prerequisite for AC12)

This is operator work, not code:

1. After Week 1 ships, let EventLog accumulate for 14 days under normal usage.
2. Export: `EventLog.query(event_type=INTENT_PUBLISHED, limit=10000)` + correlation-linked `SCORE_UPDATED`, `INTENT_RESOLVED` events.
3. Sample 50 intents stratified by final stability score.
4. For each sampled intent, ARETE assigns ground truth ∈ {0.0 (rejected/abandoned), 0.5 (incomplete/uncertain), 1.0 (shipped/adopted)}.
5. Save as `packages/forge/tests/fixtures/scorer_oracle.json`.
6. Run AC12 harness; iterate `DEFAULT_PRIORS` if needed.

---

*End of spec.*
