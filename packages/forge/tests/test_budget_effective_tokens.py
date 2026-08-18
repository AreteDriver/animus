"""Tests for the Effective-Tokens cost metric on BudgetManager.

ET = m × (1.0·I + 0.1·C + 4.0·O). Lifted from GitHub's "Improving Token
Efficiency in GitHub Agentic Workflows" (2026-05). Lets us compare workflows
across model tiers (Haiku / Sonnet / Opus) and input/cache/output mixes on
a single cost axis.
"""

from __future__ import annotations

import pytest
from animus_kernel.budget import (
    DEFAULT_MODEL_MULTIPLIERS,
    BudgetConfig,
    BudgetManager,
    UsageRecord,
    effective_tokens,
)


def _rec(**kw) -> UsageRecord:
    return UsageRecord(agent_id=kw.pop("agent_id", "a"), tokens=kw.pop("tokens", 0), **kw)


class TestEffectiveTokensFormula:
    def test_breakdown_weights_input_cache_output(self):
        # I=100, C=100, O=100, no model → m=1.0 → ET = 1·100 + 0.1·100 + 4·100 = 510
        r = _rec(input_tokens=100, cache_read_tokens=100, output_tokens=100)
        assert effective_tokens(r) == pytest.approx(510.0)

    def test_output_dominates_input(self):
        # Cost is dominated by output: same totals, but all-output vs all-input
        out = _rec(output_tokens=1000)
        inp = _rec(input_tokens=1000)
        cache = _rec(cache_read_tokens=1000)
        assert effective_tokens(out) == pytest.approx(4000.0)
        assert effective_tokens(inp) == pytest.approx(1000.0)
        assert effective_tokens(cache) == pytest.approx(100.0)
        assert effective_tokens(out) > effective_tokens(inp) > effective_tokens(cache)

    def test_no_breakdown_is_model_weighted_not_output_weighted(self):
        # Post-A1-flip: a raw-only record (no in/out breakdown) is neutral —
        # model-tier weighted but NOT output-weighted. We don't claim output
        # cost we never observed, so ET == raw for an unknown model. This is
        # what keeps ET enforcement non-breaking on the executor's raw path.
        assert effective_tokens(_rec(tokens=100)) == pytest.approx(100.0)
        # A known-expensive tier still gets penalized by its multiplier.
        assert effective_tokens(_rec(tokens=100, model="claude-opus-4-7")) == pytest.approx(500.0)

    def test_model_multiplier_haiku_cheaper_than_sonnet(self):
        haiku = _rec(input_tokens=1000, output_tokens=1000, model="claude-haiku-4-5")
        sonnet = _rec(input_tokens=1000, output_tokens=1000, model="claude-sonnet-4-6")
        opus = _rec(input_tokens=1000, output_tokens=1000, model="claude-opus-4-7")
        assert effective_tokens(haiku) < effective_tokens(sonnet) < effective_tokens(opus)
        # Sonnet base: 1.0 × (1·1000 + 4·1000) = 5000
        assert effective_tokens(sonnet) == pytest.approx(5000.0)
        # C12: Haiku-4 blended rate is $3/M (multiplier 0.3333), not legacy
        # 3.x ($0.75/M, multiplier 0.0833). Opus remains 5× sonnet.
        assert effective_tokens(haiku) == pytest.approx(1666.5)
        assert effective_tokens(opus) == pytest.approx(25000.0)

    def test_unknown_model_defaults_to_sonnet_tier(self):
        r = _rec(input_tokens=1000, output_tokens=1000, model="some-mystery-model")
        assert effective_tokens(r) == pytest.approx(5000.0)  # m=1.0 fallback

    def test_versioned_model_id_substring_match(self):
        # A long versioned id should still map via substring (e.g. "claude-sonnet" hit)
        r = _rec(input_tokens=1000, output_tokens=1000, model="claude-sonnet-4-6-1m")
        assert effective_tokens(r) == pytest.approx(5000.0)

    def test_custom_multiplier_table_overrides(self):
        r = _rec(input_tokens=1000, output_tokens=1000, model="haiku")
        # Override Haiku to 0.5; expect ET = 0.5 × 5000 = 2500
        assert effective_tokens(r, {"haiku": 0.5}) == pytest.approx(2500.0)

    def test_default_multipliers_table_is_sane(self):
        # Sanity: ordering Haiku < Sonnet < Opus must hold in the default table
        assert DEFAULT_MODEL_MULTIPLIERS["haiku"] < DEFAULT_MODEL_MULTIPLIERS["sonnet"]
        assert DEFAULT_MODEL_MULTIPLIERS["sonnet"] < DEFAULT_MODEL_MULTIPLIERS["opus"]
        assert DEFAULT_MODEL_MULTIPLIERS["ollama"] == 0.0


class TestRecordUsageBreakdown:
    def test_record_with_breakdown_stores_fields(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        rec = bm.record_usage(
            "agent1", input_tokens=300, output_tokens=200, cache_read_tokens=50, model="sonnet"
        )
        assert rec.input_tokens == 300 and rec.output_tokens == 200 and rec.cache_read_tokens == 50
        assert rec.model == "sonnet"

    def test_record_without_explicit_total_sums_the_breakdown(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        rec = bm.record_usage("agent1", input_tokens=300, output_tokens=200, cache_read_tokens=50)
        assert rec.tokens == 550  # 300 + 200 + 50
        assert bm.used == 550

    def test_record_legacy_tokens_only_still_works(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        rec = bm.record_usage("agent1", tokens=400, operation="legacy")
        assert rec.tokens == 400
        assert rec.input_tokens == 0 and rec.output_tokens == 0
        assert bm.used == 400

    def test_explicit_total_takes_precedence(self):
        # Caller is allowed to pass a different rolled-up total than I+O+C; the
        # breakdown still informs ET but the budget counts what they ask for.
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        rec = bm.record_usage("agent1", tokens=1000, input_tokens=100, output_tokens=100)
        assert rec.tokens == 1000
        assert bm.used == 1000


class TestManagerAggregateET:
    def test_total_effective_tokens_sums_across_records(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        bm.record_usage("a1", input_tokens=100, output_tokens=100, model="sonnet")  # 500
        bm.record_usage("a2", input_tokens=100, output_tokens=100, model="haiku")  # 166.65
        assert bm.total_effective_tokens() == pytest.approx(666.65)

    def test_effective_tokens_by_agent_groups_correctly(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        bm.record_usage("a1", input_tokens=100, output_tokens=100, model="sonnet")  # 500
        bm.record_usage("a1", input_tokens=50, output_tokens=50, model="sonnet")  # 250
        bm.record_usage("a2", input_tokens=100, output_tokens=100, model="haiku")  # 166.65
        by_agent = bm.effective_tokens_by_agent()
        assert by_agent["a1"] == pytest.approx(750.0)
        assert by_agent["a2"] == pytest.approx(166.65)

    def test_config_can_override_model_multipliers(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000, model_multipliers={"sonnet": 2.0}))
        bm.record_usage(
            "a1", input_tokens=100, output_tokens=100, model="sonnet"
        )  # 2.0 × 500 = 1000
        assert bm.total_effective_tokens() == pytest.approx(1000.0)

    def test_empty_history_returns_zero(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        assert bm.total_effective_tokens() == 0.0
        assert bm.effective_tokens_by_agent() == {}


class TestEffectiveTokenEnforcement:
    """A1 flip: ET is an ENFORCED ceiling by default, derived from
    total_budget when not set explicitly. The worse of raw/ET governs."""

    def test_enforced_by_default_derived_from_total_budget(self):
        # No explicit effective_token_budget → ceiling derives from
        # total_budget. An output-heavy opus record is cheap in raw tokens but
        # expensive in ET, so it trips EXCEEDED by default — this is the flip.
        mgr = BudgetManager(BudgetConfig(total_budget=10_000))
        assert mgr.effective_ceiling == 10_000  # derived
        mgr.record_usage("a", output_tokens=1_000, model="claude-opus-4-8")
        # Raw = 1000/10000 = 10% (OK), but ET = 5.0 * 4 * 1000 = 20000 > 10000.
        assert mgr.status.value == "exceeded"

    def test_escape_hatch_pure_raw_when_enforcement_off(self):
        # enforce_effective_tokens=False restores legacy pure-raw behavior.
        mgr = BudgetManager(BudgetConfig(total_budget=10_000, enforce_effective_tokens=False))
        mgr.record_usage("a", output_tokens=1_000, model="claude-opus-4-8")
        assert mgr.status.value == "ok"
        assert mgr.effective_remaining == float("inf")

    def test_raw_only_records_stay_non_breaking(self):
        # The executor records raw-only usage (no model/breakdown). With ET
        # enforced-by-default, those must behave exactly like raw (ET == raw),
        # so existing workflows do not trip earlier than before.
        mgr = BudgetManager(BudgetConfig(total_budget=10_000))
        mgr.record_usage("step1", 2_000)  # raw-only, no model
        assert mgr.effective_used == pytest.approx(2_000.0)
        assert mgr.status.value == "ok"  # 20% on both axes

    def test_et_ceiling_trips_when_raw_reads_healthy(self):
        # Output-heavy opus run: cheap in raw tokens, expensive in real cost.
        # Raw budget is generous; ET budget is tight → ET must govern.
        cfg = BudgetConfig(total_budget=1_000_000, effective_token_budget=10_000)
        mgr = BudgetManager(cfg)
        mgr.record_usage("a", output_tokens=5_000, model="claude-opus-4-8")
        # Raw ratio = 5000/1e6 = 0.5% (OK), but ET = 5.0 * 4 * 5000 = 100k >> 10k.
        assert mgr.effective_used > cfg.effective_token_budget
        assert mgr.status.value == "exceeded"

    def test_et_remaining_tracks_budget(self):
        cfg = BudgetConfig(total_budget=1_000_000, effective_token_budget=10_000)
        mgr = BudgetManager(cfg)
        mgr.record_usage("a", input_tokens=1_000)  # no model → m=1.0, ET=1000
        assert mgr.effective_used == pytest.approx(1_000.0)
        assert mgr.effective_remaining == pytest.approx(9_000.0)

    def test_reset_clears_effective(self):
        cfg = BudgetConfig(total_budget=1_000, effective_token_budget=1_000)
        mgr = BudgetManager(cfg)
        mgr.record_usage("a", output_tokens=100)
        assert mgr.effective_used > 0
        mgr.reset()
        assert mgr.effective_used == 0.0


class TestEffectiveTokenAdmission:
    """C8: the ET ceiling is enforced at ALLOCATION admission — a step that is
    cheap in raw tokens but expensive in cost-weighted ET is refused BEFORE it
    runs, not only caught post-hoc on the next recorded step."""

    def test_allocate_rejects_when_effective_estimate_exceeds_ceiling(self):
        cfg = BudgetConfig(total_budget=1_000_000, effective_token_budget=10_000)
        mgr = BudgetManager(cfg)
        # Raw fits easily (1k of 1M); ET estimate 20k blows the 10k ceiling.
        assert mgr.can_allocate(1_000, effective=20_000) is False
        assert mgr.allocate(1_000, effective=20_000) is False
        # Within the ET ceiling → admitted.
        assert mgr.allocate(1_000, effective=5_000) is True

    def test_pending_effective_reservation_blocks_second_alloc(self):
        cfg = BudgetConfig(total_budget=1_000_000, effective_token_budget=10_000)
        mgr = BudgetManager(cfg)
        assert mgr.allocate(1_000, effective=6_000) is True
        # Second step sees the 6k pending ET reservation: 6k+6k > 10k → refused.
        assert mgr.allocate(1_000, effective=6_000) is False
        mgr.release(1_000, effective=6_000)
        assert mgr.allocate(1_000, effective=6_000) is True  # freed

    def test_default_effective_estimate_is_neutral_tokens(self):
        # No explicit estimate → est == tokens; with the derived ceiling
        # (== total_budget) and unit multipliers, ET admission coincides with
        # the raw check, so legacy callers are unaffected. (reserve_tokens=0 to
        # isolate the ET path from the raw retry-buffer.)
        mgr = BudgetManager(BudgetConfig(total_budget=10_000, reserve_tokens=0))
        assert mgr.allocate(8_000) is True
        assert mgr.allocate(5_000) is False  # raw+ET both exceeded

    def test_admission_tightens_after_expensive_usage(self):
        # Once expensive opus usage exhausts the ET ceiling, a raw-cheap step is
        # refused at admission even though abundant raw headroom remains.
        cfg = BudgetConfig(total_budget=1_000_000, effective_token_budget=10_000)
        mgr = BudgetManager(cfg)
        mgr.record_usage("a", output_tokens=2_000, model="claude-opus-4-8")  # ET=40k
        assert mgr.effective_used > cfg.effective_token_budget
        assert mgr.can_allocate(1_000) is False  # raw OK, ET exhausted

    def test_off_switch_skips_admission_check(self):
        # enforce_effective_tokens=False → ceiling is inf → ET admission never
        # rejects; only raw governs.
        cfg = BudgetConfig(
            total_budget=1_000_000,
            effective_token_budget=10_000,
            enforce_effective_tokens=False,
        )
        mgr = BudgetManager(cfg)
        assert mgr.allocate(1_000, effective=10_000_000) is True


class TestSession1DoneCriteria:
    """A1 done-criteria: a parallel, output-heavy opus run trips EXCEEDED on the
    cost-weighted axis while raw tokens still read healthy — end to end through
    the executor's budget gate, not just the manager property."""

    def test_parallel_opus_run_trips_exceeded_while_raw_healthy(self):
        # Default config: ET ceiling derives from total_budget (the flip).
        mgr = BudgetManager(BudgetConfig(total_budget=200_000))
        # Several "parallel" steps, each cheap in raw but opus output-heavy.
        for i in range(5):
            mgr.record_usage(f"step{i}", output_tokens=12_000, model="claude-opus-4-8")
        # Raw used = 60k / 200k = 30% → healthy on the raw axis.
        assert mgr.used == 60_000
        assert mgr.used / mgr.total_budget < 0.5
        # ET = 5 * (5.0 * 4 * 12000) = 1.2M >> 200k ceiling → cost axis governs.
        assert mgr.effective_used > mgr.effective_ceiling
        assert mgr.status.value == "exceeded"

    def test_executor_halts_on_effective_token_overspend(self):
        # Use the kernel-side BudgetManager so its BudgetStatus enum matches
        # the one the executor compares against. The forge-side re-export
        # predates the kernel/forge split and currently has a distinct enum.
        from animus_kernel.budget import BudgetConfig, BudgetManager

        from animus_forge.workflow.executor import WorkflowExecutor
        from animus_forge.workflow.executor_results import ExecutionResult
        from animus_forge.workflow.loader import StepConfig

        mgr = BudgetManager(BudgetConfig(total_budget=200_000))
        mgr.record_usage("prior", output_tokens=15_000, model="claude-opus-4-8")
        # Raw is fine (15k/200k); ET (300k) is over the 200k ceiling.
        assert mgr.used < mgr.total_budget
        assert mgr.status.value == "exceeded"

        executor = WorkflowExecutor.__new__(WorkflowExecutor)
        executor.budget_manager = mgr
        executor.dry_run = False
        executor.memory_manager = None
        executor.feedback_engine = None
        executor.emit_callback = None

        step = StepConfig(id="next", type="claude_code", params={"estimated_tokens": 100})
        result = ExecutionResult(workflow_name="wf-et")
        assert executor._check_budget_exceeded(step, result) is True
        assert "effective-tokens" in result.error


class TestSinglePricingSource:
    """A3: estimate_cost derives from the canonical tier-multiplier table, not a
    separate hardcoded price table — one source of truth shared with ET."""

    def test_cost_tracks_tier_multiplier_ratios(self):
        bm = BudgetManager(BudgetConfig(total_budget=1_000_000))
        sonnet = bm.estimate_cost(1_000_000, "claude-sonnet-4-6")
        opus = bm.estimate_cost(1_000_000, "claude-opus-4-7")
        haiku = bm.estimate_cost(1_000_000, "claude-haiku-4-5")
        # C12: ratios match the multiplier table (opus 5x, haiku 0.3333x sonnet).
        assert opus == pytest.approx(sonnet * 5.0)
        assert haiku == pytest.approx(sonnet * 0.3333)
        # Sonnet base is the single $9/1M rate.
        assert sonnet == pytest.approx(9.0)

    def test_unknown_model_uses_sonnet_tier(self):
        bm = BudgetManager(BudgetConfig(total_budget=1_000_000))
        assert bm.estimate_cost(1_000_000, "some-mystery-model") == pytest.approx(9.0)

    def test_config_multiplier_override_flows_into_cost(self):
        bm = BudgetManager(BudgetConfig(total_budget=1_000_000, model_multipliers={"sonnet": 2.0}))
        # Overriding the tier table moves cost too — proves single source.
        assert bm.estimate_cost(1_000_000, "sonnet") == pytest.approx(18.0)
