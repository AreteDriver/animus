"""Tests for the Effective-Tokens cost metric on BudgetManager.

ET = m × (1.0·I + 0.1·C + 4.0·O). Lifted from GitHub's "Improving Token
Efficiency in GitHub Agentic Workflows" (2026-05). Lets us compare workflows
across model tiers (Haiku / Sonnet / Opus) and input/cache/output mixes on
a single cost axis.
"""

from __future__ import annotations

import pytest

from animus_forge.budget import (
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

    def test_no_breakdown_treats_total_as_output(self):
        # Conservative fallback for legacy callers — whole tokens count as output (4×).
        assert effective_tokens(_rec(tokens=100)) == pytest.approx(400.0)

    def test_model_multiplier_haiku_cheaper_than_sonnet(self):
        haiku = _rec(input_tokens=1000, output_tokens=1000, model="claude-haiku-4-5")
        sonnet = _rec(input_tokens=1000, output_tokens=1000, model="claude-sonnet-4-6")
        opus = _rec(input_tokens=1000, output_tokens=1000, model="claude-opus-4-7")
        assert effective_tokens(haiku) < effective_tokens(sonnet) < effective_tokens(opus)
        # Sonnet base: 1.0 × (1·1000 + 4·1000) = 5000
        assert effective_tokens(sonnet) == pytest.approx(5000.0)
        # Haiku at 0.08× → 400; Opus at 5× → 25000
        assert effective_tokens(haiku) == pytest.approx(400.0)
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
        bm.record_usage("a2", input_tokens=100, output_tokens=100, model="haiku")  # 40
        assert bm.total_effective_tokens() == pytest.approx(540.0)

    def test_effective_tokens_by_agent_groups_correctly(self):
        bm = BudgetManager(BudgetConfig(total_budget=10_000))
        bm.record_usage("a1", input_tokens=100, output_tokens=100, model="sonnet")  # 500
        bm.record_usage("a1", input_tokens=50, output_tokens=50, model="sonnet")  # 250
        bm.record_usage("a2", input_tokens=100, output_tokens=100, model="haiku")  # 40
        by_agent = bm.effective_tokens_by_agent()
        assert by_agent["a1"] == pytest.approx(750.0)
        assert by_agent["a2"] == pytest.approx(40.0)

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
