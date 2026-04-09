"""Tests for MEV P&L tracker."""

from __future__ import annotations

from animus_mev.models import Chain, ExtractionResult, MEVType
from animus_mev.tracker import PnLSummary, PnLTracker


class TestPnLSummary:
    def test_defaults(self):
        s = PnLSummary()
        assert s.total_opportunities == 0
        assert s.net_profit_usd == 0.0

    def test_net_profit(self):
        s = PnLSummary(total_profit_usd=100.0, total_gas_usd=20.0)
        assert s.net_profit_usd == 80.0

    def test_to_dict(self):
        s = PnLSummary(total_opportunities=5, total_executed=3)
        d = s.to_dict()
        assert d["total_opportunities"] == 5
        assert d["total_executed"] == 3
        assert "net_profit_usd" in d


class TestPnLTracker:
    def test_initial_state(self):
        tracker = PnLTracker()
        assert tracker.results == []
        summary = tracker.get_summary()
        assert summary.total_opportunities == 0

    def test_record_opportunity_seen(self):
        tracker = PnLTracker()
        tracker.record_opportunity_seen()
        tracker.record_opportunity_seen()
        summary = tracker.get_summary()
        assert summary.total_opportunities == 2

    def test_record_result(self):
        tracker = PnLTracker()
        result = ExtractionResult(
            opportunity_id="abc",
            executed=True,
            actual_profit_usd=5.0,
            gas_used_usd=0.5,
        )
        tracker.record_result(result, chain=Chain.BASE, mev_type=MEVType.BACKRUN)
        assert len(tracker.results) == 1

    def test_summary_with_results(self):
        tracker = PnLTracker()
        tracker.record_opportunity_seen()

        r1 = ExtractionResult(
            opportunity_id="a",
            executed=True,
            actual_profit_usd=5.0,
            gas_used_usd=0.5,
        )
        r2 = ExtractionResult(
            opportunity_id="b",
            executed=True,
            actual_profit_usd=3.0,
            gas_used_usd=0.3,
        )
        tracker.record_result(r1, chain=Chain.BASE, mev_type=MEVType.BACKRUN)
        tracker.record_result(r2, chain=Chain.ARB, mev_type=MEVType.ARBITRAGE)

        summary = tracker.get_summary()
        assert summary.total_executed == 2
        assert summary.total_profit_usd == 8.0
        assert summary.total_gas_usd == 0.8
        assert summary.net_profit_usd == 7.2
        assert "base" in summary.by_chain
        assert "arb" in summary.by_chain
        assert "backrun" in summary.by_type

    def test_summary_dry_runs(self):
        tracker = PnLTracker()
        r = ExtractionResult(
            opportunity_id="dry",
            executed=False,
            actual_profit_usd=5.0,
            gas_used_usd=0.1,
            error="dry_run",
        )
        tracker.record_result(r)
        summary = tracker.get_summary()
        assert summary.total_dry_runs == 1
        assert summary.total_executed == 0

    def test_summary_errors(self):
        tracker = PnLTracker()
        r = ExtractionResult(
            opportunity_id="err",
            executed=False,
            error="risk_rejected: confidence_too_low",
        )
        tracker.record_result(r)
        summary = tracker.get_summary()
        assert summary.total_errors == 1

    def test_best_worst_trade(self):
        tracker = PnLTracker()
        r1 = ExtractionResult(
            opportunity_id="a",
            executed=True,
            actual_profit_usd=10.0,
            gas_used_usd=1.0,
        )
        r2 = ExtractionResult(
            opportunity_id="b",
            executed=True,
            actual_profit_usd=0.5,
            gas_used_usd=2.0,
        )
        tracker.record_result(r1, chain=Chain.BASE, mev_type=MEVType.BACKRUN)
        tracker.record_result(r2, chain=Chain.BASE, mev_type=MEVType.BACKRUN)

        summary = tracker.get_summary()
        assert summary.best_trade_usd == 9.0  # 10 - 1
        assert summary.worst_trade_usd == -1.5  # 0.5 - 2

    def test_win_rate(self):
        tracker = PnLTracker()
        r_win = ExtractionResult(
            opportunity_id="w",
            executed=True,
            actual_profit_usd=5.0,
            gas_used_usd=0.1,
        )
        r_loss = ExtractionResult(
            opportunity_id="l",
            executed=True,
            actual_profit_usd=0.1,
            gas_used_usd=1.0,
        )
        tracker.record_result(r_win)
        tracker.record_result(r_loss)
        summary = tracker.get_summary()
        assert summary.win_rate == 0.5

    def test_win_rate_no_executed(self):
        tracker = PnLTracker()
        summary = tracker.get_summary()
        assert summary.win_rate == 0.0

    def test_reset(self):
        tracker = PnLTracker()
        tracker.record_opportunity_seen()
        r = ExtractionResult(opportunity_id="a", executed=True, actual_profit_usd=5.0)
        tracker.record_result(r, chain=Chain.BASE)
        tracker.reset()
        assert tracker.results == []
        summary = tracker.get_summary()
        assert summary.total_opportunities == 0

    def test_record_result_no_chain_type(self):
        tracker = PnLTracker()
        r = ExtractionResult(opportunity_id="a", executed=True)
        tracker.record_result(r)  # No chain or type
        assert len(tracker.results) == 1
