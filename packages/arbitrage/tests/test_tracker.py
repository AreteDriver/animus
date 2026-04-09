"""Tests for P&L tracker."""

from __future__ import annotations

import pytest

from animus_arbitrage.models import ExecutionMode, TradeResult
from animus_arbitrage.store import ArbitrageStore
from animus_arbitrage.tracker import PnLTracker

from .conftest import make_opportunity, make_result


class TestPnLTracker:
    @pytest.fixture
    def tracker(self, store: ArbitrageStore) -> PnLTracker:
        return PnLTracker(store)

    def test_empty_summary(self, tracker: PnLTracker):
        s = tracker.get_summary()
        assert s["total_trades"] == 0
        assert s["total_profit_usd"] == 0.0
        assert s["success_rate"] == 0.0

    def test_record_and_summary(self, tracker: PnLTracker):
        tracker.record_result(make_result(profit=5.0))
        tracker.record_result(make_result(profit=3.0))
        s = tracker.get_summary()
        assert s["total_trades"] == 2
        assert s["total_profit_usd"] == 8.0
        assert s["avg_profit_usd"] == 4.0

    def test_summary_with_executed(self, tracker: PnLTracker):
        tracker.record_result(make_result(executed=True, profit=10.0))
        tracker.record_result(make_result(executed=False, profit=5.0))
        s = tracker.get_summary()
        assert s["executed_trades"] == 1
        assert s["dry_run_trades"] == 1

    def test_summary_success_rate(self, tracker: PnLTracker):
        tracker.record_result(make_result(executed=True, profit=10.0))
        tracker.record_result(make_result(executed=True, profit=5.0, error="fail"))
        s = tracker.get_summary()
        assert s["success_rate"] == 0.5

    def test_summary_best_worst(self, tracker: PnLTracker):
        tracker.record_result(make_result(profit=10.0))
        tracker.record_result(make_result(profit=-2.0))
        tracker.record_result(make_result(profit=5.0))
        s = tracker.get_summary()
        assert s["best_trade_usd"] == 10.0
        assert s["worst_trade_usd"] == -2.0

    def test_summary_slippage(self, tracker: PnLTracker):
        r1 = TradeResult(
            opportunity_id="a",
            executed=True,
            mode=ExecutionMode.LIVE,
            actual_profit_usd=5.0,
            slippage_pct=0.2,
        )
        r2 = TradeResult(
            opportunity_id="b",
            executed=True,
            mode=ExecutionMode.LIVE,
            actual_profit_usd=3.0,
            slippage_pct=0.4,
        )
        tracker.record_result(r1)
        tracker.record_result(r2)
        s = tracker.get_summary()
        assert s["avg_slippage_pct"] == pytest.approx(0.3)

    def test_get_recent(self, tracker: PnLTracker):
        for i in range(5):
            tracker.record_result(make_result(opportunity_id=f"opp_{i}", profit=float(i)))
        recent = tracker.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_recent_default_limit(self, tracker: PnLTracker):
        for i in range(25):
            tracker.record_result(make_result(opportunity_id=f"opp_{i}"))
        recent = tracker.get_recent()
        assert len(recent) == 20

    def test_total_profit(self, tracker: PnLTracker):
        tracker.record_result(make_result(profit=5.0))
        tracker.record_result(make_result(profit=-2.0))
        assert tracker.total_profit() == 3.0

    def test_total_profit_empty(self, tracker: PnLTracker):
        assert tracker.total_profit() == 0.0

    def test_total_opportunities_detected(self, tracker: PnLTracker, store: ArbitrageStore):
        opp = make_opportunity()
        store.record_opportunity(opp)
        assert tracker.total_opportunities_detected() == 1

    def test_format_report(self, tracker: PnLTracker):
        tracker.record_result(make_result(profit=5.0))
        report = tracker.format_report()
        assert "Arbitrage P&L Report" in report
        assert "Total Profit" in report
        assert "Total Trades: 1" in report

    def test_format_report_empty(self, tracker: PnLTracker):
        report = tracker.format_report()
        assert "Total Trades: 0" in report

    def test_get_daily_summary(self, tracker: PnLTracker):
        tracker.record_result(make_result(profit=5.0))
        tracker.record_result(make_result(profit=3.0))
        daily = tracker.get_daily_summary()
        assert len(daily) >= 1
        for _day, stats in daily.items():
            assert "trades" in stats
            assert "profit_usd" in stats

    def test_get_daily_summary_empty(self, tracker: PnLTracker):
        daily = tracker.get_daily_summary()
        assert daily == {}

    def test_summary_opportunities_count(self, tracker: PnLTracker, store: ArbitrageStore):
        opp1 = make_opportunity()
        opp2 = make_opportunity()
        store.record_opportunity(opp1)
        store.record_opportunity(opp2)
        s = tracker.get_summary()
        assert s["total_opportunities"] == 2

    def test_no_slippage_avg_zero(self, tracker: PnLTracker):
        """When all slippages are 0.0, avg should be 0.0."""
        tracker.record_result(make_result(profit=5.0))
        s = tracker.get_summary()
        assert s["avg_slippage_pct"] == 0.0
