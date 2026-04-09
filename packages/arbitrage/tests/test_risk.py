"""Tests for risk management."""

from __future__ import annotations

import time

import pytest

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.risk import RiskManager

from .conftest import make_opportunity


class TestRiskManager:
    @pytest.fixture
    def risk(self, config: ArbitrageConfig) -> RiskManager:
        return RiskManager(config)

    def test_init(self, risk: RiskManager):
        assert risk._daily_pnl == 0.0
        assert risk._current_position_usd == 0.0
        assert risk._trades_this_hour == 0

    def test_basic_check_passes(self, risk: RiskManager):
        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is True
        assert reason == "OK"

    def test_position_limit(self, config: ArbitrageConfig):
        config.risk.max_position_usd = 100.0
        risk = RiskManager(config)
        risk._current_position_usd = 80.0

        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Position limit" in reason

    def test_daily_loss_limit(self, config: ArbitrageConfig):
        config.risk.daily_loss_limit_usd = 10.0
        risk = RiskManager(config)
        risk._daily_loss = 10.0

        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Daily loss limit" in reason

    def test_min_spread(self, config: ArbitrageConfig):
        config.risk.min_spread_pct = 2.0
        risk = RiskManager(config)

        opp = make_opportunity(spread_pct=1.0, trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Spread too low" in reason

    def test_min_confidence(self, config: ArbitrageConfig):
        config.risk.min_confidence = 0.9
        risk = RiskManager(config)

        opp = make_opportunity(confidence=0.5, trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Confidence too low" in reason

    def test_hourly_trade_limit(self, config: ArbitrageConfig):
        config.risk.max_trades_per_hour = 2
        risk = RiskManager(config)
        risk._trades_this_hour = 2

        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Hourly trade limit" in reason

    def test_concurrent_trade_limit(self, config: ArbitrageConfig):
        config.risk.max_concurrent_trades = 1
        risk = RiskManager(config)
        risk._active_trades = 1

        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Concurrent limit" in reason

    def test_cooldown(self, config: ArbitrageConfig):
        config.risk.cooldown_seconds = 60.0
        risk = RiskManager(config)
        risk._last_trade_time = time.monotonic()

        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Cooldown" in reason

    def test_cooldown_zero_no_block(self, risk: RiskManager):
        """Zero cooldown should never block."""
        risk._last_trade_time = time.monotonic()
        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is True

    def test_max_trade_size(self, config: ArbitrageConfig):
        config.execution.max_trade_size_usd = 50.0
        risk = RiskManager(config)

        opp = make_opportunity(trade_size=100.0)
        ok, reason = risk.check_trade(opp)
        assert ok is False
        assert "Trade size" in reason

    def test_record_trade(self, risk: RiskManager):
        opp = make_opportunity(trade_size=50.0, profit_after_gas=5.0)
        risk.record_trade(opp)
        assert risk._current_position_usd == 50.0
        assert risk._trades_this_hour == 1
        assert risk._active_trades == 1
        assert risk._daily_pnl == 5.0

    def test_record_trade_complete(self, risk: RiskManager):
        risk._current_position_usd = 100.0
        risk._active_trades = 2
        risk.record_trade_complete(50.0, 5.0)
        assert risk._current_position_usd == 50.0
        assert risk._active_trades == 1

    def test_record_trade_complete_loss(self, risk: RiskManager):
        risk._current_position_usd = 100.0
        risk._active_trades = 1
        risk.record_trade_complete(100.0, -10.0)
        assert risk._current_position_usd == 0.0
        assert risk._active_trades == 0
        assert risk._daily_loss == 10.0

    def test_record_trade_complete_no_negative_position(self, risk: RiskManager):
        risk._current_position_usd = 30.0
        risk._active_trades = 0
        risk.record_trade_complete(50.0, 0.0)
        assert risk._current_position_usd == 0.0
        assert risk._active_trades == 0

    def test_status(self, risk: RiskManager):
        status = risk.status
        assert "current_position_usd" in status
        assert "max_position_usd" in status
        assert "daily_pnl_usd" in status
        assert "daily_loss_usd" in status
        assert "trades_this_hour" in status
        assert "active_trades" in status
        assert "cooldown_seconds" in status

    def test_hourly_reset(self, risk: RiskManager):
        risk._trades_this_hour = 5
        risk._hour_start = time.monotonic() - 3700  # Over an hour ago
        risk._maybe_reset_counters()
        assert risk._trades_this_hour == 0

    def test_daily_reset(self, risk: RiskManager):
        risk._daily_pnl = 100.0
        risk._daily_loss = 20.0
        risk._day_start = time.monotonic() - 86500  # Over a day ago
        risk._maybe_reset_counters()
        assert risk._daily_pnl == 0.0
        assert risk._daily_loss == 0.0

    def test_no_premature_reset(self, risk: RiskManager):
        risk._trades_this_hour = 5
        risk._daily_pnl = 100.0
        risk._maybe_reset_counters()
        assert risk._trades_this_hour == 5
        assert risk._daily_pnl == 100.0

    def test_cooldown_first_trade_not_blocked(self, config: ArbitrageConfig):
        """First trade should never be cooldown-blocked."""
        config.risk.cooldown_seconds = 60.0
        risk = RiskManager(config)
        # _last_trade_time == 0.0 by default
        opp = make_opportunity(trade_size=50.0)
        ok, reason = risk.check_trade(opp)
        assert ok is True
