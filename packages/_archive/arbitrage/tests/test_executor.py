"""Tests for trade executor."""

from __future__ import annotations

import pytest

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.executor import ExecutionBlockedError, TradeExecutor
from animus_arbitrage.models import ExecutionMode
from animus_arbitrage.risk import RiskManager

from .conftest import make_opportunity


class TestTradeExecutor:
    @pytest.fixture
    def executor(self, config: ArbitrageConfig) -> TradeExecutor:
        return TradeExecutor(config)

    @pytest.fixture
    def live_executor(self, live_config: ArbitrageConfig) -> TradeExecutor:
        return TradeExecutor(live_config)

    def test_default_mode_dry_run(self, executor: TradeExecutor):
        assert executor.mode == ExecutionMode.DRY_RUN
        assert executor.is_live is False

    def test_live_mode(self, live_executor: TradeExecutor):
        assert live_executor.mode == ExecutionMode.LIVE
        assert live_executor.is_live is True

    @pytest.mark.asyncio
    async def test_dry_run_execution(self, executor: TradeExecutor):
        opp = make_opportunity(spread_pct=1.0, profit_after_gas=5.0, trade_size=50.0)
        result = await executor.execute(opp)
        assert result.executed is False
        assert result.mode == ExecutionMode.DRY_RUN
        assert result.error is None
        assert result.actual_profit_usd == 5.0

    @pytest.mark.asyncio
    async def test_dry_run_risk_blocked(self, config: ArbitrageConfig):
        config.risk.min_spread_pct = 5.0  # Very high threshold
        executor = TradeExecutor(config)
        opp = make_opportunity(spread_pct=1.0, confidence=0.01, trade_size=50.0)
        result = await executor.execute(opp)
        assert result.executed is False
        assert "Risk blocked" in (result.error or "")

    @pytest.mark.asyncio
    async def test_live_risk_blocked_raises(self, live_config: ArbitrageConfig):
        live_config.risk.min_spread_pct = 5.0
        executor = TradeExecutor(live_config)
        opp = make_opportunity(spread_pct=1.0, confidence=0.01, trade_size=50.0)
        with pytest.raises(ExecutionBlockedError, match="Risk check failed"):
            await executor.execute(opp)

    @pytest.mark.asyncio
    async def test_expired_opportunity(self, executor: TradeExecutor):
        opp = make_opportunity(expired=True, trade_size=50.0)
        result = await executor.execute(opp)
        assert result.executed is False
        assert result.error == "Opportunity expired"

    @pytest.mark.asyncio
    async def test_live_execute_no_adapter(self, live_executor: TradeExecutor):
        opp = make_opportunity(spread_pct=1.0, profit_after_gas=5.0, trade_size=50.0)
        result = await live_executor.execute(opp)
        assert result.executed is False
        assert result.mode == ExecutionMode.LIVE
        assert "No chain adapter" in (result.error or "")

    @pytest.mark.asyncio
    async def test_dry_run_records_trade(self, config: ArbitrageConfig):
        risk = RiskManager(config)
        executor = TradeExecutor(config, risk)
        opp = make_opportunity(trade_size=50.0)
        await executor.execute(opp)
        assert risk._trades_this_hour == 1
        assert risk._current_position_usd == 50.0

    @pytest.mark.asyncio
    async def test_live_execute_records_trade(self, live_config: ArbitrageConfig):
        risk = RiskManager(live_config)
        executor = TradeExecutor(live_config, risk)
        opp = make_opportunity(trade_size=50.0)
        await executor.execute(opp)
        assert risk._trades_this_hour == 1

    @pytest.mark.asyncio
    async def test_executor_uses_custom_risk_manager(self, config: ArbitrageConfig):
        risk = RiskManager(config)
        executor = TradeExecutor(config, risk)
        assert executor.risk is risk

    @pytest.mark.asyncio
    async def test_executor_creates_default_risk_manager(self, config: ArbitrageConfig):
        executor = TradeExecutor(config)
        assert executor.risk is not None

    @pytest.mark.asyncio
    async def test_opportunity_no_expiry(self, executor: TradeExecutor):
        """Opportunity with no expiry should not be blocked."""
        opp = make_opportunity(trade_size=50.0)
        opp.expires_at = None
        result = await executor.execute(opp)
        assert result.error is None or "expired" not in result.error
