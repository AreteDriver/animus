"""Tests for MEV executor."""

from __future__ import annotations

import pytest

from animus_mev.executor import Executor
from animus_mev.models import Chain, MEVOpportunity, MEVType


class TestExecutor:
    @pytest.fixture
    def executor(self, config):
        return Executor(config)

    def test_default_dry_run(self, executor):
        assert executor.is_live is False

    def test_execution_counts_start_zero(self, executor):
        assert executor.execution_count == 0
        assert executor.dry_run_count == 0

    @pytest.mark.asyncio
    async def test_dry_run_backrun(self, executor, sample_opportunity):
        result = await executor.execute(sample_opportunity)
        assert result.executed is False
        assert result.error == "dry_run"
        assert executor.dry_run_count == 1

    @pytest.mark.asyncio
    async def test_sandwich_always_blocked(self, executor, sandwich_opportunity):
        result = await executor.execute(sandwich_opportunity)
        assert result.executed is False
        assert result.error == "sandwich_execution_blocked"

    @pytest.mark.asyncio
    async def test_sandwich_blocked_even_live(self, config, sandwich_opportunity):
        config.execution.live = True
        executor = Executor(config)
        result = await executor.execute(sandwich_opportunity)
        assert result.executed is False
        assert result.error == "sandwich_execution_blocked"

    @pytest.mark.asyncio
    async def test_risk_rejection(self, config, sample_tx):
        config.risk.min_confidence = 0.99
        executor = Executor(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.5,  # Below 0.99 threshold
        )
        result = await executor.execute(opp)
        assert result.executed is False
        assert "risk_rejected" in result.error

    @pytest.mark.asyncio
    async def test_simulation_unprofitable(self, config, sample_tx):
        executor = Executor(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=0.001,
            gas_cost_usd=0.1,
            confidence=0.8,
        )
        result = await executor.execute(opp)
        assert result.executed is False
        # Risk manager catches low profit before simulation
        assert "risk_rejected" in result.error or "simulation_unprofitable" in result.error

    @pytest.mark.asyncio
    async def test_live_execution(self, config, sample_opportunity):
        config.execution.live = True
        executor = Executor(config)
        result = await executor.execute(sample_opportunity)
        assert result.executed is True
        assert executor.execution_count == 1
        assert result.actual_profit_usd > 0

    @pytest.mark.asyncio
    async def test_dry_run_counts_multiple(self, executor, sample_opportunity):
        await executor.execute(sample_opportunity)
        await executor.execute(sample_opportunity)
        assert executor.dry_run_count == 2

    @pytest.mark.asyncio
    async def test_gas_budget_exceeded(self, config, sample_tx):
        config.risk.daily_gas_budget_usd = 0.01  # Very small budget
        executor = Executor(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=5.0,  # Over budget
            confidence=0.9,
        )
        result = await executor.execute(opp)
        assert result.executed is False
        assert "risk_rejected" in result.error

    @pytest.mark.asyncio
    async def test_max_gas_per_tx(self, config, sample_tx):
        config.risk.max_gas_per_tx_usd = 0.01
        executor = Executor(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=1.0,  # Over per-tx limit
            confidence=0.9,
        )
        result = await executor.execute(opp)
        assert result.executed is False
