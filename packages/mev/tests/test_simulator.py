"""Tests for MEV trade simulator."""

from __future__ import annotations

import pytest

from animus_mev.models import Chain, MEVOpportunity, MEVType
from animus_mev.simulator import Simulator, _estimate_revert_risk, _refine_gas_estimate


class TestSimulator:
    @pytest.fixture
    def simulator(self, config):
        return Simulator(config)

    def test_create(self, simulator):
        assert simulator.simulation_count == 0

    @pytest.mark.asyncio
    async def test_simulate_profitable(self, simulator, sample_opportunity):
        result = await simulator.simulate(sample_opportunity)
        assert result.opportunity_id == sample_opportunity.opportunity_id
        assert result.profitable is True
        assert result.net_profit_usd > 0
        assert result.revert_risk >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_simulate_increments_count(self, simulator, sample_opportunity):
        assert simulator.simulation_count == 0
        await simulator.simulate(sample_opportunity)
        assert simulator.simulation_count == 1
        await simulator.simulate(sample_opportunity)
        assert simulator.simulation_count == 2

    @pytest.mark.asyncio
    async def test_simulate_sandwich_blocked(self, simulator, sandwich_opportunity):
        result = await simulator.simulate(sandwich_opportunity)
        assert result.profitable is False
        assert result.revert_risk == 1.0
        assert result.error == "sandwich_detection_only"

    @pytest.mark.asyncio
    async def test_simulate_unprofitable(self, simulator, sample_tx):
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=0.001,  # Very small
            gas_cost_usd=0.10,
            confidence=0.8,
        )
        result = await simulator.simulate(opp)
        assert result.profitable is False

    @pytest.mark.asyncio
    async def test_simulate_result_dict(self, simulator, sample_opportunity):
        result = await simulator.simulate(sample_opportunity)
        d = result.to_dict()
        assert "opportunity_id" in d
        assert "profitable" in d
        assert "revert_risk" in d


class TestRevertRisk:
    def test_backrun_lower_risk(self, sample_opportunity):
        risk = _estimate_revert_risk(sample_opportunity)
        assert risk < 0.5  # Backrun on L2 should be low risk

    def test_arbitrage_higher_risk(self, sample_tx):
        opp = MEVOpportunity(
            mev_type=MEVType.ARBITRAGE,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.8,
        )
        risk = _estimate_revert_risk(opp)
        assert risk > 0

    def test_eth_higher_risk(self, sample_tx):
        base_opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.8,
        )
        eth_opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.ETH,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.8,
        )
        base_risk = _estimate_revert_risk(base_opp)
        eth_risk = _estimate_revert_risk(eth_opp)
        assert eth_risk > base_risk

    def test_low_confidence_higher_risk(self, sample_tx):
        high_conf = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.95,
        )
        low_conf = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.3,
        )
        assert _estimate_revert_risk(low_conf) > _estimate_revert_risk(high_conf)

    def test_risk_bounded(self, sample_opportunity):
        risk = _estimate_revert_risk(sample_opportunity)
        assert 0.0 <= risk <= 0.95


class TestRefineGasEstimate:
    def test_adds_buffer(self, sample_opportunity):
        refined = _refine_gas_estimate(sample_opportunity)
        assert refined == sample_opportunity.gas_cost_usd * 1.2

    def test_zero_gas(self, sample_tx):
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=1.0,
            gas_cost_usd=0.0,
            confidence=0.8,
        )
        assert _refine_gas_estimate(opp) == 0.0
