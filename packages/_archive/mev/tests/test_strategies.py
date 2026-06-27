"""Tests for MEV strategies."""

from __future__ import annotations

import pytest

from animus_mev.models import Chain, MempoolTx, MEVType
from animus_mev.strategies.backrun import (
    BackrunStrategy,
    _calculate_confidence,
    _estimate_gas_cost,
    _estimate_price_impact,
)
from animus_mev.strategies.sandwich import (
    SandwichDetector,
    _assess_vulnerability,
    _estimate_sandwich_gas,
    _estimate_sandwich_profit,
)


class TestBackrunStrategy:
    @pytest.fixture
    def strategy(self, config):
        return BackrunStrategy(config)

    def test_name(self, strategy):
        assert strategy.name == "backrun"

    def test_is_applicable_dex_swap(self, strategy, sample_tx):
        # sample_tx is 5 ETH ($15k) to a known DEX router
        assert strategy.is_applicable(sample_tx) is True

    def test_is_applicable_not_dex(self, strategy, small_tx):
        assert strategy.is_applicable(small_tx) is False

    def test_is_applicable_below_threshold(self, strategy):
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(0.01e18),  # ~$30 — below $1000 default
            gas_price=int(0.01e9),
            input_data="0x04e45aaf" + "0" * 128,
        )
        assert strategy.is_applicable(tx) is False

    def test_is_applicable_unknown_chain(self, strategy):
        tx = MempoolTx(
            chain=Chain.ETH,  # ETH not in default config
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",
            value=int(10e18),
            gas_price=int(30e9),
            input_data="0x04e45aaf" + "0" * 128,
        )
        # ETH not in default config chains
        assert strategy.is_applicable(tx) is False

    @pytest.mark.asyncio
    async def test_evaluate_profitable(self, strategy, sample_tx):
        opp = await strategy.evaluate(sample_tx)
        assert opp is not None
        assert opp.mev_type == MEVType.BACKRUN
        assert opp.chain == Chain.BASE
        assert opp.estimated_profit_usd > 0
        assert opp.confidence > 0

    @pytest.mark.asyncio
    async def test_evaluate_not_applicable(self, strategy, small_tx):
        opp = await strategy.evaluate(small_tx)
        assert opp is None

    @pytest.mark.asyncio
    async def test_evaluate_low_confidence(self, config):
        config.risk.min_confidence = 0.99  # Unreachable
        strategy = BackrunStrategy(config)
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(1e18),  # $3k
            gas_price=int(0.01e9),
            input_data="0x" + "0" * 128,  # No known selector
        )
        opp = await strategy.evaluate(tx)
        assert opp is None


class TestBackrunHelpers:
    def test_price_impact_small(self):
        assert _estimate_price_impact(1000) == 0.001

    def test_price_impact_medium(self):
        assert _estimate_price_impact(10000) == 0.003

    def test_price_impact_large(self):
        assert _estimate_price_impact(100000) == 0.008

    def test_price_impact_whale(self):
        assert _estimate_price_impact(1000000) == 0.015

    def test_gas_cost_with_price(self):
        cost = _estimate_gas_cost(Chain.BASE, int(0.01e9))
        assert cost > 0

    def test_gas_cost_default(self):
        cost = _estimate_gas_cost(Chain.BASE, 0)
        assert cost > 0

    def test_gas_cost_eth_more_expensive(self):
        base_cost = _estimate_gas_cost(Chain.BASE, 0)
        eth_cost = _estimate_gas_cost(Chain.ETH, 0)
        assert eth_cost > base_cost

    def test_confidence_with_selector(self, sample_tx):
        conf = _calculate_confidence(sample_tx, 15000.0)
        assert conf >= 0.5
        assert conf <= 0.95

    def test_confidence_high_for_large_swap(self, sample_tx):
        conf = _calculate_confidence(sample_tx, 100000.0)
        assert conf >= 0.7

    def test_confidence_capped(self, sample_tx):
        conf = _calculate_confidence(sample_tx, 1000000.0)
        assert conf <= 0.95


class TestSandwichDetector:
    @pytest.fixture
    def detector(self, config):
        return SandwichDetector(config)

    def test_name(self, detector):
        assert detector.name == "sandwich_detector"

    def test_is_applicable_dex(self, detector, sample_tx):
        assert detector.is_applicable(sample_tx) is True

    def test_is_applicable_not_dex(self, detector, small_tx):
        assert detector.is_applicable(small_tx) is False

    @pytest.mark.asyncio
    async def test_evaluate_detects_sandwich(self, detector, sample_tx):
        opp = await detector.evaluate(sample_tx)
        # May or may not detect depending on vulnerability score
        if opp is not None:
            assert opp.mev_type == MEVType.SANDWICH_DETECT
            assert opp.metadata.get("monitoring_only") is True
            assert opp.metadata.get("reason") == "sandwich_detection_monitor"

    @pytest.mark.asyncio
    async def test_evaluate_not_applicable(self, detector, small_tx):
        opp = await detector.evaluate(small_tx)
        assert opp is None

    @pytest.mark.asyncio
    async def test_evaluate_always_sandwich_type(self, detector):
        """Sandwich detector MUST always produce SANDWICH_DETECT type."""
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0xlarge",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(100e18),  # $300k — very vulnerable
            gas_price=int(0.001e9),  # Low gas = easy to frontrun
            input_data="0x04e45aaf" + "0" * 128,
        )
        opp = await detector.evaluate(tx)
        if opp is not None:
            assert opp.mev_type == MEVType.SANDWICH_DETECT

    @pytest.mark.asyncio
    async def test_recent_txs_window(self, detector, sample_tx):
        for _ in range(60):
            await detector.evaluate(sample_tx)
        assert len(detector._recent_txs) <= detector._max_window


class TestSandwichHelpers:
    def test_vulnerability_large_swap(self, sample_tx):
        score = _assess_vulnerability(sample_tx)
        assert score > 0

    def test_vulnerability_score_capped(self):
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(1000e18),
            gas_price=1,  # Very low gas
            input_data="0x04e45aaf" + "0" * 128,
        )
        score = _assess_vulnerability(tx)
        assert score <= 0.95

    def test_vulnerability_low_gas_higher(self, sample_tx):
        low_gas_tx = MempoolTx(
            chain=sample_tx.chain,
            tx_hash=sample_tx.tx_hash,
            from_addr=sample_tx.from_addr,
            to_addr=sample_tx.to_addr,
            value=sample_tx.value,
            gas_price=int(1e9),  # 1 gwei
            input_data=sample_tx.input_data,
        )
        high_gas_tx = MempoolTx(
            chain=sample_tx.chain,
            tx_hash=sample_tx.tx_hash,
            from_addr=sample_tx.from_addr,
            to_addr=sample_tx.to_addr,
            value=sample_tx.value,
            gas_price=int(100e9),  # 100 gwei
            input_data=sample_tx.input_data,
        )
        low_score = _assess_vulnerability(low_gas_tx)
        high_score = _assess_vulnerability(high_gas_tx)
        assert low_score >= high_score

    def test_sandwich_profit_estimate(self, sample_tx):
        profit = _estimate_sandwich_profit(sample_tx)
        # 1% of swap amount
        assert profit == sample_tx.swap_amount_usd * 0.01

    def test_sandwich_gas_estimate(self, sample_tx):
        gas = _estimate_sandwich_gas(sample_tx)
        assert gas > 0

    def test_sandwich_gas_with_zero_gas_price(self):
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x2",
            value=0,
            gas_price=0,
            input_data="0x",
        )
        gas = _estimate_sandwich_gas(tx)
        assert gas > 0
