"""Tests for MEV risk management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_mev.models import Chain, MEVOpportunity, MEVType
from animus_mev.risk import RiskManager


class TestRiskManager:
    @pytest.fixture
    def risk(self, config):
        return RiskManager(config)

    def test_initial_state(self, risk):
        assert risk.daily_gas_spent == 0.0
        assert risk.active_txs == 0

    def test_daily_gas_remaining(self, risk):
        assert risk.daily_gas_remaining == risk.config.risk.daily_gas_budget_usd

    def test_check_approved(self, risk, sample_opportunity):
        result = risk.check(sample_opportunity)
        assert result.approved is True
        assert result.reason == "all_checks_passed"
        assert len(result.checks_passed) > 0
        assert len(result.checks_failed) == 0

    def test_check_sandwich_blocked(self, risk, sandwich_opportunity):
        result = risk.check(sandwich_opportunity)
        assert result.approved is False
        assert "sandwich" in result.reason
        assert "sandwich_execution_blocked" in result.checks_failed

    def test_check_low_confidence(self, config, sample_tx):
        config.risk.min_confidence = 0.99
        risk = RiskManager(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=5.0,
            gas_cost_usd=0.1,
            confidence=0.5,
        )
        result = risk.check(opp)
        assert result.approved is False
        assert any("confidence" in f for f in result.checks_failed)

    def test_check_low_profit(self, config, sample_tx):
        config.risk.min_profit_usd = 100.0
        risk = RiskManager(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=1.0,
            gas_cost_usd=0.1,
            confidence=0.8,
        )
        result = risk.check(opp)
        assert result.approved is False
        assert any("profit" in f for f in result.checks_failed)

    def test_check_gas_too_high(self, config, sample_tx):
        config.risk.max_gas_per_tx_usd = 0.01
        risk = RiskManager(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=5.0,
            confidence=0.8,
        )
        result = risk.check(opp)
        assert result.approved is False
        assert any("gas_too_high" in f for f in result.checks_failed)

    def test_check_daily_budget(self, config, sample_tx):
        risk = RiskManager(config)
        # Spend most of the budget
        risk.record_gas_spend(config.risk.daily_gas_budget_usd - 0.01)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=5.0,  # Over remaining budget
            confidence=0.8,
        )
        result = risk.check(opp)
        assert result.approved is False
        assert any("budget" in f for f in result.checks_failed)

    def test_check_concurrent_limit(self, config, sample_tx):
        risk = RiskManager(config)
        for _ in range(config.risk.max_concurrent_txs):
            risk.increment_active()
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=0.1,
            confidence=0.8,
        )
        result = risk.check(opp)
        assert result.approved is False
        assert any("concurrent" in f for f in result.checks_failed)

    def test_record_gas_spend(self, risk):
        risk.record_gas_spend(1.5)
        assert risk.daily_gas_spent == 1.5
        risk.record_gas_spend(2.5)
        assert risk.daily_gas_spent == 4.0

    def test_increment_decrement_active(self, risk):
        risk.increment_active()
        assert risk.active_txs == 1
        risk.increment_active()
        assert risk.active_txs == 2
        risk.decrement_active()
        assert risk.active_txs == 1

    def test_decrement_active_floor(self, risk):
        risk.decrement_active()
        assert risk.active_txs == 0

    def test_daily_reset(self, risk):
        risk.record_gas_spend(10.0)
        assert risk.daily_gas_spent == 10.0
        # Force reset by moving the reset time back
        risk._daily_reset = datetime.now(UTC) - timedelta(hours=25)
        assert risk.daily_gas_spent == 0.0

    def test_get_status(self, risk):
        status = risk.get_status()
        assert "daily_gas_spent_usd" in status
        assert "daily_gas_remaining_usd" in status
        assert "active_txs" in status
        assert status["sandwich_blocked"] is True

    def test_multiple_failures_first_reported(self, config, sample_tx):
        config.risk.min_confidence = 0.99
        config.risk.min_profit_usd = 1000.0
        risk = RiskManager(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=1.0,
            gas_cost_usd=0.1,
            confidence=0.5,
        )
        result = risk.check(opp)
        assert result.approved is False
        assert len(result.checks_failed) >= 2
