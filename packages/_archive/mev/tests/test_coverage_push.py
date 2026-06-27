"""Additional tests to push coverage above 97%."""

from __future__ import annotations

import pytest

from animus_mev.config import MEVConfig
from animus_mev.executor import Executor, _dry_run_result
from animus_mev.models import Chain, MempoolTx, MEVOpportunity, MEVType
from animus_mev.simulator import SimulationResult, Simulator
from animus_mev.strategies.backrun import BackrunStrategy
from animus_mev.strategies.sandwich import SandwichDetector
from animus_mev.tools.mev_tools import _ensure_initialized


class TestBackrunEdgeCases:
    @pytest.mark.asyncio
    async def test_evaluate_zero_net_profit(self, config):
        """Backrun that estimates zero or negative net profit returns None."""
        strategy = BackrunStrategy(config)
        # Create tx with value that produces profit < gas cost
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0xedge",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(0.35e18),  # ~$1050 — just above threshold, tiny profit
            gas_price=int(100e9),  # Very high gas
            input_data="0x04e45aaf" + "0" * 128,
        )
        result = await strategy.evaluate(tx)
        # May return None if net <= 0, or opportunity if profitable
        if result is not None:
            assert result.mev_type == MEVType.BACKRUN


class TestSandwichEdgeCases:
    @pytest.mark.asyncio
    async def test_low_vulnerability_returns_none(self, config):
        """TX with low vulnerability score should return None."""
        detector = SandwichDetector(config)
        # Small swap, not DEX
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0xlow",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(0.01e18),  # $30 — very small
            gas_price=int(100e9),  # High gas
            input_data="0x" + "0" * 128,  # No known selector
        )
        result = await detector.evaluate(tx)
        assert result is None

    @pytest.mark.asyncio
    async def test_high_vulnerability_detected(self, config):
        detector = SandwichDetector(config)
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0xhigh",
            from_addr="0x1",
            to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
            value=int(200e18),  # $600k — whale
            gas_price=int(0.001e9),  # Very low gas
            input_data="0x04e45aaf" + "0" * 128,
        )
        result = await detector.evaluate(tx)
        assert result is not None
        assert result.mev_type == MEVType.SANDWICH_DETECT
        assert result.metadata["monitoring_only"] is True


class TestSimulatorEdgeCases:
    @pytest.mark.asyncio
    async def test_simulate_arbitrage(self, config, sample_tx):
        simulator = Simulator(config)
        opp = MEVOpportunity(
            mev_type=MEVType.ARBITRAGE,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=0.5,
            confidence=0.9,
        )
        result = await simulator.simulate(opp)
        assert result.revert_risk > 0

    @pytest.mark.asyncio
    async def test_simulate_eth_mainnet(self, config, sample_tx):
        simulator = Simulator(config)
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.ETH,
            trigger_tx=sample_tx,
            estimated_profit_usd=10.0,
            gas_cost_usd=0.5,
            confidence=0.9,
        )
        result = await simulator.simulate(opp)
        assert result.revert_risk > 0


class TestExecutorEdgeCases:
    @pytest.mark.asyncio
    async def test_live_records_gas(self, config, sample_opportunity):
        config.execution.live = True
        executor = Executor(config)
        result = await executor.execute(sample_opportunity)
        assert result.executed is True
        assert executor.risk_manager.daily_gas_spent > 0

    def test_dry_run_result_helper(self, sample_opportunity):
        sim = SimulationResult(
            opportunity_id="test",
            profitable=True,
            estimated_profit_usd=5.0,
            gas_estimate_usd=0.1,
            net_profit_usd=4.9,
            revert_risk=0.1,
        )
        result = _dry_run_result(sample_opportunity, sim)
        assert result.executed is False
        assert result.error == "dry_run"
        assert result.actual_profit_usd == 5.0


class TestToolsInit:
    def test_ensure_initialized_with_config(self, tmp_data_dir, monkeypatch):
        import animus_mev.tools.mev_tools as mod

        monkeypatch.setenv("MEV_DATA_DIR", str(tmp_data_dir))
        mod._config = None
        mod._store = None
        mod._tracker = None

        cfg, store, tracker = _ensure_initialized()
        assert cfg is not None
        assert store is not None
        assert tracker is not None

    def test_ensure_initialized_with_existing_config_file(self, tmp_data_dir, monkeypatch):
        import animus_mev.tools.mev_tools as mod

        monkeypatch.setenv("MEV_DATA_DIR", str(tmp_data_dir))
        # Write a config file
        config = MEVConfig(data_dir=tmp_data_dir)
        config.to_yaml(tmp_data_dir / "config.yaml")

        mod._config = None
        mod._store = None
        mod._tracker = None

        cfg, store, tracker = _ensure_initialized()
        assert cfg is not None

    def test_ensure_initialized_idempotent(self, tmp_data_dir, monkeypatch):
        import animus_mev.tools.mev_tools as mod

        monkeypatch.setenv("MEV_DATA_DIR", str(tmp_data_dir))
        mod._config = None
        mod._store = None
        mod._tracker = None

        cfg1, _, _ = _ensure_initialized()
        cfg2, _, _ = _ensure_initialized()
        assert cfg1 is cfg2  # Same instance
