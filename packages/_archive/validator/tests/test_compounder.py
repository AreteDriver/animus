"""Tests for auto-compound logic."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from animus_validator.compounder import RewardCompounder
from animus_validator.config import CompoundConfig, ValidatorConfig
from animus_validator.models import CompoundAction, Network, StakePosition
from animus_validator.networks.base import BaseNetwork


def _make_position(
    network: Network = Network.SUI,
    amount: float = 100.0,
    rewards_pending: float = 1.0,
    auto_compound: bool = True,
) -> StakePosition:
    return StakePosition(
        network=network,
        validator_address="0xval",
        amount=amount,
        rewards_pending=rewards_pending,
        auto_compound=auto_compound,
    )


def _make_mock_network(
    network: Network = Network.SUI,
    compound_result: CompoundAction | None = None,
) -> BaseNetwork:
    mock = AsyncMock(spec=BaseNetwork)
    mock.network = network
    mock.compound_rewards = AsyncMock(return_value=compound_result)
    return mock


class TestRewardCompounder:
    @pytest.fixture()
    def config(self, tmp_path):
        return ValidatorConfig(data_dir=tmp_path / "data")

    def test_should_compound_true(self, config):
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=1.0)
        assert compounder.should_compound(pos, gas_estimate=0.001) is True

    def test_should_compound_disabled(self, config):
        config.compound = CompoundConfig(enabled=False)
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=1.0)
        assert compounder.should_compound(pos) is False

    def test_should_compound_not_auto(self, config):
        compounder = RewardCompounder(config, {})
        pos = _make_position(auto_compound=False)
        assert compounder.should_compound(pos) is False

    def test_should_compound_below_threshold(self, config):
        config.compound = CompoundConfig(min_reward_threshold=10.0)
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=5.0)
        assert compounder.should_compound(pos) is False

    def test_should_compound_below_gas_buffer(self, config):
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=0.01)
        # gas=0.01, buffer=2x, so needs at least 0.02
        assert compounder.should_compound(pos, gas_estimate=0.01) is False

    def test_should_compound_exactly_at_gas_buffer(self, config):
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=0.02)
        assert compounder.should_compound(pos, gas_estimate=0.01) is True

    @pytest.mark.asyncio
    async def test_compound_position_success(self, config):
        action = CompoundAction(network=Network.SUI, amount_compounded=0.0)
        mock_net = _make_mock_network(Network.SUI, compound_result=action)
        compounder = RewardCompounder(config, {Network.SUI: mock_net})

        pos = _make_position(amount=100.0, rewards_pending=1.0)
        result = await compounder.compound_position(pos)

        assert result is not None
        assert result.amount_compounded == 1.0
        assert result.new_total_stake == 101.0

    @pytest.mark.asyncio
    async def test_compound_position_no_adapter(self, config):
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=1.0)
        result = await compounder.compound_position(pos)
        assert result is None

    @pytest.mark.asyncio
    async def test_compound_position_below_threshold(self, config):
        config.compound = CompoundConfig(min_reward_threshold=10.0)
        compounder = RewardCompounder(config, {})
        pos = _make_position(rewards_pending=1.0)
        result = await compounder.compound_position(pos)
        assert result is None

    @pytest.mark.asyncio
    async def test_compound_position_adapter_returns_none(self, config):
        mock_net = _make_mock_network(Network.SUI, compound_result=None)
        compounder = RewardCompounder(config, {Network.SUI: mock_net})
        pos = _make_position(rewards_pending=1.0)
        result = await compounder.compound_position(pos)
        assert result is None

    @pytest.mark.asyncio
    async def test_compound_all(self, config):
        action = CompoundAction(network=Network.SUI, amount_compounded=0.0)
        mock_net = _make_mock_network(Network.SUI, compound_result=action)
        compounder = RewardCompounder(config, {Network.SUI: mock_net})

        positions = [
            _make_position(rewards_pending=1.0),
            _make_position(rewards_pending=0.0001, auto_compound=False),  # Skip
        ]
        results = await compounder.compound_all(positions)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_compound_all_with_gas_estimates(self, config):
        action = CompoundAction(network=Network.SUI, amount_compounded=0.0)
        mock_net = _make_mock_network(Network.SUI, compound_result=action)
        compounder = RewardCompounder(config, {Network.SUI: mock_net})

        positions = [_make_position(rewards_pending=1.0)]
        results = await compounder.compound_all(positions, {Network.SUI: 0.001})
        assert len(results) == 1

    def test_get_compound_history(self, config):
        config.ensure_dirs()
        compounder = RewardCompounder(config, {})
        assert compounder.get_compound_history() == []

    def test_total_compounded(self, config):
        config.ensure_dirs()
        compounder = RewardCompounder(config, {})
        assert compounder.total_compounded() == 0.0

    def test_compound_summary_empty(self, config):
        config.ensure_dirs()
        compounder = RewardCompounder(config, {})
        summary = compounder.compound_summary()
        assert summary["total_actions"] == 0
        assert summary["total_compounded"] == 0.0
        assert summary["by_network"] == {}

    @pytest.mark.asyncio
    async def test_compound_summary_with_data(self, config):
        action = CompoundAction(network=Network.SUI, amount_compounded=0.0)
        mock_net = _make_mock_network(Network.SUI, compound_result=action)
        compounder = RewardCompounder(config, {Network.SUI: mock_net})

        pos = _make_position(rewards_pending=5.0)
        await compounder.compound_position(pos)

        summary = compounder.compound_summary()
        assert summary["total_actions"] == 1
        assert summary["total_compounded"] == 5.0
        assert summary["by_network"]["sui"] == 5.0
