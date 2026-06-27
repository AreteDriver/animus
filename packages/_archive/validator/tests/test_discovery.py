"""Tests for opportunity discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from animus_validator.config import DiscoveryConfig, ValidatorConfig
from animus_validator.discovery import OpportunityDiscovery, StakingOpportunity
from animus_validator.models import Network, NetworkInfo
from animus_validator.networks.base import BaseNetwork


def _make_mock_network(
    network: Network = Network.SUI,
    validators: list[dict] | None = None,
    info: NetworkInfo | None = None,
) -> BaseNetwork:
    mock = AsyncMock(spec=BaseNetwork)
    mock.network = network
    mock.get_validators = AsyncMock(return_value=validators or [])
    mock.info = info or NetworkInfo(
        network=network,
        display_name=network.value.title(),
        native_token=network.value.upper()[:3],
        avg_apy_pct=5.0,
        min_stake=1.0,
    )
    return mock


class TestStakingOpportunity:
    def test_net_apy(self):
        opp = StakingOpportunity(
            network=Network.SUI,
            validator_address="0xv",
            name="Test",
            apy_pct=10.0,
            commission_pct=10.0,
        )
        assert opp.net_apy == 9.0  # 10% * (1 - 0.1)

    def test_net_apy_zero_commission(self):
        opp = StakingOpportunity(
            network=Network.SUI,
            validator_address="0xv",
            name="Test",
            apy_pct=5.0,
            commission_pct=0.0,
        )
        assert opp.net_apy == 5.0

    def test_to_dict(self):
        opp = StakingOpportunity(
            network=Network.ETHEREUM,
            validator_address="lido",
            name="Lido Finance",
            apy_pct=3.8,
            commission_pct=10.0,
            total_staked=1000000.0,
            is_testnet=False,
            url="https://lido.fi",
            tags=["liquid_staking"],
        )
        d = opp.to_dict()
        assert d["network"] == "ethereum"
        assert d["name"] == "Lido Finance"
        assert d["apy_pct"] == 3.8
        assert d["net_apy"] == pytest.approx(3.42)
        assert d["is_testnet"] is False
        assert d["tags"] == ["liquid_staking"]
        assert "discovered_at" in d

    def test_defaults(self):
        opp = StakingOpportunity(
            network=Network.SUI, validator_address="0x1", name="V", apy_pct=5.0
        )
        assert opp.min_stake == 0.0
        assert opp.commission_pct == 0.0
        assert opp.risk_score == 0.0
        assert opp.is_testnet is False
        assert opp.url == ""
        assert opp.tags == []


class TestOpportunityDiscovery:
    @pytest.fixture()
    def config(self, tmp_path):
        return ValidatorConfig(data_dir=tmp_path / "data")

    @pytest.mark.asyncio
    async def test_scan_validators(self, config):
        validators = [
            {"address": "0xv1", "name": "Val 1", "apy_pct": 5.0, "commission_pct": 5.0},
            {"address": "0xv2", "name": "Val 2", "apy_pct": 8.0, "commission_pct": 10.0},
        ]
        mock_net = _make_mock_network(Network.SUI, validators=validators)
        disc = OpportunityDiscovery(config, {Network.SUI: mock_net})

        opps = await disc.scan_validators(Network.SUI)
        assert len(opps) == 2
        assert opps[0].name == "Val 1"
        assert opps[1].name == "Val 2"

    @pytest.mark.asyncio
    async def test_scan_validators_no_adapter(self, config):
        disc = OpportunityDiscovery(config, {})
        opps = await disc.scan_validators(Network.SUI)
        assert opps == []

    @pytest.mark.asyncio
    async def test_scan_validators_error(self, config):
        mock_net = _make_mock_network(Network.SUI)
        mock_net.get_validators = AsyncMock(side_effect=RuntimeError("RPC down"))
        disc = OpportunityDiscovery(config, {Network.SUI: mock_net})

        opps = await disc.scan_validators(Network.SUI)
        assert opps == []

    @pytest.mark.asyncio
    async def test_scan_all(self, config):
        validators = [{"address": "0xv1", "name": "V1", "stake_total": 1000}]
        mock_sui = _make_mock_network(Network.SUI, validators=validators)
        mock_eth = _make_mock_network(Network.ETHEREUM, validators=[])
        disc = OpportunityDiscovery(config, {Network.SUI: mock_sui, Network.ETHEREUM: mock_eth})

        opps = await disc.scan_all()
        assert len(opps) == 1

    @pytest.mark.asyncio
    async def test_scan_all_disabled(self, config):
        config.discovery = DiscoveryConfig(enabled=False)
        disc = OpportunityDiscovery(config, {Network.SUI: _make_mock_network()})
        opps = await disc.scan_all()
        assert opps == []

    def test_filter_opportunities(self, config):
        config.discovery = DiscoveryConfig(min_apy_threshold=3.0, max_risk_score=0.5)
        disc = OpportunityDiscovery(config, {})

        opps = [
            StakingOpportunity(
                network=Network.SUI,
                validator_address="0x1",
                name="Good",
                apy_pct=5.0,
                risk_score=0.1,
            ),
            StakingOpportunity(
                network=Network.SUI,
                validator_address="0x2",
                name="LowAPY",
                apy_pct=1.0,
                risk_score=0.1,
            ),
            StakingOpportunity(
                network=Network.SUI,
                validator_address="0x3",
                name="HighRisk",
                apy_pct=10.0,
                risk_score=0.9,
            ),
        ]
        filtered = disc.filter_opportunities(opps)
        assert len(filtered) == 1
        assert filtered[0].name == "Good"

    def test_filter_sorts_by_net_apy(self, config):
        disc = OpportunityDiscovery(config, {})
        opps = [
            StakingOpportunity(network=Network.SUI, validator_address="0x1", name="A", apy_pct=5.0),
            StakingOpportunity(
                network=Network.SUI, validator_address="0x2", name="B", apy_pct=10.0
            ),
        ]
        filtered = disc.filter_opportunities(opps)
        assert filtered[0].name == "B"
        assert filtered[1].name == "A"

    @pytest.mark.asyncio
    async def test_discover(self, config):
        validators = [
            {"address": "0xv1", "name": "V1", "apy_pct": 5.0},
            {"address": "0xv2", "name": "V2", "apy_pct": 1.0},  # Below threshold
        ]
        mock_net = _make_mock_network(Network.SUI, validators=validators)
        disc = OpportunityDiscovery(config, {Network.SUI: mock_net})

        opps = await disc.discover()
        assert len(opps) == 1
        assert opps[0].name == "V1"

    def test_get_network_comparison(self, config):
        mock_sui = _make_mock_network(
            Network.SUI,
            info=NetworkInfo(
                network=Network.SUI, display_name="Sui", native_token="SUI", avg_apy_pct=3.5
            ),
        )
        mock_eth = _make_mock_network(
            Network.ETHEREUM,
            info=NetworkInfo(
                network=Network.ETHEREUM,
                display_name="Ethereum",
                native_token="ETH",
                avg_apy_pct=3.8,
            ),
        )
        disc = OpportunityDiscovery(config, {Network.SUI: mock_sui, Network.ETHEREUM: mock_eth})
        comparison = disc.get_network_comparison()
        assert len(comparison) == 2
        assert comparison[0]["network"] == "ethereum"  # Higher APY first

    def test_get_network_info(self, config):
        info = NetworkInfo(
            network=Network.SUI, display_name="Sui", native_token="SUI", avg_apy_pct=3.5
        )
        mock_net = _make_mock_network(Network.SUI, info=info)
        disc = OpportunityDiscovery(config, {Network.SUI: mock_net})

        result = disc.get_network_info(Network.SUI)
        assert result is not None
        assert result.display_name == "Sui"

    def test_get_network_info_not_found(self, config):
        disc = OpportunityDiscovery(config, {})
        result = disc.get_network_info(Network.COSMOS)
        assert result is None

    @pytest.mark.asyncio
    async def test_scan_validators_default_apy(self, config):
        """Validators without apy_pct use network avg."""
        validators = [{"address": "0xv1", "name": "V1"}]
        info = NetworkInfo(
            network=Network.SUI, display_name="Sui", native_token="SUI", avg_apy_pct=4.0
        )
        mock_net = _make_mock_network(Network.SUI, validators=validators, info=info)
        disc = OpportunityDiscovery(config, {Network.SUI: mock_net})

        opps = await disc.scan_validators(Network.SUI)
        assert opps[0].apy_pct == 4.0
