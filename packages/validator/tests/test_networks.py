"""Tests for network adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from animus_validator.models import Network, StakeStatus
from animus_validator.networks.ethereum import EthereumNetwork
from animus_validator.networks.solana import SolanaNetwork
from animus_validator.networks.sui import SuiNetwork


class TestSuiNetwork:
    def test_network_property(self):
        net = SuiNetwork()
        assert net.network == Network.SUI

    def test_info_property(self):
        net = SuiNetwork()
        info = net.info
        assert info.network == Network.SUI
        assert info.native_token == "SUI"
        assert info.display_name == "Sui"
        assert info.supports_auto_compound is True

    def test_custom_rpc(self):
        net = SuiNetwork(rpc_url="https://custom.rpc", timeout=30.0)
        assert net.rpc_url == "https://custom.rpc"
        assert net.timeout == 30.0

    @pytest.mark.asyncio
    async def test_get_stake_positions(self):
        net = SuiNetwork()
        mock_response = {
            "result": [
                {
                    "validatorAddress": "0xval1",
                    "stakes": [
                        {
                            "principal": "1000000000",
                            "estimatedReward": "50000000",
                            "status": "Active",
                        },
                        {"principal": "2000000000", "estimatedReward": "0", "status": "Pending"},
                    ],
                }
            ]
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            positions = await net.get_stake_positions("0xaddr")
            assert len(positions) == 2
            assert positions[0].amount == 1.0
            assert positions[0].rewards_pending == 0.05
            assert positions[0].validator_address == "0xval1"
            assert positions[1].amount == 2.0

    @pytest.mark.asyncio
    async def test_get_stake_positions_error(self):
        net = SuiNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            positions = await net.get_stake_positions("0xaddr")
            assert positions == []

    @pytest.mark.asyncio
    async def test_get_rewards(self):
        net = SuiNetwork()
        mock_response = {
            "result": [
                {
                    "validatorAddress": "0xval1",
                    "stakes": [
                        {
                            "principal": "1000000000",
                            "estimatedReward": "50000000",
                            "status": "Active",
                        },
                    ],
                }
            ]
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            rewards = await net.get_rewards("0xaddr")
            assert len(rewards) == 1
            assert rewards[0].amount == 0.05

    @pytest.mark.asyncio
    async def test_get_rewards_none_pending(self):
        net = SuiNetwork()
        mock_response = {
            "result": [
                {
                    "validatorAddress": "0xval1",
                    "stakes": [
                        {"principal": "1000000000", "estimatedReward": "0", "status": "Active"},
                    ],
                }
            ]
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            rewards = await net.get_rewards("0xaddr")
            assert len(rewards) == 0

    @pytest.mark.asyncio
    async def test_get_node_status(self):
        net = SuiNetwork()
        mock_response = {
            "result": {
                "activeValidators": [
                    {
                        "suiAddress": "0xval1",
                        "stakingPoolSuiBalance": "5000000000000",
                        "name": "TestValidator",
                    },
                ]
            }
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            node = await net.get_node_status("0xval1")
            assert node is not None
            assert node.address == "0xval1"
            assert node.name == "TestValidator"
            assert node.stake_amount == 5000.0

    @pytest.mark.asyncio
    async def test_get_node_status_not_found(self):
        net = SuiNetwork()
        mock_response = {"result": {"activeValidators": []}}
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            node = await net.get_node_status("0xnonexistent")
            assert node is None

    @pytest.mark.asyncio
    async def test_get_node_status_error(self):
        net = SuiNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            node = await net.get_node_status("0xval1")
            assert node is None

    @pytest.mark.asyncio
    async def test_compound_rewards(self):
        net = SuiNetwork()
        result = await net.compound_rewards("0xaddr", "0xval")
        assert result is None  # Sui doesn't support auto-compound

    @pytest.mark.asyncio
    async def test_get_validators(self):
        net = SuiNetwork()
        mock_response = {
            "result": {
                "activeValidators": [
                    {
                        "suiAddress": "0xv1",
                        "name": "V1",
                        "stakingPoolSuiBalance": "1000000000000",
                        "commissionRate": "500",
                    },
                ]
            }
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            validators = await net.get_validators()
            assert len(validators) == 1
            assert validators[0]["name"] == "V1"
            assert validators[0]["commission_pct"] == 5.0

    @pytest.mark.asyncio
    async def test_get_validators_error(self):
        net = SuiNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            validators = await net.get_validators()
            assert validators == []

    @pytest.mark.asyncio
    async def test_health_check_ok(self):
        net = SuiNetwork()
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value={"result": 100}):
            assert await net.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_fail(self):
        net = SuiNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            assert await net.health_check() is False


class TestEthereumNetwork:
    def test_network_property(self):
        net = EthereumNetwork()
        assert net.network == Network.ETHEREUM

    def test_info_property(self):
        net = EthereumNetwork()
        info = net.info
        assert info.native_token == "ETH"
        assert info.supports_auto_compound is True

    @pytest.mark.asyncio
    async def test_get_stake_positions(self):
        net = EthereumNetwork()
        with patch.object(net, "_get_eth_balance", new_callable=AsyncMock, return_value=10.5):
            positions = await net.get_stake_positions("0xaddr")
            assert len(positions) == 1
            assert positions[0].amount == 10.5
            assert positions[0].auto_compound is True

    @pytest.mark.asyncio
    async def test_get_stake_positions_zero_balance(self):
        net = EthereumNetwork()
        with patch.object(net, "_get_eth_balance", new_callable=AsyncMock, return_value=0.0):
            positions = await net.get_stake_positions("0xaddr")
            assert positions == []

    @pytest.mark.asyncio
    async def test_get_stake_positions_error(self):
        net = EthereumNetwork()
        with patch.object(
            net, "_get_eth_balance", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            positions = await net.get_stake_positions("0xaddr")
            assert positions == []

    @pytest.mark.asyncio
    async def test_get_rewards(self):
        net = EthereumNetwork()
        with patch.object(net, "_get_eth_balance", new_callable=AsyncMock, return_value=10.0):
            rewards = await net.get_rewards("0xaddr")
            # stETH auto-compounds, rewards_pending is 0
            assert rewards == []

    @pytest.mark.asyncio
    async def test_get_node_status(self):
        net = EthereumNetwork()
        node = await net.get_node_status("lido")
        assert node is not None
        assert node.network == Network.ETHEREUM
        assert node.status == StakeStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_compound_rewards(self):
        net = EthereumNetwork()
        result = await net.compound_rewards("0xaddr", "lido")
        assert result is not None
        assert result.tx_hash == "auto-compound"

    @pytest.mark.asyncio
    async def test_get_validators(self):
        net = EthereumNetwork()
        validators = await net.get_validators()
        assert len(validators) == 2
        assert validators[0]["name"] == "Lido Finance"
        assert validators[1]["name"] == "Rocket Pool"

    @pytest.mark.asyncio
    async def test_health_check_ok(self):
        net = EthereumNetwork()
        with patch.object(net, "_get_eth_balance", new_callable=AsyncMock, return_value=0.0):
            assert await net.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_fail(self):
        net = EthereumNetwork()
        with patch.object(
            net, "_get_eth_balance", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            assert await net.health_check() is False


class TestSolanaNetwork:
    def test_network_property(self):
        net = SolanaNetwork()
        assert net.network == Network.SOLANA

    def test_info_property(self):
        net = SolanaNetwork()
        info = net.info
        assert info.native_token == "SOL"
        assert info.supports_auto_compound is False

    @pytest.mark.asyncio
    async def test_get_stake_positions(self):
        net = SolanaNetwork()
        mock_response = {
            "result": [
                {
                    "account": {
                        "data": {
                            "parsed": {
                                "info": {
                                    "stake": {
                                        "delegation": {
                                            "voter": "0xvoter1",
                                            "stake": "5000000000",
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            ]
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            positions = await net.get_stake_positions("0xaddr")
            assert len(positions) == 1
            assert positions[0].amount == 5.0
            assert positions[0].validator_address == "0xvoter1"

    @pytest.mark.asyncio
    async def test_get_stake_positions_no_delegation(self):
        net = SolanaNetwork()
        mock_response = {
            "result": [{"account": {"data": {"parsed": {"info": {"stake": {"delegation": {}}}}}}}]
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            positions = await net.get_stake_positions("0xaddr")
            assert positions == []

    @pytest.mark.asyncio
    async def test_get_stake_positions_error(self):
        net = SolanaNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            positions = await net.get_stake_positions("0xaddr")
            assert positions == []

    @pytest.mark.asyncio
    async def test_get_rewards(self):
        net = SolanaNetwork()
        mock_response = {"result": [{"amount": 500000000}]}
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            rewards = await net.get_rewards("0xaddr")
            assert len(rewards) == 1
            assert rewards[0].amount == 0.5

    @pytest.mark.asyncio
    async def test_get_rewards_none_entry(self):
        net = SolanaNetwork()
        mock_response = {"result": [None]}
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            rewards = await net.get_rewards("0xaddr")
            assert rewards == []

    @pytest.mark.asyncio
    async def test_get_rewards_zero_amount(self):
        net = SolanaNetwork()
        mock_response = {"result": [{"amount": 0}]}
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            rewards = await net.get_rewards("0xaddr")
            assert rewards == []

    @pytest.mark.asyncio
    async def test_get_rewards_error(self):
        net = SolanaNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            rewards = await net.get_rewards("0xaddr")
            assert rewards == []

    @pytest.mark.asyncio
    async def test_get_node_status_current(self):
        net = SolanaNetwork()
        mock_response = {
            "result": {
                "current": [
                    {"votePubkey": "0xvote1", "activatedStake": "10000000000"},
                ],
                "delinquent": [],
            }
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            node = await net.get_node_status("0xvote1")
            assert node is not None
            assert node.uptime_pct == 100.0
            assert node.stake_amount == 10.0

    @pytest.mark.asyncio
    async def test_get_node_status_delinquent(self):
        net = SolanaNetwork()
        mock_response = {
            "result": {
                "current": [],
                "delinquent": [
                    {"votePubkey": "0xvote1", "activatedStake": "5000000000"},
                ],
            }
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            node = await net.get_node_status("0xvote1")
            assert node is not None
            assert node.uptime_pct == 0.0

    @pytest.mark.asyncio
    async def test_get_node_status_not_found(self):
        net = SolanaNetwork()
        mock_response = {"result": {"current": [], "delinquent": []}}
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            node = await net.get_node_status("0xnonexistent")
            assert node is None

    @pytest.mark.asyncio
    async def test_get_node_status_error(self):
        net = SolanaNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            node = await net.get_node_status("0xvote1")
            assert node is None

    @pytest.mark.asyncio
    async def test_compound_rewards(self):
        net = SolanaNetwork()
        result = await net.compound_rewards("0xaddr", "0xval")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_validators(self):
        net = SolanaNetwork()
        mock_response = {
            "result": {
                "current": [
                    {
                        "votePubkey": "0xv1",
                        "nodePubkey": "0xn1",
                        "activatedStake": "100000000000",
                        "commission": 5,
                        "lastVote": 12345,
                    },
                ],
                "delinquent": [],
            }
        }
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value=mock_response):
            validators = await net.get_validators()
            assert len(validators) == 1
            assert validators[0]["commission_pct"] == 5

    @pytest.mark.asyncio
    async def test_get_validators_error(self):
        net = SolanaNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            validators = await net.get_validators()
            assert validators == []

    @pytest.mark.asyncio
    async def test_health_check_ok(self):
        net = SolanaNetwork()
        with patch.object(net, "_rpc_call", new_callable=AsyncMock, return_value={"result": "ok"}):
            assert await net.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_fail(self):
        net = SolanaNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, side_effect=httpx.HTTPError("fail")
        ):
            assert await net.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_not_ok(self):
        net = SolanaNetwork()
        with patch.object(
            net, "_rpc_call", new_callable=AsyncMock, return_value={"result": "unhealthy"}
        ):
            assert await net.health_check() is False
