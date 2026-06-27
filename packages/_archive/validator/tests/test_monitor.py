"""Tests for node monitoring."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from animus_validator.config import ValidatorConfig
from animus_validator.models import Network, Node, StakeStatus
from animus_validator.monitor import NodeMonitor
from animus_validator.networks.base import BaseNetwork


def _make_mock_network(
    network: Network = Network.SUI,
    node: Node | None = None,
    health: bool = True,
) -> BaseNetwork:
    mock = AsyncMock(spec=BaseNetwork)
    mock.network = network
    mock.get_node_status = AsyncMock(return_value=node)
    mock.health_check = AsyncMock(return_value=health)
    return mock


class TestNodeMonitor:
    @pytest.fixture()
    def config(self, tmp_path):
        return ValidatorConfig(data_dir=tmp_path / "data")

    @pytest.mark.asyncio
    async def test_check_node_healthy(self, config):
        node = Node(
            network=Network.SUI,
            address="0xval",
            stake_amount=1000,
            uptime_pct=99.0,
            status=StakeStatus.ACTIVE,
        )
        mock_net = _make_mock_network(Network.SUI, node)
        monitor = NodeMonitor(config, {Network.SUI: mock_net})

        result = await monitor.check_node(Network.SUI, "0xval")
        assert result is not None
        assert result.address == "0xval"
        assert result.last_check is not None
        assert len(monitor.alerts) == 0

    @pytest.mark.asyncio
    async def test_check_node_low_uptime(self, config):
        node = Node(
            network=Network.SUI,
            address="0xval",
            stake_amount=1000,
            uptime_pct=90.0,
            status=StakeStatus.ACTIVE,
        )
        mock_net = _make_mock_network(Network.SUI, node)
        monitor = NodeMonitor(config, {Network.SUI: mock_net})

        result = await monitor.check_node(Network.SUI, "0xval")
        assert result is not None
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0]["type"] == "low_uptime"

    @pytest.mark.asyncio
    async def test_check_node_not_active(self, config):
        node = Node(
            network=Network.SUI,
            address="0xval",
            stake_amount=1000,
            uptime_pct=99.0,
            status=StakeStatus.UNSTAKING,
        )
        mock_net = _make_mock_network(Network.SUI, node)
        monitor = NodeMonitor(config, {Network.SUI: mock_net})

        result = await monitor.check_node(Network.SUI, "0xval")
        assert result is not None
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0]["type"] == "not_active"

    @pytest.mark.asyncio
    async def test_check_node_not_found(self, config):
        mock_net = _make_mock_network(Network.SUI, node=None)
        monitor = NodeMonitor(config, {Network.SUI: mock_net})

        result = await monitor.check_node(Network.SUI, "0xval")
        assert result is None
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0]["type"] == "node_not_found"

    @pytest.mark.asyncio
    async def test_check_node_no_adapter(self, config):
        monitor = NodeMonitor(config, {})
        result = await monitor.check_node(Network.SUI, "0xval")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_all_nodes(self, config):
        node1 = Node(network=Network.SUI, address="0xv1", stake_amount=100, uptime_pct=99)
        node2 = Node(network=Network.ETHEREUM, address="0xv2", stake_amount=32, uptime_pct=99)
        mock_sui = _make_mock_network(Network.SUI, node1)
        mock_eth = _make_mock_network(Network.ETHEREUM, node2)

        monitor = NodeMonitor(config, {Network.SUI: mock_sui, Network.ETHEREUM: mock_eth})
        results = await monitor.check_all_nodes(
            {
                Network.SUI: ["0xv1"],
                Network.ETHEREUM: ["0xv2"],
            }
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_network_health(self, config):
        mock_sui = _make_mock_network(Network.SUI, health=True)
        mock_eth = _make_mock_network(Network.ETHEREUM, health=False)

        monitor = NodeMonitor(config, {Network.SUI: mock_sui, Network.ETHEREUM: mock_eth})
        health = await monitor.network_health()
        assert health["sui"] is True
        assert health["ethereum"] is False

    def test_clear_alerts(self, config):
        monitor = NodeMonitor(config, {})
        monitor._alerts.append({"type": "test"})
        assert len(monitor.alerts) == 1
        monitor.clear_alerts()
        assert len(monitor.alerts) == 0

    def test_get_unhealthy_nodes(self, config, tmp_path):
        config.ensure_dirs()
        monitor = NodeMonitor(config, {})
        # Record a healthy and an unhealthy node
        monitor.store.record(
            Node(network=Network.SUI, address="0xhealthy", stake_amount=100, uptime_pct=99)
        )
        monitor.store.record(
            Node(network=Network.SUI, address="0xbad", stake_amount=100, uptime_pct=80)
        )
        unhealthy = monitor.get_unhealthy_nodes()
        assert len(unhealthy) == 1
        assert unhealthy[0].address == "0xbad"

    def test_get_node_history(self, config):
        config.ensure_dirs()
        monitor = NodeMonitor(config, {})
        node = Node(network=Network.SUI, address="0xv1", stake_amount=100)
        monitor.store.record(node)
        monitor.store.record(
            Node(network=Network.SUI, address="0xv1", stake_amount=100, uptime_pct=90)
        )
        history = monitor.get_node_history(node.node_id)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_check_node_low_uptime_and_not_active(self, config):
        """Node with both low uptime and non-active status generates two alerts."""
        node = Node(
            network=Network.SUI,
            address="0xval",
            stake_amount=1000,
            uptime_pct=50.0,
            status=StakeStatus.UNSTAKING,
        )
        mock_net = _make_mock_network(Network.SUI, node)
        monitor = NodeMonitor(config, {Network.SUI: mock_net})

        await monitor.check_node(Network.SUI, "0xval")
        assert len(monitor.alerts) == 2
        types = {a["type"] for a in monitor.alerts}
        assert "low_uptime" in types
        assert "not_active" in types

    def test_alerts_property_returns_copy(self, config):
        monitor = NodeMonitor(config, {})
        monitor._alerts.append({"type": "test"})
        alerts = monitor.alerts
        alerts.clear()
        assert len(monitor._alerts) == 1  # Original unchanged
