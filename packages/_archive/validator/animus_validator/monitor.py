"""Node health monitoring and uptime tracking."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_validator.config import ValidatorConfig
from animus_validator.models import Network, Node, StakeStatus
from animus_validator.networks.base import BaseNetwork
from animus_validator.store import NodeStore

logger = logging.getLogger("animus_validator.monitor")


class NodeMonitor:
    """Monitors validator node health and uptime."""

    def __init__(self, config: ValidatorConfig, networks: dict[Network, BaseNetwork]):
        self.config = config
        self.networks = networks
        self.store = NodeStore(config.data_dir)
        self._alerts: list[dict] = []

    @property
    def alerts(self) -> list[dict]:
        """Get pending alerts."""
        return list(self._alerts)

    def clear_alerts(self) -> None:
        self._alerts.clear()

    async def check_node(self, network: Network, validator_address: str) -> Node | None:
        """Check a single node and record status."""
        adapter = self.networks.get(network)
        if not adapter:
            logger.warning("No adapter for network: %s", network.value)
            return None

        node = await adapter.get_node_status(validator_address)
        if node is None:
            logger.warning("Node not found: %s on %s", validator_address, network.value)
            self._emit_alert("node_not_found", network, validator_address)
            return None

        node.last_check = datetime.now(UTC)
        self.store.record(node)

        # Check for issues
        if node.uptime_pct < self.config.monitor.uptime_alert_threshold:
            self._emit_alert("low_uptime", network, validator_address, uptime=node.uptime_pct)

        if node.status != StakeStatus.ACTIVE:
            self._emit_alert("not_active", network, validator_address, status=node.status.value)

        return node

    async def check_all_nodes(self, addresses: dict[Network, list[str]]) -> list[Node]:
        """Check all tracked nodes across networks."""
        results = []
        for network, addrs in addresses.items():
            for addr in addrs:
                node = await self.check_node(network, addr)
                if node:
                    results.append(node)
        return results

    async def network_health(self) -> dict[str, bool]:
        """Check RPC health for each registered network."""
        health = {}
        for network, adapter in self.networks.items():
            health[network.value] = await adapter.health_check()
        return health

    def get_unhealthy_nodes(self) -> list[Node]:
        """Return nodes below uptime threshold from latest snapshots."""
        latest = self.store.load_latest()
        return [
            node
            for node in latest.values()
            if node.uptime_pct < self.config.monitor.uptime_alert_threshold
        ]

    def get_node_history(self, node_id: str) -> list[Node]:
        """Get all recorded snapshots for a node."""
        return [n for n in self.store.load_all() if n.node_id == node_id]

    def _emit_alert(
        self, alert_type: str, network: Network, address: str, **kwargs: object
    ) -> None:
        alert = {
            "type": alert_type,
            "network": network.value,
            "address": address,
            "timestamp": datetime.now(UTC).isoformat(),
            **kwargs,
        }
        self._alerts.append(alert)
        logger.warning("Alert: %s — %s on %s", alert_type, address[:16], network.value)
