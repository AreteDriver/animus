"""Discover new staking and node incentive opportunities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from animus_validator.config import ValidatorConfig
from animus_validator.models import Network, NetworkInfo
from animus_validator.networks.base import BaseNetwork

logger = logging.getLogger("animus_validator.discovery")


@dataclass
class StakingOpportunity:
    """A discovered staking or validator opportunity."""

    network: Network
    validator_address: str
    name: str
    apy_pct: float
    min_stake: float = 0.0
    commission_pct: float = 0.0
    total_staked: float = 0.0
    risk_score: float = 0.0  # 0=safe, 1=risky
    is_testnet: bool = False
    url: str = ""
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = field(default_factory=list)

    @property
    def net_apy(self) -> float:
        """APY after commission."""
        return self.apy_pct * (1.0 - self.commission_pct / 100.0)

    def to_dict(self) -> dict:
        return {
            "network": self.network.value,
            "validator_address": self.validator_address,
            "name": self.name,
            "apy_pct": self.apy_pct,
            "net_apy": self.net_apy,
            "min_stake": self.min_stake,
            "commission_pct": self.commission_pct,
            "total_staked": self.total_staked,
            "risk_score": self.risk_score,
            "is_testnet": self.is_testnet,
            "url": self.url,
            "discovered_at": self.discovered_at.isoformat(),
            "tags": self.tags,
        }


class OpportunityDiscovery:
    """Finds new staking opportunities across networks."""

    def __init__(self, config: ValidatorConfig, networks: dict[Network, BaseNetwork]):
        self.config = config
        self.networks = networks

    async def scan_validators(self, network: Network) -> list[StakingOpportunity]:
        """Scan a specific network for high-APY validators."""
        adapter = self.networks.get(network)
        if not adapter:
            logger.warning("No adapter for network: %s", network.value)
            return []

        try:
            raw_validators = await adapter.get_validators()
        except Exception as e:
            logger.error("Failed to scan %s validators: %s", network.value, e)
            return []

        info = adapter.info
        opportunities = []
        for v in raw_validators:
            apy = v.get("apy_pct", info.avg_apy_pct)
            commission = v.get("commission_pct", 0.0)
            opp = StakingOpportunity(
                network=network,
                validator_address=v.get("address", ""),
                name=v.get("name", "Unknown"),
                apy_pct=apy,
                min_stake=v.get("min_stake", info.min_stake),
                commission_pct=commission,
                total_staked=v.get("stake_total", 0.0),
            )
            opportunities.append(opp)

        return opportunities

    async def scan_all(self) -> list[StakingOpportunity]:
        """Scan all registered networks for opportunities."""
        if not self.config.discovery.enabled:
            return []

        all_opps: list[StakingOpportunity] = []
        for network in self.networks:
            opps = await self.scan_validators(network)
            all_opps.extend(opps)

        return all_opps

    def filter_opportunities(
        self, opportunities: list[StakingOpportunity]
    ) -> list[StakingOpportunity]:
        """Filter to opportunities meeting minimum criteria."""
        filtered = []
        for opp in opportunities:
            if opp.net_apy < self.config.discovery.min_apy_threshold:
                continue
            if opp.risk_score > self.config.discovery.max_risk_score:
                continue
            filtered.append(opp)

        # Sort by net APY descending
        filtered.sort(key=lambda o: o.net_apy, reverse=True)
        return filtered

    async def discover(self) -> list[StakingOpportunity]:
        """Full discovery pipeline: scan + filter."""
        raw = await self.scan_all()
        return self.filter_opportunities(raw)

    def get_network_comparison(self) -> list[dict]:
        """Compare networks by average APY and features."""
        comparison = []
        for network, adapter in self.networks.items():
            info = adapter.info
            comparison.append(info.to_dict())
        comparison.sort(key=lambda x: x.get("avg_apy_pct", 0), reverse=True)
        return comparison

    def get_network_info(self, network: Network) -> NetworkInfo | None:
        """Get info for a specific network."""
        adapter = self.networks.get(network)
        if adapter:
            return adapter.info
        return None
