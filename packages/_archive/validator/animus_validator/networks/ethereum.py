"""Ethereum network adapter for staking operations (Lido, Rocket Pool)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from animus_validator.models import (
    CompoundAction,
    Network,
    NetworkInfo,
    Node,
    RewardRecord,
    RewardType,
    StakePosition,
    StakeStatus,
)
from animus_validator.networks.base import BaseNetwork

logger = logging.getLogger("animus_validator.networks.ethereum")

DEFAULT_ETH_RPC = "https://eth.llamarpc.com"
LIDO_APY_API = "https://eth-api.lido.fi/v1/protocol/steth/apr/sma"


class EthereumNetwork(BaseNetwork):
    """Ethereum staking via Lido and Rocket Pool liquid staking."""

    def __init__(self, rpc_url: str = DEFAULT_ETH_RPC, timeout: float = 10.0):
        self.rpc_url = rpc_url
        self.timeout = timeout

    @property
    def network(self) -> Network:
        return Network.ETHEREUM

    @property
    def info(self) -> NetworkInfo:
        return NetworkInfo(
            network=Network.ETHEREUM,
            display_name="Ethereum",
            native_token="ETH",
            avg_apy_pct=3.8,
            min_stake=0.01,  # Lido has no minimum
            unbonding_days=3,
            supports_auto_compound=True,
            rpc_url=self.rpc_url,
        )

    async def _get_eth_balance(self, address: str) -> float:
        """Fetch ETH balance via JSON-RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getBalance",
            "params": [address, "latest"],
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.rpc_url, json=payload)
            resp.raise_for_status()
            result = resp.json().get("result", "0x0")
            return int(result, 16) / 1e18

    async def get_stake_positions(self, address: str) -> list[StakePosition]:
        """Fetch stETH/rETH positions. Simplified: checks ETH balance as proxy."""
        try:
            balance = await self._get_eth_balance(address)
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Failed to fetch ETH balance for %s: %s", address, e)
            return []

        if balance <= 0:
            return []

        return [
            StakePosition(
                network=Network.ETHEREUM,
                validator_address="lido",
                amount=balance,
                apy_pct=3.8,
                rewards_pending=0.0,
                auto_compound=True,
            )
        ]

    async def get_rewards(self, address: str) -> list[RewardRecord]:
        """Lido stETH auto-compounds — rewards reflected in balance growth."""
        positions = await self.get_stake_positions(address)
        records = []
        for pos in positions:
            if pos.rewards_pending > 0:
                records.append(
                    RewardRecord(
                        network=Network.ETHEREUM,
                        amount=pos.rewards_pending,
                        reward_type=RewardType.STAKING,
                        validator_address=pos.validator_address,
                    )
                )
        return records

    async def get_node_status(self, validator_address: str) -> Node | None:
        """ETH liquid staking does not expose individual node status."""
        return Node(
            network=Network.ETHEREUM,
            address=validator_address,
            stake_amount=0.0,
            uptime_pct=99.9,
            status=StakeStatus.ACTIVE,
            name=f"ETH Validator ({validator_address[:8]}...)",
        )

    async def compound_rewards(self, address: str, validator_address: str) -> CompoundAction | None:
        """Lido stETH auto-compounds. No manual action needed."""
        logger.info("ETH staking via Lido auto-compounds — no manual action needed")
        return CompoundAction(
            network=Network.ETHEREUM,
            amount_compounded=0.0,
            tx_hash="auto-compound",
            new_total_stake=0.0,
            validator_address=validator_address,
        )

    async def get_validators(self, **kwargs: Any) -> list[dict]:
        """Return known liquid staking protocols."""
        return [
            {
                "address": "lido",
                "name": "Lido Finance",
                "apy_pct": 3.8,
                "protocol": "liquid_staking",
                "min_stake": 0.01,
            },
            {
                "address": "rocket_pool",
                "name": "Rocket Pool",
                "apy_pct": 3.5,
                "protocol": "liquid_staking",
                "min_stake": 0.01,
            },
        ]

    async def health_check(self) -> bool:
        """Check if ETH RPC is reachable."""
        try:
            await self._get_eth_balance("0x0000000000000000000000000000000000000000")
            return True
        except (httpx.HTTPError, Exception):
            return False
