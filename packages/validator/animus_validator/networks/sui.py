"""Sui network adapter for staking operations."""

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

logger = logging.getLogger("animus_validator.networks.sui")

DEFAULT_SUI_RPC = "https://fullnode.mainnet.sui.io:443"


class SuiNetwork(BaseNetwork):
    """Sui staking via JSON-RPC."""

    def __init__(self, rpc_url: str = DEFAULT_SUI_RPC, timeout: float = 10.0):
        self.rpc_url = rpc_url
        self.timeout = timeout

    @property
    def network(self) -> Network:
        return Network.SUI

    @property
    def info(self) -> NetworkInfo:
        return NetworkInfo(
            network=Network.SUI,
            display_name="Sui",
            native_token="SUI",
            avg_apy_pct=3.5,
            min_stake=1.0,
            unbonding_days=1,
            supports_auto_compound=True,
            rpc_url=self.rpc_url,
        )

    async def _rpc_call(self, method: str, params: list) -> dict:
        """Make a JSON-RPC call to Sui fullnode."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.rpc_url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def get_stake_positions(self, address: str) -> list[StakePosition]:
        """Fetch staked SUI positions via suix_getStakes."""
        try:
            result = await self._rpc_call("suix_getStakes", [address])
            stakes = result.get("result", [])
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch Sui stakes for %s: %s", address, e)
            return []

        positions = []
        for stake_group in stakes:
            validator = stake_group.get("validatorAddress", "")
            for s in stake_group.get("stakes", []):
                amount = int(s.get("principal", 0)) / 1e9
                rewards = int(s.get("estimatedReward", 0)) / 1e9
                positions.append(
                    StakePosition(
                        network=Network.SUI,
                        validator_address=validator,
                        amount=amount,
                        apy_pct=3.5,
                        rewards_pending=rewards,
                        auto_compound=False,
                    )
                )
        return positions

    async def get_rewards(self, address: str) -> list[RewardRecord]:
        """Derive reward records from stake positions."""
        positions = await self.get_stake_positions(address)
        records = []
        for pos in positions:
            if pos.rewards_pending > 0:
                records.append(
                    RewardRecord(
                        network=Network.SUI,
                        amount=pos.rewards_pending,
                        reward_type=RewardType.STAKING,
                        validator_address=pos.validator_address,
                    )
                )
        return records

    async def get_node_status(self, validator_address: str) -> Node | None:
        """Fetch validator info from latest system state."""
        try:
            result = await self._rpc_call("suix_getLatestSuiSystemState", [])
            state = result.get("result", {})
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch Sui system state: %s", e)
            return None

        for v in state.get("activeValidators", []):
            if v.get("suiAddress") == validator_address:
                stake = int(v.get("stakingPoolSuiBalance", 0)) / 1e9
                return Node(
                    network=Network.SUI,
                    address=validator_address,
                    stake_amount=stake,
                    uptime_pct=100.0,
                    status=StakeStatus.ACTIVE,
                    name=v.get("name", ""),
                )
        return None

    async def compound_rewards(self, address: str, validator_address: str) -> CompoundAction | None:
        """Sui does not natively support auto-compound. Returns None."""
        logger.info("Sui auto-compound not natively supported — manual restake required")
        return None

    async def get_validators(self, **kwargs: Any) -> list[dict]:
        """List active Sui validators."""
        try:
            result = await self._rpc_call("suix_getLatestSuiSystemState", [])
            state = result.get("result", {})
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch Sui validators: %s", e)
            return []

        validators = []
        for v in state.get("activeValidators", []):
            stake = int(v.get("stakingPoolSuiBalance", 0)) / 1e9
            validators.append(
                {
                    "address": v.get("suiAddress", ""),
                    "name": v.get("name", ""),
                    "stake_total": stake,
                    "commission_pct": int(v.get("commissionRate", 0)) / 100,
                }
            )
        return validators

    async def health_check(self) -> bool:
        """Check if Sui RPC is reachable."""
        try:
            await self._rpc_call("sui_getLatestCheckpointSequenceNumber", [])
            return True
        except (httpx.HTTPError, Exception):
            return False
