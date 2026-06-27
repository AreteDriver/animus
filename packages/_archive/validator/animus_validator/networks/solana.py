"""Solana network adapter for staking operations."""

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

logger = logging.getLogger("animus_validator.networks.solana")

DEFAULT_SOL_RPC = "https://api.mainnet-beta.solana.com"


class SolanaNetwork(BaseNetwork):
    """Solana native staking via JSON-RPC."""

    def __init__(self, rpc_url: str = DEFAULT_SOL_RPC, timeout: float = 10.0):
        self.rpc_url = rpc_url
        self.timeout = timeout

    @property
    def network(self) -> Network:
        return Network.SOLANA

    @property
    def info(self) -> NetworkInfo:
        return NetworkInfo(
            network=Network.SOLANA,
            display_name="Solana",
            native_token="SOL",
            avg_apy_pct=7.0,
            min_stake=0.01,
            unbonding_days=2,
            supports_auto_compound=False,
            rpc_url=self.rpc_url,
        )

    async def _rpc_call(self, method: str, params: list | None = None) -> dict:
        """Make a JSON-RPC call to Solana."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.rpc_url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def get_stake_positions(self, address: str) -> list[StakePosition]:
        """Fetch SOL stake accounts for an address."""
        try:
            result = await self._rpc_call(
                "getStakeAccounts",
                [address, {"encoding": "jsonParsed"}],
            )
            accounts = result.get("result", [])
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch SOL stakes for %s: %s", address, e)
            return []

        positions = []
        for acct in accounts:
            parsed = acct.get("account", {}).get("data", {}).get("parsed", {})
            info = parsed.get("info", {})
            stake_info = info.get("stake", {})
            delegation = stake_info.get("delegation", {})

            if not delegation:
                continue

            voter = delegation.get("voter", "")
            amount = int(delegation.get("stake", 0)) / 1e9
            positions.append(
                StakePosition(
                    network=Network.SOLANA,
                    validator_address=voter,
                    amount=amount,
                    apy_pct=7.0,
                    rewards_pending=0.0,
                    auto_compound=False,
                )
            )
        return positions

    async def get_rewards(self, address: str) -> list[RewardRecord]:
        """Fetch inflation rewards for a stake account."""
        try:
            # getInflationReward returns rewards for given epochs
            result = await self._rpc_call("getInflationReward", [[address]])
            rewards_data = result.get("result", [])
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch SOL rewards for %s: %s", address, e)
            return []

        records = []
        for reward in rewards_data:
            if reward is None:
                continue
            amount = reward.get("amount", 0) / 1e9
            if amount > 0:
                records.append(
                    RewardRecord(
                        network=Network.SOLANA,
                        amount=amount,
                        reward_type=RewardType.STAKING,
                        validator_address=address,
                    )
                )
        return records

    async def get_node_status(self, validator_address: str) -> Node | None:
        """Fetch validator vote account info."""
        try:
            result = await self._rpc_call("getVoteAccounts", [])
            vote_data = result.get("result", {})
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch SOL vote accounts: %s", e)
            return None

        for v in vote_data.get("current", []):
            if v.get("votePubkey") == validator_address:
                stake = int(v.get("activatedStake", 0)) / 1e9
                return Node(
                    network=Network.SOLANA,
                    address=validator_address,
                    stake_amount=stake,
                    uptime_pct=100.0,
                    status=StakeStatus.ACTIVE,
                )

        # Check delinquent validators
        for v in vote_data.get("delinquent", []):
            if v.get("votePubkey") == validator_address:
                stake = int(v.get("activatedStake", 0)) / 1e9
                return Node(
                    network=Network.SOLANA,
                    address=validator_address,
                    stake_amount=stake,
                    uptime_pct=0.0,
                    status=StakeStatus.ACTIVE,
                )

        return None

    async def compound_rewards(self, address: str, validator_address: str) -> CompoundAction | None:
        """SOL native staking does not auto-compound."""
        logger.info("SOL staking does not natively auto-compound — manual restake needed")
        return None

    async def get_validators(self, **kwargs: Any) -> list[dict]:
        """List Solana vote accounts."""
        try:
            result = await self._rpc_call("getVoteAccounts", [])
            vote_data = result.get("result", {})
        except (httpx.HTTPError, KeyError) as e:
            logger.error("Failed to fetch SOL validators: %s", e)
            return []

        validators = []
        for v in vote_data.get("current", []):
            stake = int(v.get("activatedStake", 0)) / 1e9
            validators.append(
                {
                    "address": v.get("votePubkey", ""),
                    "node_pubkey": v.get("nodePubkey", ""),
                    "stake_total": stake,
                    "commission_pct": v.get("commission", 0),
                    "last_vote": v.get("lastVote", 0),
                }
            )
        return validators

    async def health_check(self) -> bool:
        """Check if Solana RPC is reachable."""
        try:
            result = await self._rpc_call("getHealth", [])
            return result.get("result") == "ok"
        except (httpx.HTTPError, Exception):
            return False
