"""Abstract base class for network adapters."""

from __future__ import annotations

import abc
from typing import Any

from animus_validator.models import (
    CompoundAction,
    Network,
    NetworkInfo,
    Node,
    RewardRecord,
    StakePosition,
)


class BaseNetwork(abc.ABC):
    """Abstract base for chain-specific staking operations."""

    @property
    @abc.abstractmethod
    def network(self) -> Network:
        """Which network this adapter handles."""

    @property
    @abc.abstractmethod
    def info(self) -> NetworkInfo:
        """Static info about this network."""

    @abc.abstractmethod
    async def get_stake_positions(self, address: str) -> list[StakePosition]:
        """Fetch active staking positions for an address."""

    @abc.abstractmethod
    async def get_rewards(self, address: str) -> list[RewardRecord]:
        """Fetch pending/claimed rewards for an address."""

    @abc.abstractmethod
    async def get_node_status(self, validator_address: str) -> Node | None:
        """Fetch node health and status."""

    @abc.abstractmethod
    async def compound_rewards(self, address: str, validator_address: str) -> CompoundAction | None:
        """Restake pending rewards. Returns action record or None if below threshold."""

    @abc.abstractmethod
    async def get_validators(self, **kwargs: Any) -> list[dict]:
        """List available validators with APY info."""

    async def health_check(self) -> bool:
        """Check if the network RPC is reachable."""
        return True
