"""Auto-compound rewards when threshold is met."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_validator.config import ValidatorConfig
from animus_validator.models import CompoundAction, Network, StakePosition
from animus_validator.networks.base import BaseNetwork
from animus_validator.store import CompoundStore

logger = logging.getLogger("animus_validator.compounder")


class RewardCompounder:
    """Auto-compounds staking rewards when they exceed a gas-adjusted threshold."""

    def __init__(self, config: ValidatorConfig, networks: dict[Network, BaseNetwork]):
        self.config = config
        self.networks = networks
        self.store = CompoundStore(config.data_dir)

    def should_compound(self, position: StakePosition, gas_estimate: float = 0.001) -> bool:
        """Determine if rewards justify compounding given gas costs."""
        if not self.config.compound.enabled:
            return False
        if not position.auto_compound:
            return False
        if position.rewards_pending < self.config.compound.min_reward_threshold:
            return False
        # Rewards must exceed gas cost * multiplier
        min_profitable = gas_estimate * self.config.compound.gas_buffer_multiplier
        return position.rewards_pending >= min_profitable

    async def compound_position(
        self,
        position: StakePosition,
        gas_estimate: float = 0.001,
    ) -> CompoundAction | None:
        """Attempt to compound a single position."""
        if not self.should_compound(position, gas_estimate):
            return None

        adapter = self.networks.get(position.network)
        if not adapter:
            logger.warning("No adapter for network: %s", position.network.value)
            return None

        action = await adapter.compound_rewards(
            address=position.validator_address,
            validator_address=position.validator_address,
        )

        if action:
            action.amount_compounded = position.rewards_pending
            action.new_total_stake = position.amount + position.rewards_pending
            action.timestamp = datetime.now(UTC)
            self.store.record(action)
            logger.info(
                "Compounded %.6f on %s (new total: %.6f)",
                action.amount_compounded,
                position.network.value,
                action.new_total_stake,
            )

        return action

    async def compound_all(
        self,
        positions: list[StakePosition],
        gas_estimates: dict[Network, float] | None = None,
    ) -> list[CompoundAction]:
        """Attempt to compound all eligible positions."""
        gas = gas_estimates or {}
        actions = []
        for pos in positions:
            gas_est = gas.get(pos.network, 0.001)
            action = await self.compound_position(pos, gas_est)
            if action:
                actions.append(action)
        return actions

    def get_compound_history(self) -> list[CompoundAction]:
        """Get all historical compound actions."""
        return self.store.load_all()

    def total_compounded(self) -> float:
        """Total amount compounded across all networks."""
        return self.store.total_compounded()

    def compound_summary(self) -> dict:
        """Summary of compounding activity."""
        actions = self.store.load_all()
        by_network: dict[str, float] = {}
        for a in actions:
            by_network[a.network.value] = by_network.get(a.network.value, 0.0) + a.amount_compounded
        return {
            "total_actions": len(actions),
            "total_compounded": self.store.total_compounded(),
            "by_network": by_network,
        }
