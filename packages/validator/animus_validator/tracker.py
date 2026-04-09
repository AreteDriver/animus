"""Rewards tracking and reporting."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from animus_validator.config import ValidatorConfig
from animus_validator.models import Network, RewardRecord
from animus_validator.store import PositionStore, RewardStore

logger = logging.getLogger("animus_validator.tracker")


class RewardTracker:
    """Tracks and reports on staking rewards across networks."""

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.reward_store = RewardStore(config.data_dir)
        self.position_store = PositionStore(config.data_dir)

    def record_reward(self, reward: RewardRecord) -> None:
        """Record a new reward event."""
        self.reward_store.record(reward)

    def record_rewards(self, rewards: list[RewardRecord]) -> int:
        """Record multiple reward events. Returns count."""
        return self.reward_store.record_batch(rewards)

    def total_earned(self) -> float:
        """Total rewards earned across all networks (token amounts)."""
        return self.reward_store.total_earned()

    def total_earned_usd(self) -> float:
        """Total rewards earned in USD."""
        return self.reward_store.total_earned_usd()

    def earnings_by_network(self) -> dict[str, float]:
        """Total earnings grouped by network."""
        by_net: dict[str, float] = defaultdict(float)
        for r in self.reward_store.load_all():
            by_net[r.network.value] += r.amount
        return dict(by_net)

    def earnings_by_network_usd(self) -> dict[str, float]:
        """Total USD earnings grouped by network."""
        by_net: dict[str, float] = defaultdict(float)
        for r in self.reward_store.load_all():
            by_net[r.network.value] += r.amount_usd
        return dict(by_net)

    def recent_rewards(self, days: int = 7) -> list[RewardRecord]:
        """Get rewards from the last N days."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        return [
            r
            for r in self.reward_store.load_all()
            if r.timestamp.tzinfo is not None and r.timestamp >= cutoff
        ]

    def daily_average(self, days: int = 30) -> float:
        """Average daily reward over the last N days."""
        recent = self.recent_rewards(days)
        if not recent:
            return 0.0
        return sum(r.amount for r in recent) / days

    def daily_average_usd(self, days: int = 30) -> float:
        """Average daily USD reward over the last N days."""
        recent = self.recent_rewards(days)
        if not recent:
            return 0.0
        return sum(r.amount_usd for r in recent) / days

    def yield_report(self) -> dict:
        """Generate a comprehensive yield report."""
        all_rewards = self.reward_store.load_all()
        all_positions = self.position_store.load_all()

        total_staked = sum(p.amount for p in all_positions)
        total_pending = sum(p.rewards_pending for p in all_positions)
        positions_by_net: dict[str, list] = defaultdict(list)
        for p in all_positions:
            positions_by_net[p.network.value].append(p.to_dict())

        return {
            "total_rewards": self.total_earned(),
            "total_rewards_usd": self.total_earned_usd(),
            "total_staked": total_staked,
            "total_pending_rewards": total_pending,
            "reward_count": len(all_rewards),
            "position_count": len(all_positions),
            "earnings_by_network": self.earnings_by_network(),
            "earnings_by_network_usd": self.earnings_by_network_usd(),
            "daily_avg_30d": self.daily_average(30),
            "daily_avg_usd_30d": self.daily_average_usd(30),
            "positions_by_network": dict(positions_by_net),
        }

    def positions_summary(self) -> list[dict]:
        """Summarize all active positions."""
        positions = self.position_store.load_all()
        return [p.to_dict() for p in positions]

    def network_rewards(self, network: Network) -> list[RewardRecord]:
        """Get all rewards for a specific network."""
        return self.reward_store.by_network(network)
