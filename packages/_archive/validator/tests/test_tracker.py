"""Tests for rewards tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_validator.config import ValidatorConfig
from animus_validator.models import Network, RewardRecord, StakePosition
from animus_validator.tracker import RewardTracker


@pytest.fixture()
def config(tmp_path):
    c = ValidatorConfig(data_dir=tmp_path / "data")
    c.ensure_dirs()
    return c


@pytest.fixture()
def tracker(config):
    return RewardTracker(config)


class TestRewardTracker:
    def test_empty_tracker(self, tracker):
        assert tracker.total_earned() == 0.0
        assert tracker.total_earned_usd() == 0.0
        assert tracker.earnings_by_network() == {}
        assert tracker.earnings_by_network_usd() == {}
        assert tracker.recent_rewards(7) == []
        assert tracker.daily_average(30) == 0.0
        assert tracker.daily_average_usd(30) == 0.0

    def test_record_reward(self, tracker):
        r = RewardRecord(network=Network.SUI, amount=1.5, amount_usd=4.50)
        tracker.record_reward(r)
        assert tracker.total_earned() == 1.5
        assert tracker.total_earned_usd() == 4.50

    def test_record_rewards(self, tracker):
        rewards = [
            RewardRecord(network=Network.SUI, amount=1.0),
            RewardRecord(network=Network.ETHEREUM, amount=0.5),
        ]
        count = tracker.record_rewards(rewards)
        assert count == 2
        assert tracker.total_earned() == 1.5

    def test_earnings_by_network(self, tracker):
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=1.0))
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=2.0))
        tracker.record_reward(RewardRecord(network=Network.ETHEREUM, amount=0.5))

        by_net = tracker.earnings_by_network()
        assert by_net["sui"] == 3.0
        assert by_net["ethereum"] == 0.5

    def test_earnings_by_network_usd(self, tracker):
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=1.0, amount_usd=3.0))
        tracker.record_reward(RewardRecord(network=Network.ETHEREUM, amount=0.5, amount_usd=1500.0))
        by_net = tracker.earnings_by_network_usd()
        assert by_net["sui"] == 3.0
        assert by_net["ethereum"] == 1500.0

    def test_recent_rewards(self, tracker):
        now = datetime.now(UTC)
        old = now - timedelta(days=30)
        recent = now - timedelta(days=1)

        tracker.record_reward(RewardRecord(network=Network.SUI, amount=1.0, timestamp=old))
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=2.0, timestamp=recent))

        recent_rewards = tracker.recent_rewards(7)
        assert len(recent_rewards) == 1
        assert recent_rewards[0].amount == 2.0

    def test_recent_rewards_filters_naive_timestamps(self, tracker):
        """Rewards with naive timestamps (no tzinfo) are excluded."""
        naive_ts = datetime(2026, 1, 1)  # No tzinfo
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=1.0, timestamp=naive_ts))
        recent = tracker.recent_rewards(365 * 10)  # Very wide window
        assert len(recent) == 0

    def test_daily_average(self, tracker):
        now = datetime.now(UTC)
        for i in range(10):
            tracker.record_reward(
                RewardRecord(
                    network=Network.SUI,
                    amount=1.0,
                    timestamp=now - timedelta(days=i),
                )
            )
        avg = tracker.daily_average(30)
        assert avg == pytest.approx(10.0 / 30)

    def test_daily_average_usd(self, tracker):
        now = datetime.now(UTC)
        for i in range(5):
            tracker.record_reward(
                RewardRecord(
                    network=Network.SUI,
                    amount=1.0,
                    amount_usd=3.0,
                    timestamp=now - timedelta(days=i),
                )
            )
        avg = tracker.daily_average_usd(30)
        assert avg == pytest.approx(15.0 / 30)

    def test_yield_report_empty(self, tracker):
        report = tracker.yield_report()
        assert report["total_rewards"] == 0.0
        assert report["total_staked"] == 0.0
        assert report["position_count"] == 0
        assert report["reward_count"] == 0

    def test_yield_report_with_data(self, tracker):
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=1.0, amount_usd=3.0))
        tracker.position_store.record(
            StakePosition(
                network=Network.SUI,
                validator_address="0xv1",
                amount=100.0,
                rewards_pending=0.5,
            )
        )

        report = tracker.yield_report()
        assert report["total_rewards"] == 1.0
        assert report["total_rewards_usd"] == 3.0
        assert report["total_staked"] == 100.0
        assert report["total_pending_rewards"] == 0.5
        assert report["position_count"] == 1
        assert report["reward_count"] == 1
        assert "sui" in report["positions_by_network"]

    def test_positions_summary(self, tracker):
        tracker.position_store.record(
            StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0)
        )
        summary = tracker.positions_summary()
        assert len(summary) == 1
        assert summary[0]["network"] == "sui"

    def test_positions_summary_empty(self, tracker):
        assert tracker.positions_summary() == []

    def test_network_rewards(self, tracker):
        tracker.record_reward(RewardRecord(network=Network.SUI, amount=1.0))
        tracker.record_reward(RewardRecord(network=Network.ETHEREUM, amount=0.5))

        sui_rewards = tracker.network_rewards(Network.SUI)
        assert len(sui_rewards) == 1
        assert sui_rewards[0].network == Network.SUI

        sol_rewards = tracker.network_rewards(Network.SOLANA)
        assert len(sol_rewards) == 0
