"""Tests for persistent storage."""

from __future__ import annotations

import pytest

from animus_validator.models import (
    CompoundAction,
    Network,
    Node,
    RewardRecord,
    StakePosition,
)
from animus_validator.store import CompoundStore, NodeStore, PositionStore, RewardStore


@pytest.fixture()
def data_dir(tmp_path):
    return tmp_path / "data"


class TestPositionStore:
    def test_empty_store(self, data_dir):
        store = PositionStore(data_dir)
        assert store.total_count() == 0
        assert store.load_all() == []

    def test_record_position(self, data_dir):
        store = PositionStore(data_dir)
        pos = StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0)
        assert store.record(pos) is True
        assert store.total_count() == 1

    def test_dedup(self, data_dir):
        store = PositionStore(data_dir)
        pos = StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0)
        assert store.record(pos) is True
        assert store.record(pos) is False
        assert store.total_count() == 1

    def test_is_new(self, data_dir):
        store = PositionStore(data_dir)
        pos = StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0)
        assert store.is_new(pos) is True
        store.record(pos)
        assert store.is_new(pos) is False

    def test_record_batch(self, data_dir):
        store = PositionStore(data_dir)
        positions = [
            StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0),
            StakePosition(network=Network.ETHEREUM, validator_address="0xv2", amount=32.0),
            StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0),  # dup
        ]
        count = store.record_batch(positions)
        assert count == 2
        assert store.total_count() == 2

    def test_load_all(self, data_dir):
        store = PositionStore(data_dir)
        store.record(StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0))
        store.record(StakePosition(network=Network.ETHEREUM, validator_address="0xv2", amount=32.0))
        loaded = store.load_all()
        assert len(loaded) == 2
        assert loaded[0].network == Network.SUI
        assert loaded[1].network == Network.ETHEREUM

    def test_by_network(self, data_dir):
        store = PositionStore(data_dir)
        store.record(StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0))
        store.record(StakePosition(network=Network.ETHEREUM, validator_address="0xv2", amount=32.0))
        sui = store.by_network(Network.SUI)
        assert len(sui) == 1
        assert sui[0].network == Network.SUI

    def test_persistence(self, data_dir):
        store1 = PositionStore(data_dir)
        store1.record(StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0))

        store2 = PositionStore(data_dir)
        assert store2.total_count() == 1
        assert (
            store2.is_new(
                StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0)
            )
            is False
        )

    def test_corrupt_line_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "positions").mkdir(parents=True, exist_ok=True)
        f = data_dir / "positions" / "stakes.jsonl"
        f.write_text("not valid json\n")
        store = PositionStore(data_dir)
        assert store.total_count() == 0
        assert store.load_all() == []

    def test_load_all_empty_lines(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "positions").mkdir(parents=True, exist_ok=True)
        f = data_dir / "positions" / "stakes.jsonl"
        f.write_text("\n\n\n")
        store = PositionStore(data_dir)
        assert store.load_all() == []


class TestRewardStore:
    def test_empty_store(self, data_dir):
        store = RewardStore(data_dir)
        assert store.load_all() == []
        assert store.total_earned() == 0.0
        assert store.total_earned_usd() == 0.0

    def test_record_reward(self, data_dir):
        store = RewardStore(data_dir)
        r = RewardRecord(network=Network.SUI, amount=1.5, amount_usd=4.50)
        store.record(r)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].amount == 1.5

    def test_record_batch(self, data_dir):
        store = RewardStore(data_dir)
        rewards = [
            RewardRecord(network=Network.SUI, amount=1.0),
            RewardRecord(network=Network.ETHEREUM, amount=0.5),
        ]
        count = store.record_batch(rewards)
        assert count == 2
        assert len(store.load_all()) == 2

    def test_total_earned(self, data_dir):
        store = RewardStore(data_dir)
        store.record(RewardRecord(network=Network.SUI, amount=1.0, amount_usd=3.0))
        store.record(RewardRecord(network=Network.ETHEREUM, amount=0.5, amount_usd=1500.0))
        assert store.total_earned() == 1.5
        assert store.total_earned_usd() == 1503.0

    def test_by_network(self, data_dir):
        store = RewardStore(data_dir)
        store.record(RewardRecord(network=Network.SUI, amount=1.0))
        store.record(RewardRecord(network=Network.ETHEREUM, amount=0.5))
        sui = store.by_network(Network.SUI)
        assert len(sui) == 1

    def test_corrupt_line_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "rewards").mkdir(parents=True, exist_ok=True)
        f = data_dir / "rewards" / "history.jsonl"
        f.write_text("bad json\n")
        store = RewardStore(data_dir)
        assert store.load_all() == []

    def test_empty_lines_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "rewards").mkdir(parents=True, exist_ok=True)
        f = data_dir / "rewards" / "history.jsonl"
        f.write_text("\n\n")
        store = RewardStore(data_dir)
        assert store.load_all() == []


class TestCompoundStore:
    def test_empty_store(self, data_dir):
        store = CompoundStore(data_dir)
        assert store.load_all() == []
        assert store.total_compounded() == 0.0

    def test_record_action(self, data_dir):
        store = CompoundStore(data_dir)
        a = CompoundAction(network=Network.SUI, amount_compounded=5.0, tx_hash="0xhash")
        store.record(a)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].amount_compounded == 5.0

    def test_total_compounded(self, data_dir):
        store = CompoundStore(data_dir)
        store.record(CompoundAction(network=Network.SUI, amount_compounded=5.0))
        store.record(CompoundAction(network=Network.ETHEREUM, amount_compounded=0.5))
        assert store.total_compounded() == 5.5

    def test_corrupt_line_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "compounds").mkdir(parents=True, exist_ok=True)
        f = data_dir / "compounds" / "actions.jsonl"
        f.write_text("invalid\n")
        store = CompoundStore(data_dir)
        assert store.load_all() == []

    def test_empty_lines_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "compounds").mkdir(parents=True, exist_ok=True)
        f = data_dir / "compounds" / "actions.jsonl"
        f.write_text("\n\n")
        store = CompoundStore(data_dir)
        assert store.load_all() == []


class TestNodeStore:
    def test_empty_store(self, data_dir):
        store = NodeStore(data_dir)
        assert store.total_count() == 0
        assert store.load_all() == []
        assert store.load_latest() == {}

    def test_record_node(self, data_dir):
        store = NodeStore(data_dir)
        node = Node(network=Network.SUI, address="0xv1", stake_amount=1000.0)
        assert store.record(node) is True
        assert store.total_count() == 1

    def test_record_update(self, data_dir):
        store = NodeStore(data_dir)
        node = Node(network=Network.SUI, address="0xv1", stake_amount=1000.0)
        assert store.record(node) is True
        assert store.record(node) is False  # Already seen
        assert store.total_count() == 1

    def test_load_latest(self, data_dir):
        store = NodeStore(data_dir)
        node1 = Node(
            network=Network.SUI,
            address="0xv1",
            stake_amount=1000.0,
            uptime_pct=99.0,
        )
        node2 = Node(
            network=Network.SUI,
            address="0xv1",
            stake_amount=1000.0,
            uptime_pct=95.0,
        )
        store.record(node1)
        store.record(node2)

        latest = store.load_latest()
        assert len(latest) == 1
        # Last write wins
        assert list(latest.values())[0].uptime_pct == 95.0

    def test_load_all(self, data_dir):
        store = NodeStore(data_dir)
        store.record(Node(network=Network.SUI, address="0xv1", stake_amount=100))
        store.record(Node(network=Network.ETHEREUM, address="0xv2", stake_amount=32))
        # append same node with updated uptime
        store.record(Node(network=Network.SUI, address="0xv1", stake_amount=100, uptime_pct=90))
        all_nodes = store.load_all()
        assert len(all_nodes) == 3  # All entries, not deduped

    def test_corrupt_line_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        f = data_dir / "nodes.jsonl"
        f.write_text("corrupt\n")
        store = NodeStore(data_dir)
        assert store.total_count() == 0
        assert store.load_all() == []

    def test_empty_lines_skipped(self, data_dir):
        data_dir.mkdir(parents=True, exist_ok=True)
        f = data_dir / "nodes.jsonl"
        f.write_text("\n\n")
        store = NodeStore(data_dir)
        assert store.load_all() == []

    def test_persistence(self, data_dir):
        store1 = NodeStore(data_dir)
        store1.record(Node(network=Network.SUI, address="0xv1", stake_amount=100))

        store2 = NodeStore(data_dir)
        assert store2.total_count() == 1
