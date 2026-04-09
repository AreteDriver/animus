"""Tests for data models."""

from __future__ import annotations

from datetime import UTC, datetime

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


class TestNetwork:
    def test_all_networks(self):
        assert Network.SUI.value == "sui"
        assert Network.ETHEREUM.value == "ethereum"
        assert Network.SOLANA.value == "solana"
        assert Network.COSMOS.value == "cosmos"
        assert Network.AVALANCHE.value == "avalanche"
        assert Network.POLYGON.value == "polygon"

    def test_network_from_value(self):
        assert Network("sui") == Network.SUI
        assert Network("ethereum") == Network.ETHEREUM


class TestStakeStatus:
    def test_all_statuses(self):
        assert StakeStatus.ACTIVE.value == "active"
        assert StakeStatus.UNSTAKING.value == "unstaking"
        assert StakeStatus.WITHDRAWN.value == "withdrawn"
        assert StakeStatus.PENDING.value == "pending"


class TestRewardType:
    def test_all_types(self):
        assert RewardType.STAKING.value == "staking"
        assert RewardType.VALIDATION.value == "validation"
        assert RewardType.DELEGATION.value == "delegation"


class TestNode:
    def test_create_node(self):
        node = Node(
            network=Network.SUI,
            address="0xabc123",
            stake_amount=1000.0,
        )
        assert node.network == Network.SUI
        assert node.address == "0xabc123"
        assert node.stake_amount == 1000.0
        assert node.rewards_earned == 0.0
        assert node.uptime_pct == 100.0
        assert node.last_check is None
        assert node.status == StakeStatus.ACTIVE
        assert node.name == ""

    def test_node_id_deterministic(self):
        node1 = Node(network=Network.SUI, address="0xabc", stake_amount=100)
        node2 = Node(network=Network.SUI, address="0xabc", stake_amount=200)
        assert node1.node_id == node2.node_id  # Same network+address

    def test_node_id_differs_by_network(self):
        node1 = Node(network=Network.SUI, address="0xabc", stake_amount=100)
        node2 = Node(network=Network.ETHEREUM, address="0xabc", stake_amount=100)
        assert node1.node_id != node2.node_id

    def test_is_healthy(self):
        node = Node(network=Network.SUI, address="0x1", stake_amount=100, uptime_pct=99.0)
        assert node.is_healthy is True

    def test_is_not_healthy_low_uptime(self):
        node = Node(network=Network.SUI, address="0x1", stake_amount=100, uptime_pct=90.0)
        assert node.is_healthy is False

    def test_is_not_healthy_not_active(self):
        node = Node(
            network=Network.SUI,
            address="0x1",
            stake_amount=100,
            uptime_pct=100.0,
            status=StakeStatus.UNSTAKING,
        )
        assert node.is_healthy is False

    def test_to_dict(self):
        now = datetime.now(UTC)
        node = Node(
            network=Network.SUI,
            address="0xabc",
            stake_amount=1000.0,
            rewards_earned=50.0,
            uptime_pct=99.5,
            last_check=now,
            status=StakeStatus.ACTIVE,
            name="My Validator",
        )
        d = node.to_dict()
        assert d["network"] == "sui"
        assert d["address"] == "0xabc"
        assert d["stake_amount"] == 1000.0
        assert d["rewards_earned"] == 50.0
        assert d["uptime_pct"] == 99.5
        assert d["last_check"] == now.isoformat()
        assert d["status"] == "active"
        assert d["name"] == "My Validator"
        assert "node_id" in d

    def test_to_dict_no_last_check(self):
        node = Node(network=Network.SUI, address="0x1", stake_amount=100)
        d = node.to_dict()
        assert d["last_check"] is None

    def test_from_dict(self):
        now = datetime.now(UTC)
        data = {
            "network": "ethereum",
            "address": "0xdef",
            "stake_amount": 32.0,
            "rewards_earned": 1.5,
            "uptime_pct": 98.0,
            "last_check": now.isoformat(),
            "status": "active",
            "name": "ETH Node",
        }
        node = Node.from_dict(data)
        assert node.network == Network.ETHEREUM
        assert node.address == "0xdef"
        assert node.stake_amount == 32.0
        assert node.name == "ETH Node"

    def test_from_dict_defaults(self):
        data = {"network": "sui", "address": "0x1"}
        node = Node.from_dict(data)
        assert node.stake_amount == 0.0
        assert node.uptime_pct == 100.0
        assert node.status == StakeStatus.ACTIVE
        assert node.last_check is None
        assert node.name == ""

    def test_roundtrip(self):
        node = Node(
            network=Network.SOLANA,
            address="abc123",
            stake_amount=500.0,
            rewards_earned=25.0,
            uptime_pct=97.0,
            status=StakeStatus.ACTIVE,
            name="SOL Node",
        )
        restored = Node.from_dict(node.to_dict())
        assert restored.network == node.network
        assert restored.address == node.address
        assert restored.stake_amount == node.stake_amount


class TestStakePosition:
    def test_create_position(self):
        pos = StakePosition(
            network=Network.SUI,
            validator_address="0xval1",
            amount=1000.0,
        )
        assert pos.network == Network.SUI
        assert pos.amount == 1000.0
        assert pos.apy_pct == 0.0
        assert pos.rewards_pending == 0.0
        assert pos.auto_compound is False

    def test_position_id_deterministic(self):
        pos1 = StakePosition(network=Network.SUI, validator_address="0xv", amount=100)
        pos2 = StakePosition(network=Network.SUI, validator_address="0xv", amount=100)
        assert pos1.position_id == pos2.position_id

    def test_position_id_differs_by_amount(self):
        pos1 = StakePosition(network=Network.SUI, validator_address="0xv", amount=100)
        pos2 = StakePosition(network=Network.SUI, validator_address="0xv", amount=200)
        assert pos1.position_id != pos2.position_id

    def test_annual_yield_estimate(self):
        pos = StakePosition(
            network=Network.SUI, validator_address="0xv", amount=1000.0, apy_pct=5.0
        )
        assert pos.annual_yield_estimate == 50.0

    def test_annual_yield_zero_apy(self):
        pos = StakePosition(
            network=Network.SUI, validator_address="0xv", amount=1000.0, apy_pct=0.0
        )
        assert pos.annual_yield_estimate == 0.0

    def test_to_dict(self):
        pos = StakePosition(
            network=Network.ETHEREUM,
            validator_address="0xval",
            amount=32.0,
            apy_pct=3.8,
            rewards_pending=0.5,
            auto_compound=True,
        )
        d = pos.to_dict()
        assert d["network"] == "ethereum"
        assert d["validator_address"] == "0xval"
        assert d["amount"] == 32.0
        assert d["apy_pct"] == 3.8
        assert d["auto_compound"] is True
        assert "position_id" in d
        assert "staked_at" in d

    def test_from_dict(self):
        data = {
            "network": "solana",
            "validator_address": "abc",
            "amount": 100.0,
            "apy_pct": 7.0,
            "rewards_pending": 2.0,
            "auto_compound": True,
            "staked_at": "2026-01-01T00:00:00+00:00",
        }
        pos = StakePosition.from_dict(data)
        assert pos.network == Network.SOLANA
        assert pos.amount == 100.0
        assert pos.auto_compound is True

    def test_from_dict_defaults(self):
        data = {"network": "sui", "validator_address": "0x1"}
        pos = StakePosition.from_dict(data)
        assert pos.amount == 0.0
        assert pos.apy_pct == 0.0
        assert pos.auto_compound is False

    def test_from_dict_no_staked_at(self):
        data = {"network": "sui", "validator_address": "0x1", "amount": 10.0}
        pos = StakePosition.from_dict(data)
        assert pos.staked_at is not None

    def test_roundtrip(self):
        pos = StakePosition(
            network=Network.ETHEREUM,
            validator_address="lido",
            amount=10.0,
            apy_pct=3.8,
            rewards_pending=0.1,
            auto_compound=True,
        )
        restored = StakePosition.from_dict(pos.to_dict())
        assert restored.network == pos.network
        assert restored.amount == pos.amount
        assert restored.auto_compound == pos.auto_compound


class TestRewardRecord:
    def test_create_reward(self):
        r = RewardRecord(
            network=Network.SUI,
            amount=1.5,
        )
        assert r.network == Network.SUI
        assert r.amount == 1.5
        assert r.amount_usd == 0.0
        assert r.reward_type == RewardType.STAKING
        assert r.validator_address == ""

    def test_record_id_deterministic(self):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        r1 = RewardRecord(network=Network.SUI, amount=1.0, timestamp=ts, validator_address="0x1")
        r2 = RewardRecord(network=Network.SUI, amount=2.0, timestamp=ts, validator_address="0x1")
        assert r1.record_id == r2.record_id  # Same network+addr+timestamp

    def test_to_dict(self):
        r = RewardRecord(
            network=Network.ETHEREUM,
            amount=0.5,
            amount_usd=1500.0,
            reward_type=RewardType.DELEGATION,
            validator_address="0xval",
        )
        d = r.to_dict()
        assert d["network"] == "ethereum"
        assert d["amount"] == 0.5
        assert d["amount_usd"] == 1500.0
        assert d["reward_type"] == "delegation"
        assert "record_id" in d

    def test_from_dict(self):
        data = {
            "network": "solana",
            "amount": 3.0,
            "amount_usd": 450.0,
            "reward_type": "validation",
            "timestamp": "2026-03-01T00:00:00+00:00",
            "validator_address": "abc",
        }
        r = RewardRecord.from_dict(data)
        assert r.network == Network.SOLANA
        assert r.amount == 3.0
        assert r.reward_type == RewardType.VALIDATION

    def test_from_dict_defaults(self):
        data = {"network": "sui"}
        r = RewardRecord.from_dict(data)
        assert r.amount == 0.0
        assert r.reward_type == RewardType.STAKING
        assert r.validator_address == ""

    def test_from_dict_no_timestamp(self):
        data = {"network": "sui", "amount": 1.0}
        r = RewardRecord.from_dict(data)
        assert r.timestamp is not None

    def test_roundtrip(self):
        r = RewardRecord(
            network=Network.ETHEREUM,
            amount=0.1,
            amount_usd=300.0,
            reward_type=RewardType.STAKING,
            validator_address="lido",
        )
        restored = RewardRecord.from_dict(r.to_dict())
        assert restored.network == r.network
        assert restored.amount == r.amount


class TestCompoundAction:
    def test_create_action(self):
        a = CompoundAction(
            network=Network.SUI,
            amount_compounded=5.0,
        )
        assert a.network == Network.SUI
        assert a.amount_compounded == 5.0
        assert a.tx_hash == ""
        assert a.new_total_stake == 0.0

    def test_to_dict(self):
        a = CompoundAction(
            network=Network.ETHEREUM,
            amount_compounded=0.5,
            tx_hash="0xabc",
            new_total_stake=32.5,
            validator_address="lido",
        )
        d = a.to_dict()
        assert d["network"] == "ethereum"
        assert d["amount_compounded"] == 0.5
        assert d["tx_hash"] == "0xabc"
        assert d["new_total_stake"] == 32.5

    def test_from_dict(self):
        data = {
            "network": "solana",
            "amount_compounded": 3.0,
            "tx_hash": "hash123",
            "timestamp": "2026-03-01T00:00:00+00:00",
            "new_total_stake": 103.0,
            "validator_address": "abc",
        }
        a = CompoundAction.from_dict(data)
        assert a.network == Network.SOLANA
        assert a.amount_compounded == 3.0
        assert a.tx_hash == "hash123"

    def test_from_dict_defaults(self):
        data = {"network": "sui"}
        a = CompoundAction.from_dict(data)
        assert a.amount_compounded == 0.0
        assert a.tx_hash == ""
        assert a.new_total_stake == 0.0

    def test_from_dict_no_timestamp(self):
        data = {"network": "sui", "amount_compounded": 1.0}
        a = CompoundAction.from_dict(data)
        assert a.timestamp is not None

    def test_roundtrip(self):
        a = CompoundAction(
            network=Network.ETHEREUM,
            amount_compounded=1.0,
            tx_hash="0xhash",
            new_total_stake=33.0,
            validator_address="rocketpool",
        )
        restored = CompoundAction.from_dict(a.to_dict())
        assert restored.network == a.network
        assert restored.amount_compounded == a.amount_compounded
        assert restored.tx_hash == a.tx_hash


class TestNetworkInfo:
    def test_create_info(self):
        info = NetworkInfo(
            network=Network.SUI,
            display_name="Sui",
            native_token="SUI",
        )
        assert info.network == Network.SUI
        assert info.display_name == "Sui"
        assert info.native_token == "SUI"
        assert info.avg_apy_pct == 0.0
        assert info.supports_auto_compound is False

    def test_to_dict(self):
        info = NetworkInfo(
            network=Network.ETHEREUM,
            display_name="Ethereum",
            native_token="ETH",
            avg_apy_pct=3.8,
            min_stake=0.01,
            unbonding_days=3,
            supports_auto_compound=True,
            rpc_url="https://eth.rpc",
        )
        d = info.to_dict()
        assert d["network"] == "ethereum"
        assert d["avg_apy_pct"] == 3.8
        assert d["supports_auto_compound"] is True

    def test_from_dict(self):
        data = {
            "network": "solana",
            "display_name": "Solana",
            "native_token": "SOL",
            "avg_apy_pct": 7.0,
            "unbonding_days": 2,
        }
        info = NetworkInfo.from_dict(data)
        assert info.network == Network.SOLANA
        assert info.avg_apy_pct == 7.0

    def test_from_dict_defaults(self):
        data = {"network": "sui", "display_name": "Sui", "native_token": "SUI"}
        info = NetworkInfo.from_dict(data)
        assert info.avg_apy_pct == 0.0
        assert info.min_stake == 0.0
        assert info.supports_auto_compound is False

    def test_roundtrip(self):
        info = NetworkInfo(
            network=Network.AVALANCHE,
            display_name="Avalanche",
            native_token="AVAX",
            avg_apy_pct=8.0,
            min_stake=25.0,
            unbonding_days=14,
        )
        restored = NetworkInfo.from_dict(info.to_dict())
        assert restored.network == info.network
        assert restored.avg_apy_pct == info.avg_apy_pct
