"""Data models for Animus Validator."""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime


class Network(str, enum.Enum):
    """Supported blockchain networks."""

    SUI = "sui"
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    COSMOS = "cosmos"
    AVALANCHE = "avalanche"
    POLYGON = "polygon"


class StakeStatus(str, enum.Enum):
    """Lifecycle of a staking position."""

    ACTIVE = "active"
    UNSTAKING = "unstaking"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class RewardType(str, enum.Enum):
    """Types of validator rewards."""

    STAKING = "staking"
    VALIDATION = "validation"
    DELEGATION = "delegation"


@dataclass
class Node:
    """A validator or staking node."""

    network: Network
    address: str
    stake_amount: float
    rewards_earned: float = 0.0
    uptime_pct: float = 100.0
    last_check: datetime | None = None
    status: StakeStatus = StakeStatus.ACTIVE
    name: str = ""

    @property
    def node_id(self) -> str:
        """Deterministic ID from network + address."""
        raw = f"{self.network.value}:{self.address}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def is_healthy(self) -> bool:
        """Node is healthy if uptime > 95% and active."""
        return self.uptime_pct >= 95.0 and self.status == StakeStatus.ACTIVE

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "network": self.network.value,
            "address": self.address,
            "stake_amount": self.stake_amount,
            "rewards_earned": self.rewards_earned,
            "uptime_pct": self.uptime_pct,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "status": self.status.value,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        return cls(
            network=Network(data["network"]),
            address=data["address"],
            stake_amount=data.get("stake_amount", 0.0),
            rewards_earned=data.get("rewards_earned", 0.0),
            uptime_pct=data.get("uptime_pct", 100.0),
            last_check=(
                datetime.fromisoformat(data["last_check"]) if data.get("last_check") else None
            ),
            status=StakeStatus(data.get("status", "active")),
            name=data.get("name", ""),
        )


@dataclass
class StakePosition:
    """A staking position on a validator."""

    network: Network
    validator_address: str
    amount: float
    apy_pct: float = 0.0
    rewards_pending: float = 0.0
    auto_compound: bool = False
    staked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def position_id(self) -> str:
        """Deterministic ID from network + validator."""
        raw = f"{self.network.value}:{self.validator_address}:{self.amount}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def annual_yield_estimate(self) -> float:
        """Estimated annual yield in token terms."""
        return self.amount * (self.apy_pct / 100.0)

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "network": self.network.value,
            "validator_address": self.validator_address,
            "amount": self.amount,
            "apy_pct": self.apy_pct,
            "rewards_pending": self.rewards_pending,
            "auto_compound": self.auto_compound,
            "staked_at": self.staked_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StakePosition:
        return cls(
            network=Network(data["network"]),
            validator_address=data["validator_address"],
            amount=data.get("amount", 0.0),
            apy_pct=data.get("apy_pct", 0.0),
            rewards_pending=data.get("rewards_pending", 0.0),
            auto_compound=data.get("auto_compound", False),
            staked_at=(
                datetime.fromisoformat(data["staked_at"])
                if data.get("staked_at")
                else datetime.now(UTC)
            ),
        )


@dataclass
class RewardRecord:
    """A recorded reward event."""

    network: Network
    amount: float
    amount_usd: float = 0.0
    reward_type: RewardType = RewardType.STAKING
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    validator_address: str = ""

    @property
    def record_id(self) -> str:
        raw = f"{self.network.value}:{self.validator_address}:{self.timestamp.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "network": self.network.value,
            "amount": self.amount,
            "amount_usd": self.amount_usd,
            "reward_type": self.reward_type.value,
            "timestamp": self.timestamp.isoformat(),
            "validator_address": self.validator_address,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RewardRecord:
        return cls(
            network=Network(data["network"]),
            amount=data.get("amount", 0.0),
            amount_usd=data.get("amount_usd", 0.0),
            reward_type=RewardType(data.get("reward_type", "staking")),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now(UTC)
            ),
            validator_address=data.get("validator_address", ""),
        )


@dataclass
class CompoundAction:
    """Record of a compound (restake) action."""

    network: Network
    amount_compounded: float
    tx_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    new_total_stake: float = 0.0
    validator_address: str = ""

    def to_dict(self) -> dict:
        return {
            "network": self.network.value,
            "amount_compounded": self.amount_compounded,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp.isoformat(),
            "new_total_stake": self.new_total_stake,
            "validator_address": self.validator_address,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompoundAction:
        return cls(
            network=Network(data["network"]),
            amount_compounded=data.get("amount_compounded", 0.0),
            tx_hash=data.get("tx_hash", ""),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now(UTC)
            ),
            new_total_stake=data.get("new_total_stake", 0.0),
            validator_address=data.get("validator_address", ""),
        )


@dataclass
class NetworkInfo:
    """Metadata about a supported network."""

    network: Network
    display_name: str
    native_token: str
    avg_apy_pct: float = 0.0
    min_stake: float = 0.0
    unbonding_days: int = 0
    supports_auto_compound: bool = False
    rpc_url: str = ""

    def to_dict(self) -> dict:
        return {
            "network": self.network.value,
            "display_name": self.display_name,
            "native_token": self.native_token,
            "avg_apy_pct": self.avg_apy_pct,
            "min_stake": self.min_stake,
            "unbonding_days": self.unbonding_days,
            "supports_auto_compound": self.supports_auto_compound,
            "rpc_url": self.rpc_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NetworkInfo:
        return cls(
            network=Network(data["network"]),
            display_name=data.get("display_name", ""),
            native_token=data.get("native_token", ""),
            avg_apy_pct=data.get("avg_apy_pct", 0.0),
            min_stake=data.get("min_stake", 0.0),
            unbonding_days=data.get("unbonding_days", 0),
            supports_auto_compound=data.get("supports_auto_compound", False),
            rpc_url=data.get("rpc_url", ""),
        )
