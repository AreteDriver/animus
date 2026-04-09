"""Data models for Animus Faucet."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime


class Chain(str, enum.Enum):
    """Supported blockchain networks."""

    BTC = "btc"
    ETH = "eth"
    BASE_SEPOLIA = "base_sepolia"
    SUI_TESTNET = "sui_testnet"
    SOL = "sol"
    DOGE = "doge"
    LTC = "ltc"
    MATIC = "matic"


class CaptchaType(str, enum.Enum):
    """Types of captcha encountered on faucets."""

    NONE = "none"
    HCAPTCHA = "hcaptcha"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    TURNSTILE = "turnstile"
    CUSTOM = "custom"


class ClaimStatus(str, enum.Enum):
    """Result of a faucet claim attempt."""

    SUCCESS = "success"
    CAPTCHA_FAILED = "captcha_failed"
    RATE_LIMITED = "rate_limited"
    BANNED = "banned"
    NETWORK_ERROR = "network_error"
    FAUCET_EMPTY = "faucet_empty"
    FAUCET_DOWN = "faucet_down"
    UNKNOWN_ERROR = "unknown_error"


class DriverType(str, enum.Enum):
    """How to interact with the faucet."""

    API = "api"
    BROWSER = "browser"
    CLI = "cli"


@dataclass
class FaucetDefinition:
    """Definition of a faucet to farm."""

    name: str
    url: str
    chain: Chain
    driver_type: DriverType
    captcha_type: CaptchaType = CaptchaType.NONE
    schedule_cron: str = "0 * * * *"  # hourly default
    expected_payout_usd: float = 0.01
    enabled: bool = True
    requires_proxy: bool = False
    notes: str = ""
    # Driver-specific config
    driver_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "chain": self.chain.value,
            "driver_type": self.driver_type.value,
            "captcha_type": self.captcha_type.value,
            "schedule_cron": self.schedule_cron,
            "expected_payout_usd": self.expected_payout_usd,
            "enabled": self.enabled,
            "requires_proxy": self.requires_proxy,
            "notes": self.notes,
            "driver_config": self.driver_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FaucetDefinition:
        return cls(
            name=data["name"],
            url=data["url"],
            chain=Chain(data["chain"]),
            driver_type=DriverType(data["driver_type"]),
            captcha_type=CaptchaType(data.get("captcha_type", "none")),
            schedule_cron=data.get("schedule_cron", "0 * * * *"),
            expected_payout_usd=data.get("expected_payout_usd", 0.01),
            enabled=data.get("enabled", True),
            requires_proxy=data.get("requires_proxy", False),
            notes=data.get("notes", ""),
            driver_config=data.get("driver_config", {}),
        )


@dataclass
class ClaimResult:
    """Result of a single claim attempt."""

    faucet_name: str
    chain: Chain
    status: ClaimStatus
    amount: float | None = None
    amount_usd: float | None = None
    tx_hash: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_detail: str | None = None
    proxy_used: str | None = None

    def to_dict(self) -> dict:
        return {
            "faucet_name": self.faucet_name,
            "chain": self.chain.value,
            "status": self.status.value,
            "amount": self.amount,
            "amount_usd": self.amount_usd,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp.isoformat(),
            "error_detail": self.error_detail,
            "proxy_used": self.proxy_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClaimResult:
        return cls(
            faucet_name=data["faucet_name"],
            chain=Chain(data["chain"]),
            status=ClaimStatus(data["status"]),
            amount=data.get("amount"),
            amount_usd=data.get("amount_usd"),
            tx_hash=data.get("tx_hash"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error_detail=data.get("error_detail"),
            proxy_used=data.get("proxy_used"),
        )

    @property
    def succeeded(self) -> bool:
        return self.status == ClaimStatus.SUCCESS


@dataclass
class FaucetHealth:
    """Health status of a faucet."""

    name: str
    enabled: bool
    last_claim_at: datetime | None = None
    last_status: ClaimStatus | None = None
    success_count: int = 0
    fail_count: int = 0
    total_earned_usd: float = 0.0
    consecutive_failures: int = 0
    banned: bool = False

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def should_disable(self) -> bool:
        """Auto-disable after 10 consecutive failures or if banned."""
        return self.banned or self.consecutive_failures >= 10

    def record_claim(self, result: ClaimResult) -> None:
        """Update health from a claim result."""
        self.last_claim_at = result.timestamp
        self.last_status = result.status

        if result.succeeded:
            self.success_count += 1
            self.consecutive_failures = 0
            if result.amount_usd:
                self.total_earned_usd += result.amount_usd
        else:
            self.fail_count += 1
            self.consecutive_failures += 1
            if result.status == ClaimStatus.BANNED:
                self.banned = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "last_claim_at": self.last_claim_at.isoformat() if self.last_claim_at else None,
            "last_status": self.last_status.value if self.last_status else None,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "total_earned_usd": self.total_earned_usd,
            "consecutive_failures": self.consecutive_failures,
            "banned": self.banned,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FaucetHealth:
        return cls(
            name=data["name"],
            enabled=data["enabled"],
            last_claim_at=(
                datetime.fromisoformat(data["last_claim_at"]) if data.get("last_claim_at") else None
            ),
            last_status=(ClaimStatus(data["last_status"]) if data.get("last_status") else None),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            total_earned_usd=data.get("total_earned_usd", 0.0),
            consecutive_failures=data.get("consecutive_failures", 0),
            banned=data.get("banned", False),
        )


@dataclass
class TreasuryBalance:
    """Balance for a single chain wallet."""

    chain: Chain
    address: str
    balance: float
    balance_usd: float | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "chain": self.chain.value,
            "address": self.address,
            "balance": self.balance,
            "balance_usd": self.balance_usd,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreasuryBalance:
        return cls(
            chain=Chain(data["chain"]),
            address=data["address"],
            balance=data["balance"],
            balance_usd=data.get("balance_usd"),
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )
