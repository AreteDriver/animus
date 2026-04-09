"""Data models for Animus Referral."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime


class ProgramType(str, enum.Enum):
    """Category of referral program."""

    EXCHANGE = "exchange"
    DEFI = "defi"
    FAUCET = "faucet"
    WALLET = "wallet"
    QUEST_PLATFORM = "quest_platform"
    OTHER = "other"


class BonusType(str, enum.Enum):
    """How the referral bonus is calculated."""

    FLAT = "flat"
    PERCENTAGE = "percentage"
    TOKEN = "token"


class BonusStatus(str, enum.Enum):
    """Status of a referral bonus payout."""

    PENDING = "pending"
    PAID = "paid"
    EXPIRED = "expired"


@dataclass
class ReferralProgram:
    """Definition of a referral program on a platform."""

    name: str
    platform_url: str
    program_type: ProgramType
    bonus_referrer_usd: float = 0.0
    bonus_referee_usd: float = 0.0
    bonus_type: BonusType = BonusType.FLAT
    requirements: str = ""
    referral_url_template: str = ""
    active: bool = True
    chain: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "platform_url": self.platform_url,
            "program_type": self.program_type.value,
            "bonus_referrer_usd": self.bonus_referrer_usd,
            "bonus_referee_usd": self.bonus_referee_usd,
            "bonus_type": self.bonus_type.value,
            "requirements": self.requirements,
            "referral_url_template": self.referral_url_template,
            "active": self.active,
            "chain": self.chain,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReferralProgram:
        return cls(
            name=data["name"],
            platform_url=data["platform_url"],
            program_type=ProgramType(data["program_type"]),
            bonus_referrer_usd=data.get("bonus_referrer_usd", 0.0),
            bonus_referee_usd=data.get("bonus_referee_usd", 0.0),
            bonus_type=BonusType(data.get("bonus_type", "flat")),
            requirements=data.get("requirements", ""),
            referral_url_template=data.get("referral_url_template", ""),
            active=data.get("active", True),
            chain=data.get("chain", ""),
        )


@dataclass
class Account:
    """An account on a platform that can participate in referral programs."""

    alias: str
    platform: str
    email_hash: str = ""
    wallet_address: str = ""
    referral_code: str = ""
    referred_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "alias": self.alias,
            "platform": self.platform,
            "email_hash": self.email_hash,
            "wallet_address": self.wallet_address,
            "referral_code": self.referral_code,
            "referred_by": self.referred_by,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Account:
        return cls(
            alias=data["alias"],
            platform=data["platform"],
            email_hash=data.get("email_hash", ""),
            wallet_address=data.get("wallet_address", ""),
            referral_code=data.get("referral_code", ""),
            referred_by=data.get("referred_by", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class ReferralLink:
    """A generated referral link for a specific program/account pair."""

    program_name: str
    referrer_account: str
    link_url: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    clicks: int = 0
    signups: int = 0
    earned_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "program_name": self.program_name,
            "referrer_account": self.referrer_account,
            "link_url": self.link_url,
            "created_at": self.created_at.isoformat(),
            "clicks": self.clicks,
            "signups": self.signups,
            "earned_usd": self.earned_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReferralLink:
        return cls(
            program_name=data["program_name"],
            referrer_account=data["referrer_account"],
            link_url=data["link_url"],
            created_at=datetime.fromisoformat(data["created_at"]),
            clicks=data.get("clicks", 0),
            signups=data.get("signups", 0),
            earned_usd=data.get("earned_usd", 0.0),
        )


@dataclass
class ReferralBonus:
    """A bonus earned from a referral conversion."""

    program_name: str
    account_alias: str
    amount: float
    amount_usd: float = 0.0
    token: str = "USD"
    status: BonusStatus = BonusStatus.PENDING
    earned_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "program_name": self.program_name,
            "account_alias": self.account_alias,
            "amount": self.amount,
            "amount_usd": self.amount_usd,
            "token": self.token,
            "status": self.status.value,
            "earned_at": self.earned_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReferralBonus:
        return cls(
            program_name=data["program_name"],
            account_alias=data["account_alias"],
            amount=data["amount"],
            amount_usd=data.get("amount_usd", 0.0),
            token=data.get("token", "USD"),
            status=BonusStatus(data.get("status", "pending")),
            earned_at=datetime.fromisoformat(data["earned_at"]),
        )
