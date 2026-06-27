"""Data models for Animus Prospector."""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime


class OpportunityType(str, enum.Enum):
    """Categories of crypto opportunities."""

    FAUCET = "faucet"
    AIRDROP = "airdrop"
    GIVEAWAY = "giveaway"
    QUEST = "quest"
    BOUNTY = "bounty"
    GRANT = "grant"
    TESTNET = "testnet"
    REFERRAL = "referral"
    DEFI_YIELD = "defi_yield"


class Chain(str, enum.Enum):
    """Blockchain networks."""

    BTC = "btc"
    ETH = "eth"
    BASE = "base"
    BASE_SEPOLIA = "base_sepolia"
    SUI = "sui"
    SUI_TESTNET = "sui_testnet"
    SOL = "sol"
    ARB = "arb"
    OP = "op"
    MATIC = "matic"
    MULTI = "multi"
    UNKNOWN = "unknown"


class Action(str, enum.Enum):
    """What to do with a discovered opportunity."""

    AUTO_EXECUTE = "auto_execute"
    ADD_TO_FAUCET = "add_to_faucet"
    FLAG_FOR_REVIEW = "flag_for_review"
    SKIP = "skip"
    EXPIRED = "expired"


class SourceType(str, enum.Enum):
    """Where opportunities are discovered."""

    TWITTER = "twitter"
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    WEB_SCRAPE = "web_scrape"
    RSS = "rss"
    API = "api"
    AGGREGATOR = "aggregator"
    MANUAL = "manual"


class OpportunityStatus(str, enum.Enum):
    """Lifecycle of an opportunity."""

    DISCOVERED = "discovered"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


@dataclass
class Opportunity:
    """A discovered crypto funding opportunity."""

    title: str
    url: str
    opportunity_type: OpportunityType
    source_type: SourceType
    source_url: str = ""
    chain: Chain = Chain.UNKNOWN
    estimated_value_usd: float = 0.0
    effort_minutes: int = 0
    risk_score: float = 0.0  # 0=safe, 1=scam
    expiry: datetime | None = None
    requirements: list[str] = field(default_factory=list)
    auto_actionable: bool = False
    status: OpportunityStatus = OpportunityStatus.DISCOVERED
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_data: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def opportunity_id(self) -> str:
        """Deterministic ID from URL for dedup."""
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]

    @property
    def roi_score(self) -> float:
        """Value per minute of effort. Higher is better."""
        if self.effort_minutes <= 0:
            return self.estimated_value_usd * 100 if self.estimated_value_usd > 0 else 0.0
        return self.estimated_value_usd / self.effort_minutes

    @property
    def is_expired(self) -> bool:
        if self.expiry is None:
            return False
        now = datetime.now(UTC)
        if self.expiry.tzinfo is None:
            return False
        return now > self.expiry

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "title": self.title,
            "url": self.url,
            "opportunity_type": self.opportunity_type.value,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "chain": self.chain.value,
            "estimated_value_usd": self.estimated_value_usd,
            "effort_minutes": self.effort_minutes,
            "risk_score": self.risk_score,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "requirements": self.requirements,
            "auto_actionable": self.auto_actionable,
            "status": self.status.value,
            "discovered_at": self.discovered_at.isoformat(),
            "raw_data": self.raw_data,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Opportunity:
        return cls(
            title=data["title"],
            url=data["url"],
            opportunity_type=OpportunityType(data["opportunity_type"]),
            source_type=SourceType(data["source_type"]),
            source_url=data.get("source_url", ""),
            chain=Chain(data.get("chain", "unknown")),
            estimated_value_usd=data.get("estimated_value_usd", 0.0),
            effort_minutes=data.get("effort_minutes", 0),
            risk_score=data.get("risk_score", 0.0),
            expiry=(datetime.fromisoformat(data["expiry"]) if data.get("expiry") else None),
            requirements=data.get("requirements", []),
            auto_actionable=data.get("auto_actionable", False),
            status=OpportunityStatus(data.get("status", "discovered")),
            discovered_at=(
                datetime.fromisoformat(data["discovered_at"])
                if data.get("discovered_at")
                else datetime.now(UTC)
            ),
            raw_data=data.get("raw_data", {}),
            tags=data.get("tags", []),
        )


@dataclass
class EvaluationResult:
    """LLM-scored evaluation of an opportunity."""

    opportunity_id: str
    roi_score: float  # value / effort
    scam_probability: float  # 0-1
    recommended_action: Action
    reasoning: str
    confidence: float = 0.0  # 0-1
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "roi_score": self.roi_score,
            "scam_probability": self.scam_probability,
            "recommended_action": self.recommended_action.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationResult:
        return cls(
            opportunity_id=data["opportunity_id"],
            roi_score=data.get("roi_score", 0.0),
            scam_probability=data.get("scam_probability", 0.5),
            recommended_action=Action(data["recommended_action"]),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.0),
            evaluated_at=(
                datetime.fromisoformat(data["evaluated_at"])
                if data.get("evaluated_at")
                else datetime.now(UTC)
            ),
        )


@dataclass
class HunterResult:
    """Output from a hunter scan."""

    hunter_name: str
    opportunities: list[Opportunity] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    scan_duration_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.opportunities)

    @property
    def new_count(self) -> int:
        return sum(1 for o in self.opportunities if o.status == OpportunityStatus.DISCOVERED)
