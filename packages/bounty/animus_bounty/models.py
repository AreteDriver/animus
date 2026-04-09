"""Data models for Animus Bounty."""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime


class Platform(str, enum.Enum):
    """Bounty source platforms."""

    GITCOIN = "gitcoin"
    GITHUB = "github"
    REPLIT = "replit"
    BOUNTING = "bounting"
    LAYER3 = "layer3"
    ONLYDUST = "onlydust"
    GENERIC = "generic"


class BountyStatus(str, enum.Enum):
    """Lifecycle of a bounty."""

    OPEN = "open"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    PAID = "paid"
    EXPIRED = "expired"
    REJECTED = "rejected"


class Difficulty(str, enum.Enum):
    """Bounty difficulty level."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Bounty:
    """A discovered developer bounty or task."""

    title: str
    url: str
    platform: Platform
    description_snippet: str = ""
    reward_usd: float = 0.0
    reward_token: str = ""
    chain: str = ""
    skills_required: list[str] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    deadline: datetime | None = None
    auto_claimable: bool = False
    status: BountyStatus = BountyStatus.OPEN
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: list[str] = field(default_factory=list)
    repo_url: str = ""
    raw_data: dict = field(default_factory=dict)

    @property
    def bounty_id(self) -> str:
        """Deterministic ID from URL for dedup."""
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        now = datetime.now(UTC)
        if self.deadline.tzinfo is None:
            return False
        return now > self.deadline

    @property
    def is_low_effort(self) -> bool:
        """Heuristic: documentation, typo, or good-first-issue bounties."""
        low_effort_labels = {
            "good first issue",
            "good-first-issue",
            "documentation",
            "docs",
            "typo",
            "beginner",
            "easy",
            "hacktoberfest",
        }
        label_set = {label.lower() for label in self.labels}
        if label_set & low_effort_labels:
            return True
        if self.difficulty in (Difficulty.TRIVIAL, Difficulty.EASY):
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "bounty_id": self.bounty_id,
            "title": self.title,
            "url": self.url,
            "platform": self.platform.value,
            "description_snippet": self.description_snippet,
            "reward_usd": self.reward_usd,
            "reward_token": self.reward_token,
            "chain": self.chain,
            "skills_required": self.skills_required,
            "difficulty": self.difficulty.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "auto_claimable": self.auto_claimable,
            "status": self.status.value,
            "discovered_at": self.discovered_at.isoformat(),
            "labels": self.labels,
            "repo_url": self.repo_url,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Bounty:
        return cls(
            title=data["title"],
            url=data["url"],
            platform=Platform(data["platform"]),
            description_snippet=data.get("description_snippet", ""),
            reward_usd=data.get("reward_usd", 0.0),
            reward_token=data.get("reward_token", ""),
            chain=data.get("chain", ""),
            skills_required=data.get("skills_required", []),
            difficulty=Difficulty(data.get("difficulty", "medium")),
            deadline=(datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None),
            auto_claimable=data.get("auto_claimable", False),
            status=BountyStatus(data.get("status", "open")),
            discovered_at=(
                datetime.fromisoformat(data["discovered_at"])
                if data.get("discovered_at")
                else datetime.now(UTC)
            ),
            labels=data.get("labels", []),
            repo_url=data.get("repo_url", ""),
            raw_data=data.get("raw_data", {}),
        )


@dataclass
class SkillMatch:
    """LLM-scored skill matching result for a bounty."""

    bounty_id: str
    match_score: float  # 0-1, how well skills match
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    estimated_hours: float = 0.0
    reasoning: str = ""
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "bounty_id": self.bounty_id,
            "match_score": self.match_score,
            "matching_skills": self.matching_skills,
            "missing_skills": self.missing_skills,
            "estimated_hours": self.estimated_hours,
            "reasoning": self.reasoning,
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillMatch:
        return cls(
            bounty_id=data["bounty_id"],
            match_score=data.get("match_score", 0.0),
            matching_skills=data.get("matching_skills", []),
            missing_skills=data.get("missing_skills", []),
            estimated_hours=data.get("estimated_hours", 0.0),
            reasoning=data.get("reasoning", ""),
            evaluated_at=(
                datetime.fromisoformat(data["evaluated_at"])
                if data.get("evaluated_at")
                else datetime.now(UTC)
            ),
        )


@dataclass
class ClaimResult:
    """Tracks the lifecycle of a claimed bounty."""

    bounty_id: str
    status: BountyStatus = BountyStatus.CLAIMED
    claimed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_at: datetime | None = None
    paid_at: datetime | None = None
    actual_reward: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "bounty_id": self.bounty_id,
            "status": self.status.value,
            "claimed_at": self.claimed_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "actual_reward": self.actual_reward,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClaimResult:
        return cls(
            bounty_id=data["bounty_id"],
            status=BountyStatus(data.get("status", "claimed")),
            claimed_at=(
                datetime.fromisoformat(data["claimed_at"])
                if data.get("claimed_at")
                else datetime.now(UTC)
            ),
            submitted_at=(
                datetime.fromisoformat(data["submitted_at"]) if data.get("submitted_at") else None
            ),
            paid_at=(datetime.fromisoformat(data["paid_at"]) if data.get("paid_at") else None),
            actual_reward=data.get("actual_reward", 0.0),
            notes=data.get("notes", ""),
        )
