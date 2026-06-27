"""Tests for animus_bounty.models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from animus_bounty.models import (
    Bounty,
    BountyStatus,
    ClaimResult,
    Difficulty,
    Platform,
    SkillMatch,
)


class TestPlatformEnum:
    def test_all_values(self):
        assert Platform.GITCOIN.value == "gitcoin"
        assert Platform.GITHUB.value == "github"
        assert Platform.REPLIT.value == "replit"
        assert Platform.BOUNTING.value == "bounting"
        assert Platform.LAYER3.value == "layer3"
        assert Platform.ONLYDUST.value == "onlydust"
        assert Platform.GENERIC.value == "generic"

    def test_from_string(self):
        assert Platform("github") == Platform.GITHUB


class TestBountyStatusEnum:
    def test_all_values(self):
        assert BountyStatus.OPEN.value == "open"
        assert BountyStatus.CLAIMED.value == "claimed"
        assert BountyStatus.IN_PROGRESS.value == "in_progress"
        assert BountyStatus.SUBMITTED.value == "submitted"
        assert BountyStatus.PAID.value == "paid"
        assert BountyStatus.EXPIRED.value == "expired"
        assert BountyStatus.REJECTED.value == "rejected"


class TestDifficultyEnum:
    def test_all_values(self):
        assert Difficulty.TRIVIAL.value == "trivial"
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"
        assert Difficulty.EXPERT.value == "expert"


class TestBounty:
    def test_bounty_id_deterministic(self, sample_bounty):
        """Same URL always produces same ID."""
        b2 = Bounty(
            title="Different title",
            url=sample_bounty.url,
            platform=Platform.GENERIC,
        )
        assert sample_bounty.bounty_id == b2.bounty_id

    def test_bounty_id_different_urls(self):
        b1 = Bounty(title="A", url="https://a.com/1", platform=Platform.GITHUB)
        b2 = Bounty(title="A", url="https://a.com/2", platform=Platform.GITHUB)
        assert b1.bounty_id != b2.bounty_id

    def test_is_expired_none_deadline(self, sample_bounty):
        assert sample_bounty.deadline is None
        assert not sample_bounty.is_expired

    def test_is_expired_past_deadline(self, expired_bounty):
        assert expired_bounty.is_expired

    def test_is_expired_future_deadline(self):
        b = Bounty(
            title="Future",
            url="https://example.com/future",
            platform=Platform.GITHUB,
            deadline=datetime.now(UTC) + timedelta(days=30),
        )
        assert not b.is_expired

    def test_is_expired_naive_deadline(self):
        """Naive datetime without tz returns False (can't compare safely)."""
        b = Bounty(
            title="Naive",
            url="https://example.com/naive",
            platform=Platform.GITHUB,
            deadline=datetime(2020, 1, 1),
        )
        assert not b.is_expired

    def test_is_low_effort_labels(self, sample_bounty):
        assert sample_bounty.is_low_effort

    def test_is_low_effort_difficulty(self):
        b = Bounty(
            title="Easy",
            url="https://example.com/easy",
            platform=Platform.GITHUB,
            difficulty=Difficulty.TRIVIAL,
        )
        assert b.is_low_effort

    def test_not_low_effort(self, hard_bounty):
        assert not hard_bounty.is_low_effort

    def test_is_low_effort_hacktoberfest(self):
        b = Bounty(
            title="Hacktoberfest",
            url="https://example.com/hf",
            platform=Platform.GITHUB,
            labels=["hacktoberfest"],
            difficulty=Difficulty.MEDIUM,
        )
        assert b.is_low_effort

    def test_to_dict(self, sample_bounty):
        d = sample_bounty.to_dict()
        assert d["bounty_id"] == sample_bounty.bounty_id
        assert d["title"] == "Fix typo in README"
        assert d["platform"] == "github"
        assert d["reward_usd"] == 25.0
        assert d["difficulty"] == "easy"
        assert d["deadline"] is None
        assert d["status"] == "open"
        assert d["labels"] == ["good first issue", "documentation", "bounty"]
        assert d["repo_url"] == "https://github.com/example/repo"

    def test_to_dict_with_deadline(self):
        dl = datetime(2026, 12, 31, tzinfo=UTC)
        b = Bounty(
            title="Deadline",
            url="https://example.com/dl",
            platform=Platform.GITHUB,
            deadline=dl,
        )
        d = b.to_dict()
        assert d["deadline"] == dl.isoformat()

    def test_from_dict(self, sample_bounty):
        d = sample_bounty.to_dict()
        restored = Bounty.from_dict(d)
        assert restored.title == sample_bounty.title
        assert restored.url == sample_bounty.url
        assert restored.platform == sample_bounty.platform
        assert restored.reward_usd == sample_bounty.reward_usd
        assert restored.difficulty == sample_bounty.difficulty
        assert restored.labels == sample_bounty.labels

    def test_from_dict_minimal(self):
        d = {"title": "Min", "url": "https://min.com", "platform": "generic"}
        b = Bounty.from_dict(d)
        assert b.title == "Min"
        assert b.platform == Platform.GENERIC
        assert b.reward_usd == 0.0
        assert b.difficulty == Difficulty.MEDIUM
        assert b.status == BountyStatus.OPEN

    def test_from_dict_with_deadline(self):
        d = {
            "title": "DL",
            "url": "https://dl.com",
            "platform": "github",
            "deadline": "2026-12-31T00:00:00+00:00",
        }
        b = Bounty.from_dict(d)
        assert b.deadline is not None
        assert b.deadline.year == 2026

    def test_roundtrip(self, sample_bounty):
        restored = Bounty.from_dict(sample_bounty.to_dict())
        assert restored.bounty_id == sample_bounty.bounty_id
        assert restored.to_dict()["title"] == sample_bounty.to_dict()["title"]

    def test_default_values(self):
        b = Bounty(title="T", url="https://t.com", platform=Platform.GITHUB)
        assert b.description_snippet == ""
        assert b.reward_usd == 0.0
        assert b.reward_token == ""
        assert b.chain == ""
        assert b.skills_required == []
        assert b.difficulty == Difficulty.MEDIUM
        assert b.deadline is None
        assert b.auto_claimable is False
        assert b.status == BountyStatus.OPEN
        assert b.labels == []
        assert b.repo_url == ""
        assert b.raw_data == {}


class TestSkillMatch:
    def test_to_dict(self, sample_match):
        d = sample_match.to_dict()
        assert d["match_score"] == 0.85
        assert d["matching_skills"] == ["python", "markdown"]
        assert d["missing_skills"] == []
        assert d["estimated_hours"] == 1.0

    def test_from_dict(self, sample_match):
        d = sample_match.to_dict()
        restored = SkillMatch.from_dict(d)
        assert restored.bounty_id == sample_match.bounty_id
        assert restored.match_score == 0.85
        assert restored.matching_skills == ["python", "markdown"]

    def test_from_dict_minimal(self):
        d = {"bounty_id": "abc123", "match_score": 0.5}
        m = SkillMatch.from_dict(d)
        assert m.bounty_id == "abc123"
        assert m.matching_skills == []
        assert m.estimated_hours == 0.0

    def test_roundtrip(self, sample_match):
        restored = SkillMatch.from_dict(sample_match.to_dict())
        assert restored.match_score == sample_match.match_score


class TestClaimResult:
    def test_to_dict(self, sample_claim):
        d = sample_claim.to_dict()
        assert d["status"] == "claimed"
        assert d["submitted_at"] is None
        assert d["paid_at"] is None
        assert d["actual_reward"] == 0.0

    def test_from_dict(self, sample_claim):
        d = sample_claim.to_dict()
        restored = ClaimResult.from_dict(d)
        assert restored.bounty_id == sample_claim.bounty_id
        assert restored.status == BountyStatus.CLAIMED

    def test_from_dict_with_dates(self):
        now = datetime.now(UTC)
        d = {
            "bounty_id": "test",
            "status": "paid",
            "claimed_at": now.isoformat(),
            "submitted_at": now.isoformat(),
            "paid_at": now.isoformat(),
            "actual_reward": 100.0,
            "notes": "Done",
        }
        c = ClaimResult.from_dict(d)
        assert c.status == BountyStatus.PAID
        assert c.actual_reward == 100.0
        assert c.submitted_at is not None
        assert c.paid_at is not None

    def test_from_dict_minimal(self):
        d = {"bounty_id": "x"}
        c = ClaimResult.from_dict(d)
        assert c.bounty_id == "x"
        assert c.status == BountyStatus.CLAIMED
        assert c.actual_reward == 0.0

    def test_roundtrip(self, sample_claim):
        restored = ClaimResult.from_dict(sample_claim.to_dict())
        assert restored.bounty_id == sample_claim.bounty_id
        assert restored.notes == sample_claim.notes
