"""Shared test fixtures for animus_bounty."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_bounty.config import BountyConfig
from animus_bounty.models import (
    Bounty,
    BountyStatus,
    ClaimResult,
    Difficulty,
    Platform,
    SkillMatch,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    data_dir = tmp_path / "bounty_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_config(tmp_data_dir):
    return BountyConfig(data_dir=tmp_data_dir)


@pytest.fixture
def sample_bounty():
    return Bounty(
        title="Fix typo in README",
        url="https://github.com/example/repo/issues/42",
        platform=Platform.GITHUB,
        description_snippet="There is a typo in the installation section of the README.",
        reward_usd=25.0,
        skills_required=["python", "markdown"],
        difficulty=Difficulty.EASY,
        auto_claimable=True,
        labels=["good first issue", "documentation", "bounty"],
        repo_url="https://github.com/example/repo",
    )


@pytest.fixture
def hard_bounty():
    return Bounty(
        title="Implement ZK proof verification in Solidity",
        url="https://gitcoin.co/bounty/9999",
        platform=Platform.GITCOIN,
        description_snippet="Implement a ZK-SNARK verifier contract for the rollup bridge.",
        reward_usd=5000.0,
        reward_token="ETH",
        chain="eth",
        skills_required=["solidity", "cryptography", "zk-proofs"],
        difficulty=Difficulty.EXPERT,
        labels=["cryptography", "smart-contracts"],
    )


@pytest.fixture
def expired_bounty():
    return Bounty(
        title="Expired task",
        url="https://example.com/expired",
        platform=Platform.GENERIC,
        deadline=datetime.now(UTC) - timedelta(days=7),
    )


@pytest.fixture
def sample_match(sample_bounty):
    return SkillMatch(
        bounty_id=sample_bounty.bounty_id,
        match_score=0.85,
        matching_skills=["python", "markdown"],
        missing_skills=[],
        estimated_hours=1.0,
        reasoning="Strong match for documentation task",
    )


@pytest.fixture
def sample_claim(sample_bounty):
    return ClaimResult(
        bounty_id=sample_bounty.bounty_id,
        status=BountyStatus.CLAIMED,
        notes="Claimed via auto-claim",
    )
