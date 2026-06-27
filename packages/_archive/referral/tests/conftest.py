"""Shared fixtures for referral tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from animus_referral.config import ReferralConfig
from animus_referral.manager import ReferralManager
from animus_referral.models import (
    Account,
    BonusStatus,
    BonusType,
    ProgramType,
    ReferralBonus,
    ReferralLink,
    ReferralProgram,
)
from animus_referral.store import AccountStore, BonusStore, LinkStore
from animus_referral.tracker import ReferralTracker


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "referral_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def config(tmp_data_dir: Path) -> ReferralConfig:
    """Config pointing to temp dir."""
    cfg = ReferralConfig(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def manager(config: ReferralConfig, tmp_data_dir: Path) -> ReferralManager:
    """Manager with temp storage."""
    return ReferralManager(config, data_dir=tmp_data_dir)


@pytest.fixture
def tracker(tmp_data_dir: Path) -> ReferralTracker:
    """Tracker with temp storage."""
    return ReferralTracker(tmp_data_dir)


@pytest.fixture
def account_store(tmp_data_dir: Path) -> AccountStore:
    return AccountStore(tmp_data_dir)


@pytest.fixture
def link_store(tmp_data_dir: Path) -> LinkStore:
    return LinkStore(tmp_data_dir)


@pytest.fixture
def bonus_store(tmp_data_dir: Path) -> BonusStore:
    return BonusStore(tmp_data_dir)


@pytest.fixture
def sample_program() -> ReferralProgram:
    return ReferralProgram(
        name="test_exchange",
        platform_url="https://test.exchange",
        program_type=ProgramType.EXCHANGE,
        bonus_referrer_usd=10.0,
        bonus_referee_usd=5.0,
        bonus_type=BonusType.FLAT,
        requirements="Must trade $100",
        referral_url_template="https://test.exchange/join/{code}",
        active=True,
        chain="eth",
    )


@pytest.fixture
def sample_account() -> Account:
    return Account(
        alias="alice",
        platform="test_exchange",
        email_hash="abc123",
        wallet_address="0xaaa",
        referral_code="ALICE123",
        referred_by="",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def sample_account_bob() -> Account:
    return Account(
        alias="bob",
        platform="test_exchange",
        email_hash="def456",
        wallet_address="0xbbb",
        referral_code="BOB456",
        referred_by="alice",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )


@pytest.fixture
def sample_link() -> ReferralLink:
    return ReferralLink(
        program_name="test_exchange",
        referrer_account="alice",
        link_url="https://test.exchange/join/ALICE123",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        clicks=50,
        signups=5,
        earned_usd=50.0,
    )


@pytest.fixture
def sample_bonus() -> ReferralBonus:
    return ReferralBonus(
        program_name="test_exchange",
        account_alias="alice",
        amount=10.0,
        amount_usd=10.0,
        token="USDT",
        status=BonusStatus.PAID,
        earned_at=datetime(2026, 1, 5, tzinfo=UTC),
    )


@pytest.fixture
def sample_bonus_pending() -> ReferralBonus:
    return ReferralBonus(
        program_name="test_exchange",
        account_alias="bob",
        amount=5.0,
        amount_usd=5.0,
        token="USDT",
        status=BonusStatus.PENDING,
        earned_at=datetime(2026, 3, 1, tzinfo=UTC),
    )
