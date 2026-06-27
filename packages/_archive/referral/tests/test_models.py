"""Tests for referral data models."""

from __future__ import annotations

from datetime import UTC, datetime

from animus_referral.models import (
    Account,
    BonusStatus,
    BonusType,
    ProgramType,
    ReferralBonus,
    ReferralLink,
    ReferralProgram,
)


class TestProgramType:
    def test_values(self):
        assert ProgramType.EXCHANGE == "exchange"
        assert ProgramType.DEFI == "defi"
        assert ProgramType.FAUCET == "faucet"
        assert ProgramType.WALLET == "wallet"
        assert ProgramType.QUEST_PLATFORM == "quest_platform"
        assert ProgramType.OTHER == "other"

    def test_from_value(self):
        assert ProgramType("exchange") == ProgramType.EXCHANGE

    def test_all_types(self):
        assert len(ProgramType) == 6


class TestBonusType:
    def test_values(self):
        assert BonusType.FLAT == "flat"
        assert BonusType.PERCENTAGE == "percentage"
        assert BonusType.TOKEN == "token"


class TestBonusStatus:
    def test_values(self):
        assert BonusStatus.PENDING == "pending"
        assert BonusStatus.PAID == "paid"
        assert BonusStatus.EXPIRED == "expired"


class TestReferralProgram:
    def test_create(self, sample_program):
        assert sample_program.name == "test_exchange"
        assert sample_program.program_type == ProgramType.EXCHANGE
        assert sample_program.bonus_referrer_usd == 10.0
        assert sample_program.active is True

    def test_to_dict(self, sample_program):
        d = sample_program.to_dict()
        assert d["name"] == "test_exchange"
        assert d["program_type"] == "exchange"
        assert d["bonus_type"] == "flat"
        assert d["active"] is True
        assert d["chain"] == "eth"

    def test_from_dict(self, sample_program):
        d = sample_program.to_dict()
        restored = ReferralProgram.from_dict(d)
        assert restored.name == sample_program.name
        assert restored.program_type == sample_program.program_type
        assert restored.bonus_referrer_usd == sample_program.bonus_referrer_usd
        assert restored.bonus_type == sample_program.bonus_type
        assert restored.chain == sample_program.chain

    def test_from_dict_defaults(self):
        d = {"name": "min", "platform_url": "https://x.com", "program_type": "other"}
        p = ReferralProgram.from_dict(d)
        assert p.bonus_referrer_usd == 0.0
        assert p.bonus_type == BonusType.FLAT
        assert p.active is True
        assert p.chain == ""

    def test_roundtrip(self, sample_program):
        d = sample_program.to_dict()
        restored = ReferralProgram.from_dict(d)
        assert restored.to_dict() == d


class TestAccount:
    def test_create(self, sample_account):
        assert sample_account.alias == "alice"
        assert sample_account.platform == "test_exchange"
        assert sample_account.wallet_address == "0xaaa"

    def test_to_dict(self, sample_account):
        d = sample_account.to_dict()
        assert d["alias"] == "alice"
        assert d["platform"] == "test_exchange"
        assert d["email_hash"] == "abc123"
        assert "created_at" in d

    def test_from_dict(self, sample_account):
        d = sample_account.to_dict()
        restored = Account.from_dict(d)
        assert restored.alias == sample_account.alias
        assert restored.platform == sample_account.platform
        assert restored.created_at == sample_account.created_at

    def test_from_dict_defaults(self):
        d = {
            "alias": "min",
            "platform": "test",
            "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        }
        a = Account.from_dict(d)
        assert a.email_hash == ""
        assert a.wallet_address == ""
        assert a.referral_code == ""
        assert a.referred_by == ""

    def test_roundtrip(self, sample_account):
        d = sample_account.to_dict()
        restored = Account.from_dict(d)
        assert restored.to_dict() == d

    def test_default_created_at(self):
        a = Account(alias="x", platform="y")
        assert a.created_at is not None
        assert a.created_at.tzinfo is not None


class TestReferralLink:
    def test_create(self, sample_link):
        assert sample_link.program_name == "test_exchange"
        assert sample_link.referrer_account == "alice"
        assert sample_link.clicks == 50
        assert sample_link.signups == 5

    def test_to_dict(self, sample_link):
        d = sample_link.to_dict()
        assert d["program_name"] == "test_exchange"
        assert d["referrer_account"] == "alice"
        assert d["earned_usd"] == 50.0

    def test_from_dict(self, sample_link):
        d = sample_link.to_dict()
        restored = ReferralLink.from_dict(d)
        assert restored.program_name == sample_link.program_name
        assert restored.clicks == sample_link.clicks

    def test_from_dict_defaults(self):
        d = {
            "program_name": "x",
            "referrer_account": "y",
            "link_url": "https://x.com",
            "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        }
        link = ReferralLink.from_dict(d)
        assert link.clicks == 0
        assert link.signups == 0
        assert link.earned_usd == 0.0

    def test_roundtrip(self, sample_link):
        d = sample_link.to_dict()
        restored = ReferralLink.from_dict(d)
        assert restored.to_dict() == d

    def test_default_created_at(self):
        link = ReferralLink(program_name="x", referrer_account="y", link_url="https://x.com")
        assert link.created_at is not None


class TestReferralBonus:
    def test_create(self, sample_bonus):
        assert sample_bonus.program_name == "test_exchange"
        assert sample_bonus.amount == 10.0
        assert sample_bonus.status == BonusStatus.PAID

    def test_to_dict(self, sample_bonus):
        d = sample_bonus.to_dict()
        assert d["status"] == "paid"
        assert d["token"] == "USDT"
        assert d["amount_usd"] == 10.0

    def test_from_dict(self, sample_bonus):
        d = sample_bonus.to_dict()
        restored = ReferralBonus.from_dict(d)
        assert restored.status == BonusStatus.PAID
        assert restored.amount == sample_bonus.amount

    def test_from_dict_defaults(self):
        d = {
            "program_name": "x",
            "account_alias": "y",
            "amount": 1.0,
            "earned_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        }
        b = ReferralBonus.from_dict(d)
        assert b.amount_usd == 0.0
        assert b.token == "USD"
        assert b.status == BonusStatus.PENDING

    def test_roundtrip(self, sample_bonus):
        d = sample_bonus.to_dict()
        restored = ReferralBonus.from_dict(d)
        assert restored.to_dict() == d

    def test_default_earned_at(self):
        b = ReferralBonus(program_name="x", account_alias="y", amount=1.0)
        assert b.earned_at is not None
