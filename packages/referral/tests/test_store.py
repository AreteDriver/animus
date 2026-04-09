"""Tests for JSONL persistence stores."""

from __future__ import annotations

from datetime import UTC, datetime

from animus_referral.models import (
    Account,
    BonusStatus,
    ReferralBonus,
    ReferralLink,
)


class TestAccountStore:
    def test_save_and_load(self, account_store, sample_account):
        account_store.save(sample_account)
        accounts = account_store.load_all()
        assert len(accounts) == 1
        assert accounts[0].alias == "alice"

    def test_save_multiple(self, account_store, sample_account, sample_account_bob):
        account_store.save(sample_account)
        account_store.save(sample_account_bob)
        assert len(account_store.load_all()) == 2

    def test_save_update(self, account_store, sample_account):
        account_store.save(sample_account)
        updated = Account(
            alias="alice",
            platform="test_exchange",
            referral_code="NEWALICE",
            created_at=sample_account.created_at,
        )
        account_store.save(updated)
        accounts = account_store.load_all()
        assert len(accounts) == 1
        assert accounts[0].referral_code == "NEWALICE"

    def test_get(self, account_store, sample_account):
        account_store.save(sample_account)
        found = account_store.get("alice", "test_exchange")
        assert found is not None
        assert found.alias == "alice"

    def test_get_not_found(self, account_store):
        assert account_store.get("nobody", "nowhere") is None

    def test_get_by_alias(self, account_store, sample_account):
        account_store.save(sample_account)
        other = Account(alias="alice", platform="other_exchange", created_at=datetime.now(UTC))
        account_store.save(other)
        results = account_store.get_by_alias("alice")
        assert len(results) == 2

    def test_get_by_platform(self, account_store, sample_account, sample_account_bob):
        account_store.save(sample_account)
        account_store.save(sample_account_bob)
        results = account_store.get_by_platform("test_exchange")
        assert len(results) == 2

    def test_get_by_platform_empty(self, account_store):
        assert account_store.get_by_platform("nonexistent") == []

    def test_remove(self, account_store, sample_account):
        account_store.save(sample_account)
        assert account_store.remove("alice", "test_exchange") is True
        assert len(account_store.load_all()) == 0

    def test_remove_not_found(self, account_store):
        assert account_store.remove("nobody", "nowhere") is False

    def test_count(self, account_store, sample_account, sample_account_bob):
        assert account_store.count() == 0
        account_store.save(sample_account)
        assert account_store.count() == 1
        account_store.save(sample_account_bob)
        assert account_store.count() == 2

    def test_empty_load(self, account_store):
        assert account_store.load_all() == []


class TestLinkStore:
    def test_save_and_load(self, link_store, sample_link):
        link_store.save(sample_link)
        links = link_store.load_all()
        assert len(links) == 1
        assert links[0].link_url == sample_link.link_url

    def test_save_update(self, link_store, sample_link):
        link_store.save(sample_link)
        updated = ReferralLink(
            program_name="test_exchange",
            referrer_account="alice",
            link_url="https://new.url",
            created_at=sample_link.created_at,
            clicks=100,
        )
        link_store.save(updated)
        links = link_store.load_all()
        assert len(links) == 1
        assert links[0].clicks == 100

    def test_get(self, link_store, sample_link):
        link_store.save(sample_link)
        found = link_store.get("test_exchange", "alice")
        assert found is not None
        assert found.link_url == sample_link.link_url

    def test_get_not_found(self, link_store):
        assert link_store.get("x", "y") is None

    def test_get_for_program(self, link_store, sample_link):
        link_store.save(sample_link)
        other = ReferralLink(
            program_name="test_exchange",
            referrer_account="bob",
            link_url="https://other.url",
            created_at=datetime.now(UTC),
        )
        link_store.save(other)
        results = link_store.get_for_program("test_exchange")
        assert len(results) == 2

    def test_get_for_program_empty(self, link_store):
        assert link_store.get_for_program("nonexistent") == []

    def test_count(self, link_store, sample_link):
        assert link_store.count() == 0
        link_store.save(sample_link)
        assert link_store.count() == 1

    def test_empty_load(self, link_store):
        assert link_store.load_all() == []


class TestBonusStore:
    def test_record_and_load(self, bonus_store, sample_bonus):
        bonus_store.record(sample_bonus)
        bonuses = bonus_store.load_all()
        assert len(bonuses) == 1
        assert bonuses[0].amount == 10.0

    def test_record_multiple(self, bonus_store, sample_bonus, sample_bonus_pending):
        bonus_store.record(sample_bonus)
        bonus_store.record(sample_bonus_pending)
        assert len(bonus_store.load_all()) == 2

    def test_get_for_program(self, bonus_store, sample_bonus):
        bonus_store.record(sample_bonus)
        other = ReferralBonus(
            program_name="other_program",
            account_alias="alice",
            amount=5.0,
            earned_at=datetime.now(UTC),
        )
        bonus_store.record(other)
        results = bonus_store.get_for_program("test_exchange")
        assert len(results) == 1

    def test_get_for_account(self, bonus_store, sample_bonus, sample_bonus_pending):
        bonus_store.record(sample_bonus)
        bonus_store.record(sample_bonus_pending)
        results = bonus_store.get_for_account("alice")
        assert len(results) == 1
        assert results[0].amount == 10.0

    def test_total_earned_usd(self, bonus_store, sample_bonus, sample_bonus_pending):
        bonus_store.record(sample_bonus)
        bonus_store.record(sample_bonus_pending)
        # Only paid bonuses count
        total = bonus_store.total_earned_usd()
        assert total == 10.0

    def test_update_status(self, bonus_store, sample_bonus_pending):
        bonus_store.record(sample_bonus_pending)
        result = bonus_store.update_status(
            "test_exchange",
            "bob",
            sample_bonus_pending.earned_at.isoformat(),
            "paid",
        )
        assert result is True
        bonuses = bonus_store.load_all()
        assert bonuses[0].status == BonusStatus.PAID

    def test_update_status_not_found(self, bonus_store):
        result = bonus_store.update_status("x", "y", "2026-01-01T00:00:00+00:00", "paid")
        assert result is False

    def test_count(self, bonus_store, sample_bonus):
        assert bonus_store.count() == 0
        bonus_store.record(sample_bonus)
        assert bonus_store.count() == 1

    def test_empty_load(self, bonus_store):
        assert bonus_store.load_all() == []

    def test_get_for_program_empty(self, bonus_store):
        assert bonus_store.get_for_program("nonexistent") == []

    def test_get_for_account_empty(self, bonus_store):
        assert bonus_store.get_for_account("nonexistent") == []
