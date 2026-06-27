"""Tests for the referral tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from animus_referral.models import BonusStatus, ReferralBonus, ReferralLink
from animus_referral.store import BonusStore, LinkStore


class TestTracker:
    def test_total_earned_usd(self, tracker, tmp_data_dir, sample_bonus):
        store = BonusStore(tmp_data_dir)
        store.record(sample_bonus)
        assert tracker.total_earned_usd() == 10.0

    def test_total_earned_usd_excludes_pending(self, tracker, tmp_data_dir, sample_bonus_pending):
        store = BonusStore(tmp_data_dir)
        store.record(sample_bonus_pending)
        assert tracker.total_earned_usd() == 0.0

    def test_total_earned_usd_with_days(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        old = ReferralBonus(
            program_name="x",
            account_alias="a",
            amount=10.0,
            amount_usd=10.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC) - timedelta(days=60),
        )
        recent = ReferralBonus(
            program_name="x",
            account_alias="b",
            amount=5.0,
            amount_usd=5.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC) - timedelta(days=5),
        )
        store.record(old)
        store.record(recent)
        assert tracker.total_earned_usd(days=30) == 5.0
        assert tracker.total_earned_usd() == 15.0

    def test_total_pending_usd(self, tracker, tmp_data_dir, sample_bonus_pending):
        store = BonusStore(tmp_data_dir)
        store.record(sample_bonus_pending)
        assert tracker.total_pending_usd() == 5.0

    def test_earnings_by_program(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        b1 = ReferralBonus(
            program_name="prog_a",
            account_alias="a",
            amount=10.0,
            amount_usd=10.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC),
        )
        b2 = ReferralBonus(
            program_name="prog_b",
            account_alias="a",
            amount=5.0,
            amount_usd=5.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC),
        )
        store.record(b1)
        store.record(b2)
        by_prog = tracker.earnings_by_program()
        assert by_prog["prog_a"] == 10.0
        assert by_prog["prog_b"] == 5.0

    def test_earnings_by_program_with_days(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        old = ReferralBonus(
            program_name="prog_a",
            account_alias="a",
            amount=10.0,
            amount_usd=10.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC) - timedelta(days=60),
        )
        store.record(old)
        by_prog = tracker.earnings_by_program(days=30)
        assert by_prog == {}

    def test_earnings_by_account(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        b1 = ReferralBonus(
            program_name="prog",
            account_alias="alice",
            amount=10.0,
            amount_usd=10.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC),
        )
        b2 = ReferralBonus(
            program_name="prog",
            account_alias="bob",
            amount=5.0,
            amount_usd=5.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC),
        )
        store.record(b1)
        store.record(b2)
        by_acct = tracker.earnings_by_account()
        assert by_acct["alice"] == 10.0
        assert by_acct["bob"] == 5.0

    def test_earnings_by_account_with_days(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        old = ReferralBonus(
            program_name="prog",
            account_alias="alice",
            amount=10.0,
            amount_usd=10.0,
            status=BonusStatus.PAID,
            earned_at=datetime.now(UTC) - timedelta(days=60),
        )
        store.record(old)
        by_acct = tracker.earnings_by_account(days=30)
        assert by_acct == {}

    def test_conversion_rate(self, tracker, tmp_data_dir):
        store = LinkStore(tmp_data_dir)
        link = ReferralLink(
            program_name="x",
            referrer_account="a",
            link_url="https://x.com",
            created_at=datetime.now(UTC),
            clicks=100,
            signups=10,
        )
        store.save(link)
        assert tracker.conversion_rate() == 0.1

    def test_conversion_rate_zero_clicks(self, tracker):
        assert tracker.conversion_rate() == 0.0

    def test_top_programs(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        for i in range(7):
            b = ReferralBonus(
                program_name=f"prog_{i}",
                account_alias="a",
                amount=float(i * 10),
                amount_usd=float(i * 10),
                status=BonusStatus.PAID,
                earned_at=datetime.now(UTC),
            )
            store.record(b)
        top = tracker.top_programs(n=3)
        assert len(top) == 3
        assert top[0][1] >= top[1][1] >= top[2][1]

    def test_top_accounts(self, tracker, tmp_data_dir):
        store = BonusStore(tmp_data_dir)
        for alias in ["alice", "bob", "carol"]:
            b = ReferralBonus(
                program_name="prog",
                account_alias=alias,
                amount=10.0,
                amount_usd=10.0,
                status=BonusStatus.PAID,
                earned_at=datetime.now(UTC),
            )
            store.record(b)
        top = tracker.top_accounts(n=2)
        assert len(top) == 2

    def test_get_report(self, tracker, tmp_data_dir, sample_bonus):
        store = BonusStore(tmp_data_dir)
        store.record(sample_bonus)
        report = tracker.get_report(days=365)
        assert "total_earned_usd" in report
        assert "total_pending_usd" in report
        assert "by_program" in report
        assert "conversion_rate" in report
        assert report["period_days"] == 365

    def test_get_report_empty(self, tracker):
        report = tracker.get_report()
        assert report["total_earned_usd"] == 0.0
        assert report["total_pending_usd"] == 0.0
        assert report["by_program"] == {}

    def test_mark_bonus_paid(self, tracker, tmp_data_dir, sample_bonus_pending):
        store = BonusStore(tmp_data_dir)
        store.record(sample_bonus_pending)
        result = tracker.mark_bonus_paid("test_exchange", "bob", sample_bonus_pending.earned_at)
        assert result is True
        bonuses = store.load_all()
        assert bonuses[0].status == BonusStatus.PAID

    def test_mark_bonus_expired(self, tracker, tmp_data_dir, sample_bonus_pending):
        store = BonusStore(tmp_data_dir)
        store.record(sample_bonus_pending)
        result = tracker.mark_bonus_expired("test_exchange", "bob", sample_bonus_pending.earned_at)
        assert result is True
        bonuses = store.load_all()
        assert bonuses[0].status == BonusStatus.EXPIRED

    def test_record_bonus(self, tracker, tmp_data_dir):
        b = ReferralBonus(
            program_name="x",
            account_alias="a",
            amount=1.0,
            earned_at=datetime.now(UTC),
        )
        tracker.record_bonus(b)
        store = BonusStore(tmp_data_dir)
        assert len(store.load_all()) == 1
