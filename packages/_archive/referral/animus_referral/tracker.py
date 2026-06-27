"""Referral tracker — aggregates and reports on bonus earnings."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from animus_referral.models import BonusStatus, ReferralBonus
from animus_referral.store import BonusStore, LinkStore

logger = logging.getLogger("animus_referral.tracker")


class ReferralTracker:
    """Tracks and reports on referral program earnings.

    Provides aggregation by program, account, time period, and status.
    """

    def __init__(self, data_dir: Path):
        self._bonuses = BonusStore(data_dir)
        self._links = LinkStore(data_dir)

    def total_earned_usd(self, days: int | None = None) -> float:
        """Total USD earned from paid bonuses, optionally within N days."""
        bonuses = self._bonuses.load_all()
        paid = [b for b in bonuses if b.status == BonusStatus.PAID]
        if days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            paid = [b for b in paid if b.earned_at >= cutoff]
        return sum(b.amount_usd for b in paid)

    def total_pending_usd(self) -> float:
        """Total USD pending from unconfirmed bonuses."""
        bonuses = self._bonuses.load_all()
        return sum(b.amount_usd for b in bonuses if b.status == BonusStatus.PENDING)

    def earnings_by_program(self, days: int | None = None) -> dict[str, float]:
        """Breakdown of earnings by program name."""
        bonuses = self._bonuses.load_all()
        paid = [b for b in bonuses if b.status == BonusStatus.PAID]
        if days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            paid = [b for b in paid if b.earned_at >= cutoff]

        by_program: dict[str, float] = {}
        for b in paid:
            by_program[b.program_name] = by_program.get(b.program_name, 0) + b.amount_usd
        return by_program

    def earnings_by_account(self, days: int | None = None) -> dict[str, float]:
        """Breakdown of earnings by account alias."""
        bonuses = self._bonuses.load_all()
        paid = [b for b in bonuses if b.status == BonusStatus.PAID]
        if days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            paid = [b for b in paid if b.earned_at >= cutoff]

        by_account: dict[str, float] = {}
        for b in paid:
            by_account[b.account_alias] = by_account.get(b.account_alias, 0) + b.amount_usd
        return by_account

    def conversion_rate(self) -> float:
        """Overall conversion rate: signups / clicks across all links."""
        links = self._links.load_all()
        total_clicks = sum(link.clicks for link in links)
        total_signups = sum(link.signups for link in links)
        if total_clicks == 0:
            return 0.0
        return total_signups / total_clicks

    def top_programs(self, n: int = 5) -> list[tuple[str, float]]:
        """Top N programs by total earnings."""
        by_program = self.earnings_by_program()
        sorted_programs = sorted(by_program.items(), key=lambda x: x[1], reverse=True)
        return sorted_programs[:n]

    def top_accounts(self, n: int = 5) -> list[tuple[str, float]]:
        """Top N accounts by total earnings."""
        by_account = self.earnings_by_account()
        sorted_accounts = sorted(by_account.items(), key=lambda x: x[1], reverse=True)
        return sorted_accounts[:n]

    def get_report(self, days: int = 30) -> dict:
        """Generate a full earnings report."""
        return {
            "period_days": days,
            "total_earned_usd": self.total_earned_usd(days=days),
            "total_pending_usd": self.total_pending_usd(),
            "all_time_earned_usd": self.total_earned_usd(),
            "by_program": self.earnings_by_program(days=days),
            "by_account": self.earnings_by_account(days=days),
            "conversion_rate": self.conversion_rate(),
            "top_programs": self.top_programs(),
            "top_accounts": self.top_accounts(),
        }

    def mark_bonus_paid(self, program_name: str, account_alias: str, earned_at: datetime) -> bool:
        """Mark a pending bonus as paid."""
        return self._bonuses.update_status(
            program_name, account_alias, earned_at.isoformat(), BonusStatus.PAID.value
        )

    def mark_bonus_expired(
        self, program_name: str, account_alias: str, earned_at: datetime
    ) -> bool:
        """Mark a pending bonus as expired."""
        return self._bonuses.update_status(
            program_name, account_alias, earned_at.isoformat(), BonusStatus.EXPIRED.value
        )

    def record_bonus(self, bonus: ReferralBonus) -> None:
        """Record a new bonus."""
        self._bonuses.record(bonus)
