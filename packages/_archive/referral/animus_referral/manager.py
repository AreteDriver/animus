"""Referral manager — cross-refer accounts, track links, optimize chains."""

from __future__ import annotations

import logging
from pathlib import Path

from animus_referral.config import ReferralConfig
from animus_referral.models import (
    Account,
    BonusStatus,
    ProgramType,
    ReferralBonus,
    ReferralLink,
    ReferralProgram,
)
from animus_referral.programs import ALL_PROGRAMS
from animus_referral.store import AccountStore, BonusStore, LinkStore

logger = logging.getLogger("animus_referral.manager")


class ReferralManager:
    """Orchestrates referral program management across accounts.

    Handles cross-referral between accounts, link generation,
    bonus tracking, and optimal chain calculation.
    """

    def __init__(self, config: ReferralConfig, data_dir: Path | None = None):
        self.config = config
        self._data_dir = data_dir or config.data_dir
        self._accounts = AccountStore(self._data_dir)
        self._links = LinkStore(self._data_dir)
        self._bonuses = BonusStore(self._data_dir)
        self._ensure_builtin_programs()

    def _ensure_builtin_programs(self) -> None:
        """Load built-in program definitions into config if not already present."""
        for program in ALL_PROGRAMS:
            self.config.add_program(program)

    # --- Account management ---

    def add_account(self, account: Account) -> None:
        """Register an account."""
        self._accounts.save(account)
        logger.info("Added account %s on %s", account.alias, account.platform)

    def get_accounts(self, platform: str | None = None) -> list[Account]:
        """List accounts, optionally filtered by platform."""
        if platform:
            return self._accounts.get_by_platform(platform)
        return self._accounts.load_all()

    def get_account(self, alias: str, platform: str) -> Account | None:
        """Get a specific account."""
        return self._accounts.get(alias, platform)

    def remove_account(self, alias: str, platform: str) -> bool:
        """Remove an account."""
        return self._accounts.remove(alias, platform)

    # --- Program management ---

    def add_program(self, program: ReferralProgram) -> bool:
        """Add a new referral program to the registry."""
        return self.config.add_program(program)

    def get_programs(
        self,
        program_type: ProgramType | None = None,
        active_only: bool = True,
    ) -> list[ReferralProgram]:
        """List programs, optionally filtered by type."""
        programs = self.config.get_active_programs() if active_only else self.config.programs
        if program_type:
            programs = [p for p in programs if p.program_type == program_type]
        return programs

    def get_high_value_programs(self) -> list[ReferralProgram]:
        """Return programs above the high-value threshold."""
        return self.config.get_high_value_programs()

    # --- Link management ---

    def generate_link(
        self, program_name: str, referrer_alias: str, referral_code: str
    ) -> ReferralLink | None:
        """Generate a referral link for a program and account.

        If a link already exists, returns the existing one.
        """
        program = self.config.get_program(program_name)
        if program is None:
            logger.warning("Program '%s' not found", program_name)
            return None

        existing = self._links.get(program_name, referrer_alias)
        if existing:
            return existing

        if program.referral_url_template:
            url = program.referral_url_template.format(code=referral_code)
        else:
            url = f"{program.platform_url}?ref={referral_code}"

        link = ReferralLink(
            program_name=program_name,
            referrer_account=referrer_alias,
            link_url=url,
        )
        self._links.save(link)
        logger.info("Generated link for %s via %s", program_name, referrer_alias)
        return link

    def get_link(self, program_name: str, referrer_alias: str) -> ReferralLink | None:
        """Get an existing referral link."""
        return self._links.get(program_name, referrer_alias)

    def get_links(self, program_name: str | None = None) -> list[ReferralLink]:
        """List all links, optionally filtered by program."""
        if program_name:
            return self._links.get_for_program(program_name)
        return self._links.load_all()

    # --- Bonus tracking ---

    def record_bonus(self, bonus: ReferralBonus) -> None:
        """Record a referral bonus."""
        self._bonuses.record(bonus)
        logger.info(
            "Recorded bonus: %s %.2f %s for %s",
            bonus.program_name,
            bonus.amount,
            bonus.token,
            bonus.account_alias,
        )

    def get_bonuses(
        self,
        program_name: str | None = None,
        account_alias: str | None = None,
    ) -> list[ReferralBonus]:
        """List bonuses with optional filters."""
        if program_name:
            return self._bonuses.get_for_program(program_name)
        if account_alias:
            return self._bonuses.get_for_account(account_alias)
        return self._bonuses.load_all()

    # --- Referral graph ---

    def get_referral_graph(self) -> dict[str, list[str]]:
        """Build a graph of who referred whom.

        Returns a dict mapping referrer alias -> list of referee aliases.
        """
        accounts = self._accounts.load_all()
        graph: dict[str, list[str]] = {}
        for account in accounts:
            if account.referred_by:
                referrer = account.referred_by
                if referrer not in graph:
                    graph[referrer] = []
                graph[referrer].append(account.alias)
        return graph

    def get_referral_chain(self, alias: str) -> list[str]:
        """Trace the referral chain upward from an account.

        Returns [alias, referred_by, referred_by's referrer, ...] until root.
        """
        accounts_by_alias: dict[str, Account] = {}
        for acct in self._accounts.load_all():
            accounts_by_alias[acct.alias] = acct

        chain = [alias]
        seen = {alias}
        current = alias
        while current in accounts_by_alias and accounts_by_alias[current].referred_by:
            referrer = accounts_by_alias[current].referred_by
            if referrer in seen:
                break  # cycle protection
            chain.append(referrer)
            seen.add(referrer)
            current = referrer
        return chain

    # --- Cross-referral optimization ---

    def suggest_cross_referrals(self, platform: str) -> list[tuple[str, str]]:
        """Suggest which accounts should refer which others on a platform.

        Returns list of (referrer_alias, referee_alias) pairs where the
        referee has no referrer yet.
        """
        accounts = self._accounts.get_by_platform(platform)
        unreferred = [a for a in accounts if not a.referred_by]
        referrers = [a for a in accounts if a.referral_code]

        suggestions = []
        for referee in unreferred:
            for referrer in referrers:
                if referrer.alias != referee.alias:
                    suggestions.append((referrer.alias, referee.alias))
                    break  # one suggestion per unreferred account
        return suggestions

    def calculate_optimal_chain(self, aliases: list[str], platform: str) -> list[str]:
        """Calculate optimal referral chain ordering for max bonuses.

        Given N accounts, A refers B, B refers C, etc. The first account
        has no referrer. Returns ordered list of aliases.
        Heuristic: put accounts with existing referral codes first.
        """
        if len(aliases) <= 1:
            return list(aliases)

        accounts = {
            a.alias: a
            for a in self._accounts.load_all()
            if a.alias in aliases and a.platform == platform
        }

        with_codes = [a for a in aliases if a in accounts and accounts[a].referral_code]
        without_codes = [a for a in aliases if a not in with_codes]

        return with_codes + without_codes

    # --- Summary ---

    def get_summary(self) -> dict:
        """Get an overall referral program summary."""
        accounts = self._accounts.load_all()
        links = self._links.load_all()
        bonuses = self._bonuses.load_all()

        paid_bonuses = [b for b in bonuses if b.status == BonusStatus.PAID]
        pending_bonuses = [b for b in bonuses if b.status == BonusStatus.PENDING]

        total_earned = sum(b.amount_usd for b in paid_bonuses)
        total_pending = sum(b.amount_usd for b in pending_bonuses)
        total_signups = sum(link.signups for link in links)

        return {
            "total_accounts": len(accounts),
            "total_programs": len(self.config.programs),
            "active_programs": len(self.config.get_active_programs()),
            "total_links": len(links),
            "total_signups": total_signups,
            "total_bonuses": len(bonuses),
            "total_earned_usd": total_earned,
            "total_pending_usd": total_pending,
        }
