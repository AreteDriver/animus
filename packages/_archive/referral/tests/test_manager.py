"""Tests for the referral manager."""

from __future__ import annotations

from datetime import UTC, datetime

from animus_referral.models import (
    Account,
    ProgramType,
    ReferralBonus,
    ReferralProgram,
)


class TestManagerAccounts:
    def test_add_account(self, manager, sample_account):
        manager.add_account(sample_account)
        accounts = manager.get_accounts()
        assert len(accounts) == 1
        assert accounts[0].alias == "alice"

    def test_get_accounts_by_platform(self, manager, sample_account, sample_account_bob):
        manager.add_account(sample_account)
        manager.add_account(sample_account_bob)
        results = manager.get_accounts(platform="test_exchange")
        assert len(results) == 2

    def test_get_accounts_all(self, manager, sample_account):
        manager.add_account(sample_account)
        other = Account(alias="carol", platform="other_platform", created_at=datetime.now(UTC))
        manager.add_account(other)
        all_accts = manager.get_accounts()
        assert len(all_accts) == 2

    def test_get_account(self, manager, sample_account):
        manager.add_account(sample_account)
        found = manager.get_account("alice", "test_exchange")
        assert found is not None
        assert found.alias == "alice"

    def test_get_account_not_found(self, manager):
        assert manager.get_account("nobody", "nowhere") is None

    def test_remove_account(self, manager, sample_account):
        manager.add_account(sample_account)
        assert manager.remove_account("alice", "test_exchange") is True
        assert len(manager.get_accounts()) == 0

    def test_remove_account_not_found(self, manager):
        assert manager.remove_account("nobody", "nowhere") is False


class TestManagerPrograms:
    def test_builtin_programs_loaded(self, manager):
        programs = manager.get_programs(active_only=False)
        assert len(programs) > 0

    def test_add_program(self, manager):
        p = ReferralProgram(
            name="custom_program",
            platform_url="https://custom.com",
            program_type=ProgramType.OTHER,
        )
        assert manager.add_program(p) is True
        found = manager.config.get_program("custom_program")
        assert found is not None

    def test_add_program_duplicate(self, manager, sample_program):
        # sample_program may already be added by builtins; try custom
        p = ReferralProgram(
            name="unique_custom",
            platform_url="https://unique.com",
            program_type=ProgramType.OTHER,
        )
        manager.add_program(p)
        assert manager.add_program(p) is False

    def test_get_programs_by_type(self, manager):
        exchange_programs = manager.get_programs(program_type=ProgramType.EXCHANGE)
        for p in exchange_programs:
            assert p.program_type == ProgramType.EXCHANGE

    def test_get_programs_active_only(self, manager):
        inactive = ReferralProgram(
            name="inactive_prog",
            platform_url="https://x.com",
            program_type=ProgramType.OTHER,
            active=False,
        )
        manager.add_program(inactive)
        active = manager.get_programs(active_only=True)
        assert all(p.active for p in active)

    def test_get_programs_include_inactive(self, manager):
        inactive = ReferralProgram(
            name="inactive_prog2",
            platform_url="https://x.com",
            program_type=ProgramType.OTHER,
            active=False,
        )
        manager.add_program(inactive)
        all_progs = manager.get_programs(active_only=False)
        assert any(not p.active for p in all_progs)

    def test_get_high_value_programs(self, manager):
        manager.config.high_value_threshold_usd = 5.0
        high_value = manager.get_high_value_programs()
        for p in high_value:
            assert p.bonus_referrer_usd >= 5.0


class TestManagerLinks:
    def test_generate_link(self, manager):
        link = manager.generate_link("coinbase", "alice", "ABC123")
        assert link is not None
        assert "ABC123" in link.link_url
        assert link.program_name == "coinbase"
        assert link.referrer_account == "alice"

    def test_generate_link_existing(self, manager):
        link1 = manager.generate_link("coinbase", "alice", "ABC123")
        link2 = manager.generate_link("coinbase", "alice", "DIFFERENT")
        assert link1.link_url == link2.link_url  # returns existing

    def test_generate_link_unknown_program(self, manager):
        link = manager.generate_link("nonexistent_program", "alice", "CODE")
        assert link is None

    def test_generate_link_no_template(self, manager):
        p = ReferralProgram(
            name="no_template",
            platform_url="https://example.com",
            program_type=ProgramType.OTHER,
            referral_url_template="",
        )
        manager.add_program(p)
        link = manager.generate_link("no_template", "alice", "CODE")
        assert link is not None
        assert link.link_url == "https://example.com?ref=CODE"

    def test_get_link(self, manager):
        manager.generate_link("coinbase", "alice", "ABC123")
        found = manager.get_link("coinbase", "alice")
        assert found is not None

    def test_get_link_not_found(self, manager):
        assert manager.get_link("x", "y") is None

    def test_get_links_all(self, manager):
        manager.generate_link("coinbase", "alice", "ABC")
        manager.generate_link("coinbase", "bob", "DEF")
        links = manager.get_links()
        assert len(links) == 2

    def test_get_links_by_program(self, manager):
        manager.generate_link("coinbase", "alice", "ABC")
        manager.generate_link("coinbase", "bob", "DEF")
        links = manager.get_links(program_name="coinbase")
        assert len(links) == 2


class TestManagerBonuses:
    def test_record_bonus(self, manager, sample_bonus):
        manager.record_bonus(sample_bonus)
        bonuses = manager.get_bonuses()
        assert len(bonuses) == 1

    def test_get_bonuses_by_program(self, manager, sample_bonus):
        manager.record_bonus(sample_bonus)
        other = ReferralBonus(
            program_name="other", account_alias="alice", amount=1.0, earned_at=datetime.now(UTC)
        )
        manager.record_bonus(other)
        results = manager.get_bonuses(program_name="test_exchange")
        assert len(results) == 1

    def test_get_bonuses_by_account(self, manager, sample_bonus, sample_bonus_pending):
        manager.record_bonus(sample_bonus)
        manager.record_bonus(sample_bonus_pending)
        results = manager.get_bonuses(account_alias="alice")
        assert len(results) == 1


class TestManagerReferralGraph:
    def test_referral_graph(self, manager, sample_account, sample_account_bob):
        manager.add_account(sample_account)
        manager.add_account(sample_account_bob)
        graph = manager.get_referral_graph()
        assert "alice" in graph
        assert "bob" in graph["alice"]

    def test_referral_graph_empty(self, manager):
        assert manager.get_referral_graph() == {}

    def test_referral_chain(self, manager):
        a = Account(alias="a", platform="p", referral_code="A", created_at=datetime.now(UTC))
        b = Account(
            alias="b",
            platform="p",
            referral_code="B",
            referred_by="a",
            created_at=datetime.now(UTC),
        )
        c = Account(alias="c", platform="p", referred_by="b", created_at=datetime.now(UTC))
        manager.add_account(a)
        manager.add_account(b)
        manager.add_account(c)
        chain = manager.get_referral_chain("c")
        assert chain == ["c", "b", "a"]

    def test_referral_chain_single(self, manager, sample_account):
        manager.add_account(sample_account)
        chain = manager.get_referral_chain("alice")
        assert chain == ["alice"]

    def test_referral_chain_cycle_protection(self, manager):
        a = Account(alias="a", platform="p", referred_by="b", created_at=datetime.now(UTC))
        b = Account(alias="b", platform="p", referred_by="a", created_at=datetime.now(UTC))
        manager.add_account(a)
        manager.add_account(b)
        chain = manager.get_referral_chain("a")
        # Should not infinite loop
        assert len(chain) == 2


class TestManagerCrossReferral:
    def test_suggest_cross_referrals(self, manager):
        a = Account(alias="a", platform="p", referral_code="A_CODE", created_at=datetime.now(UTC))
        b = Account(alias="b", platform="p", created_at=datetime.now(UTC))
        manager.add_account(a)
        manager.add_account(b)
        suggestions = manager.suggest_cross_referrals("p")
        assert len(suggestions) == 1
        assert suggestions[0] == ("a", "b")

    def test_suggest_cross_referrals_none_needed(self, manager, sample_account, sample_account_bob):
        manager.add_account(sample_account)
        manager.add_account(sample_account_bob)
        # bob already referred by alice, alice has no referrer but alice has a code
        suggestions = manager.suggest_cross_referrals("test_exchange")
        # alice is unreferred, bob has code => bob can refer alice
        unreferred = [s for s in suggestions if s[1] == "alice"]
        assert len(unreferred) <= 1

    def test_calculate_optimal_chain(self, manager):
        a = Account(alias="a", platform="p", referral_code="A_CODE", created_at=datetime.now(UTC))
        b = Account(alias="b", platform="p", created_at=datetime.now(UTC))
        c = Account(alias="c", platform="p", referral_code="C_CODE", created_at=datetime.now(UTC))
        manager.add_account(a)
        manager.add_account(b)
        manager.add_account(c)
        chain = manager.calculate_optimal_chain(["a", "b", "c"], "p")
        # Accounts with codes first
        assert chain[0] in ("a", "c")
        assert chain[1] in ("a", "c")
        assert chain[2] == "b"

    def test_calculate_optimal_chain_empty(self, manager):
        assert manager.calculate_optimal_chain([], "p") == []

    def test_calculate_optimal_chain_single(self, manager, sample_account):
        manager.add_account(sample_account)
        chain = manager.calculate_optimal_chain(["alice"], "test_exchange")
        assert chain == ["alice"]


class TestManagerSummary:
    def test_get_summary_empty(self, manager):
        summary = manager.get_summary()
        assert summary["total_accounts"] == 0
        assert summary["total_links"] == 0
        assert summary["total_bonuses"] == 0
        assert summary["total_earned_usd"] == 0
        assert summary["total_pending_usd"] == 0

    def test_get_summary_with_data(
        self, manager, sample_account, sample_bonus, sample_bonus_pending
    ):
        manager.add_account(sample_account)
        manager.generate_link("coinbase", "alice", "ABC")
        manager.record_bonus(sample_bonus)
        manager.record_bonus(sample_bonus_pending)

        summary = manager.get_summary()
        assert summary["total_accounts"] == 1
        assert summary["total_links"] == 1
        assert summary["total_bonuses"] == 2
        assert summary["total_earned_usd"] == 10.0
        assert summary["total_pending_usd"] == 5.0
