"""Tests for program registries."""

from __future__ import annotations

from animus_referral.models import ProgramType, ReferralProgram
from animus_referral.programs import ALL_PROGRAMS, DEFI_PROGRAMS, EXCHANGE_PROGRAMS, FAUCET_PROGRAMS
from animus_referral.programs.base import BaseProgram
from animus_referral.programs.defi import DefiPrograms
from animus_referral.programs.exchange import ExchangePrograms
from animus_referral.programs.faucet import FaucetPrograms


class TestAllPrograms:
    def test_all_programs_not_empty(self):
        assert len(ALL_PROGRAMS) > 0

    def test_all_programs_is_sum(self):
        assert len(ALL_PROGRAMS) == len(EXCHANGE_PROGRAMS) + len(DEFI_PROGRAMS) + len(
            FAUCET_PROGRAMS
        )

    def test_all_programs_have_names(self):
        for p in ALL_PROGRAMS:
            assert p.name
            assert p.platform_url

    def test_unique_names(self):
        names = [p.name for p in ALL_PROGRAMS]
        assert len(names) == len(set(names))


class TestExchangePrograms:
    def test_registry(self):
        ep = ExchangePrograms()
        programs = ep.get_programs()
        assert len(programs) > 0
        for p in programs:
            assert p.program_type == ProgramType.EXCHANGE

    def test_generate_link_with_template(self):
        ep = ExchangePrograms()
        coinbase = ep.get_program_by_name("coinbase")
        assert coinbase is not None
        link = ep.generate_link(coinbase, "MY_CODE")
        assert "MY_CODE" in link

    def test_generate_link_fallback(self):
        ep = ExchangePrograms()
        p = ReferralProgram(
            name="no_template",
            platform_url="https://example.com",
            program_type=ProgramType.EXCHANGE,
            referral_url_template="",
        )
        link = ep.generate_link(p, "CODE")
        assert link == "https://example.com?ref=CODE"

    def test_get_program_by_name(self):
        ep = ExchangePrograms()
        assert ep.get_program_by_name("coinbase") is not None
        assert ep.get_program_by_name("nonexistent") is None

    def test_known_exchanges(self):
        names = [p.name for p in EXCHANGE_PROGRAMS]
        assert "coinbase" in names
        assert "binance" in names
        assert "kraken" in names
        assert "bybit" in names


class TestDefiPrograms:
    def test_registry(self):
        dp = DefiPrograms()
        programs = dp.get_programs()
        assert len(programs) > 0
        for p in programs:
            assert p.program_type == ProgramType.DEFI

    def test_generate_link(self):
        dp = DefiPrograms()
        programs = dp.get_programs()
        link = dp.generate_link(programs[0], "MY_CODE")
        assert "MY_CODE" in link

    def test_generate_link_fallback(self):
        dp = DefiPrograms()
        p = ReferralProgram(
            name="no_template",
            platform_url="https://defi.com",
            program_type=ProgramType.DEFI,
            referral_url_template="",
        )
        link = dp.generate_link(p, "CODE")
        assert link == "https://defi.com?ref=CODE"

    def test_known_defi(self):
        names = [p.name for p in DEFI_PROGRAMS]
        assert "uniswap" in names
        assert "aave" in names

    def test_get_program_by_name(self):
        dp = DefiPrograms()
        assert dp.get_program_by_name("uniswap") is not None
        assert dp.get_program_by_name("nonexistent") is None


class TestFaucetPrograms:
    def test_registry(self):
        fp = FaucetPrograms()
        programs = fp.get_programs()
        assert len(programs) > 0
        for p in programs:
            assert p.program_type == ProgramType.FAUCET

    def test_generate_link(self):
        fp = FaucetPrograms()
        programs = fp.get_programs()
        link = fp.generate_link(programs[0], "MY_CODE")
        assert "MY_CODE" in link

    def test_generate_link_fallback(self):
        fp = FaucetPrograms()
        p = ReferralProgram(
            name="no_template",
            platform_url="https://faucet.com",
            program_type=ProgramType.FAUCET,
            referral_url_template="",
        )
        link = fp.generate_link(p, "CODE")
        assert link == "https://faucet.com?ref=CODE"

    def test_known_faucets(self):
        names = [p.name for p in FAUCET_PROGRAMS]
        assert "freebitcoin" in names
        assert "cointiply" in names

    def test_get_program_by_name(self):
        fp = FaucetPrograms()
        assert fp.get_program_by_name("freebitcoin") is not None
        assert fp.get_program_by_name("nonexistent") is None


class TestBaseProgram:
    def test_abstract(self):
        """BaseProgram cannot be instantiated."""
        try:
            BaseProgram()
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_get_program_by_name_inherited(self):
        ep = ExchangePrograms()
        result = ep.get_program_by_name("coinbase")
        assert result is not None
        assert result.name == "coinbase"
