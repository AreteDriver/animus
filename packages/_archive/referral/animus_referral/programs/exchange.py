"""CEX referral program definitions."""

from __future__ import annotations

from animus_referral.models import BonusType, ProgramType, ReferralProgram
from animus_referral.programs.base import BaseProgram

EXCHANGE_PROGRAMS = [
    ReferralProgram(
        name="coinbase",
        platform_url="https://www.coinbase.com",
        program_type=ProgramType.EXCHANGE,
        bonus_referrer_usd=10.0,
        bonus_referee_usd=10.0,
        bonus_type=BonusType.FLAT,
        requirements="Referee must buy or sell $100+ in crypto within 180 days",
        referral_url_template="https://www.coinbase.com/join/{code}",
        active=True,
    ),
    ReferralProgram(
        name="binance",
        platform_url="https://www.binance.com",
        program_type=ProgramType.EXCHANGE,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="Referee registers and trades. Up to 40% commission kickback",
        referral_url_template="https://accounts.binance.com/register?ref={code}",
        active=True,
    ),
    ReferralProgram(
        name="kraken",
        platform_url="https://www.kraken.com",
        program_type=ProgramType.EXCHANGE,
        bonus_referrer_usd=10.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.FLAT,
        requirements="Referee must trade $100+ within 30 days",
        referral_url_template="https://www.kraken.com/sign-up?referral={code}",
        active=True,
    ),
    ReferralProgram(
        name="bybit",
        platform_url="https://www.bybit.com",
        program_type=ProgramType.EXCHANGE,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="Up to 30% commission. Referee registers and deposits",
        referral_url_template="https://www.bybit.com/invite?ref={code}",
        active=True,
    ),
]


class ExchangePrograms(BaseProgram):
    """CEX referral program registry."""

    def get_programs(self) -> list[ReferralProgram]:
        return list(EXCHANGE_PROGRAMS)

    def generate_link(self, program: ReferralProgram, referral_code: str) -> str:
        if program.referral_url_template:
            return program.referral_url_template.format(code=referral_code)
        return f"{program.platform_url}?ref={referral_code}"
