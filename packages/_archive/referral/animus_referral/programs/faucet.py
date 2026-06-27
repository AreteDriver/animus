"""Faucet referral program definitions."""

from __future__ import annotations

from animus_referral.models import BonusType, ProgramType, ReferralProgram
from animus_referral.programs.base import BaseProgram

FAUCET_PROGRAMS = [
    ReferralProgram(
        name="freebitcoin",
        platform_url="https://freebitco.in",
        program_type=ProgramType.FAUCET,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="50% of referee base prize on every roll",
        referral_url_template="https://freebitco.in/?r={code}",
        active=True,
        chain="btc",
    ),
    ReferralProgram(
        name="cointiply",
        platform_url="https://cointiply.com",
        program_type=ProgramType.FAUCET,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="25% of referee earnings",
        referral_url_template="https://cointiply.com/members/join/{code}",
        active=True,
        chain="btc",
    ),
    ReferralProgram(
        name="firefaucet",
        platform_url="https://firefaucet.win",
        program_type=ProgramType.FAUCET,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="20% of referee auto-faucet claims",
        referral_url_template="https://firefaucet.win/ref/{code}",
        active=True,
        chain="btc",
    ),
]


class FaucetPrograms(BaseProgram):
    """Faucet referral program registry."""

    def get_programs(self) -> list[ReferralProgram]:
        return list(FAUCET_PROGRAMS)

    def generate_link(self, program: ReferralProgram, referral_code: str) -> str:
        if program.referral_url_template:
            return program.referral_url_template.format(code=referral_code)
        return f"{program.platform_url}?ref={referral_code}"
