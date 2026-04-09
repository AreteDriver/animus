"""DeFi protocol referral program definitions."""

from __future__ import annotations

from animus_referral.models import BonusType, ProgramType, ReferralProgram
from animus_referral.programs.base import BaseProgram

DEFI_PROGRAMS = [
    ReferralProgram(
        name="uniswap",
        platform_url="https://app.uniswap.org",
        program_type=ProgramType.DEFI,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="Swap fee sharing via referral link",
        referral_url_template="https://app.uniswap.org/swap?referral={code}",
        active=True,
        chain="ethereum",
    ),
    ReferralProgram(
        name="aave",
        platform_url="https://app.aave.com",
        program_type=ProgramType.DEFI,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="Earn percentage of referred user lending fees",
        referral_url_template="https://app.aave.com/?referral={code}",
        active=True,
        chain="ethereum",
    ),
    ReferralProgram(
        name="1inch",
        platform_url="https://app.1inch.io",
        program_type=ProgramType.DEFI,
        bonus_referrer_usd=0.0,
        bonus_referee_usd=0.0,
        bonus_type=BonusType.PERCENTAGE,
        requirements="Swap fee sharing program",
        referral_url_template="https://app.1inch.io/#/referral?code={code}",
        active=True,
        chain="ethereum",
    ),
]


class DefiPrograms(BaseProgram):
    """DeFi protocol referral program registry."""

    def get_programs(self) -> list[ReferralProgram]:
        return list(DEFI_PROGRAMS)

    def generate_link(self, program: ReferralProgram, referral_code: str) -> str:
        if program.referral_url_template:
            return program.referral_url_template.format(code=referral_code)
        return f"{program.platform_url}?ref={referral_code}"
