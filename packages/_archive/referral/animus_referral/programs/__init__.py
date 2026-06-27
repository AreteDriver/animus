"""Referral program definitions."""

from animus_referral.programs.base import BaseProgram
from animus_referral.programs.defi import DEFI_PROGRAMS
from animus_referral.programs.exchange import EXCHANGE_PROGRAMS
from animus_referral.programs.faucet import FAUCET_PROGRAMS

ALL_PROGRAMS = EXCHANGE_PROGRAMS + DEFI_PROGRAMS + FAUCET_PROGRAMS

__all__ = [
    "ALL_PROGRAMS",
    "BaseProgram",
    "DEFI_PROGRAMS",
    "EXCHANGE_PROGRAMS",
    "FAUCET_PROGRAMS",
]
