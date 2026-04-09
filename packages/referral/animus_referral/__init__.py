"""Animus Referral — Autonomous referral program stacking module."""

__version__ = "0.1.0"

from animus_referral.config import ReferralConfig
from animus_referral.manager import ReferralManager
from animus_referral.models import (
    Account,
    BonusStatus,
    BonusType,
    ProgramType,
    ReferralBonus,
    ReferralLink,
    ReferralProgram,
)
from animus_referral.tracker import ReferralTracker

__all__ = [
    "Account",
    "BonusStatus",
    "BonusType",
    "ProgramType",
    "ReferralBonus",
    "ReferralConfig",
    "ReferralLink",
    "ReferralManager",
    "ReferralProgram",
    "ReferralTracker",
    "__version__",
]
