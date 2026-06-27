"""Animus Faucet — Autonomous crypto faucet farming module."""

__version__ = "0.1.0"

from animus_faucet.config import FaucetConfig
from animus_faucet.models import (
    CaptchaType,
    Chain,
    ClaimResult,
    ClaimStatus,
    FaucetDefinition,
    FaucetHealth,
    TreasuryBalance,
)
from animus_faucet.scheduler import FaucetScheduler
from animus_faucet.treasury import Treasury

__all__ = [
    "FaucetConfig",
    "FaucetScheduler",
    "Treasury",
    "Chain",
    "CaptchaType",
    "ClaimResult",
    "ClaimStatus",
    "FaucetDefinition",
    "FaucetHealth",
    "TreasuryBalance",
    "__version__",
]
