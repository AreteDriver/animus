"""Animus Bounty — Autonomous bounty and task claiming module."""

__version__ = "0.1.0"

from animus_bounty.config import BountyConfig
from animus_bounty.matcher import SkillMatcher
from animus_bounty.models import (
    Bounty,
    BountyStatus,
    ClaimResult,
    Platform,
    SkillMatch,
)
from animus_bounty.store import BountyStore
from animus_bounty.tracker import BountyTracker

__all__ = [
    "Bounty",
    "BountyConfig",
    "BountyStatus",
    "BountyStore",
    "BountyTracker",
    "ClaimResult",
    "Platform",
    "SkillMatch",
    "SkillMatcher",
    "__version__",
]
