"""Disclosure classification for memory and LLM requests.

Originally defined in ``animus.memory.types`` (Stage 2.A). Hoisted into
this shared package as part of Stage 3.D (tier-aware LLM dispatch) so
Forge providers can reference the same enum without a cross-package
import cycle.

The legacy import path ``from animus.memory.types import Sensitivity``
continues to work via a re-export — see ``packages/core/animus/memory/
types.py``.
"""

from __future__ import annotations

from enum import Enum


class Sensitivity(Enum):
    """Disclosure classification — four tiers, least to most sensitive.

    Ordering (for max-tier comparisons): PUBLIC < PERSONAL < CONFIDENTIAL < SECRET.
    """

    PUBLIC = "public"  # Safe to surface anywhere — public refs, podcast facts
    PERSONAL = "personal"  # Default for own-notes — emails, drafts, decisions
    CONFIDENTIAL = "confidential"  # Client / legal / employer
    SECRET = "secret"  # Credentials, financial, must never cross boundaries
