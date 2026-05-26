"""Vendored egress policy helper for Forge (Stage 3.C sibling adopter).

This is a small duplicate of ``animus.network.egress`` because Forge
cannot import from animus core: Core → Forge already exists (optional),
so Forge → Core would create a circular dependency. Vendoring is the
pragmatic alternative until a shared lower-level package becomes
warranted (which would be the moment tier-aware dispatch lands and the
two copies need to share more than the offline kill-switch).

**Keep this in sync with ``animus.network.egress`` when the policy
changes.** Both copies are ~30 lines of pure logic with no external deps
so drift is detectable by inspection.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


class EgressDeniedError(RuntimeError):
    """Raised when a cloud call is attempted but policy blocks it."""


def _extract_host(destination: str) -> str:
    """Strip scheme, port, and path from a destination string."""
    if destination in _LOOPBACK_HOSTS:
        return destination.lower()
    if "://" in destination:
        parsed = urlparse(destination)
        host = parsed.hostname or ""
    else:
        candidate = destination.split("/")[0]
        if candidate.count(":") <= 1:
            candidate = candidate.split(":")[0]
        host = candidate
    return host.lower()


def is_egress_allowed(destination: str) -> bool:
    """Return True iff outbound traffic to ``destination`` is permitted.

    Rules (subset of the core helper — Forge does not yet have tier
    awareness in its dispatch layer):

    - Loopback (``localhost``, ``127.0.0.1``, ``::1``, ``0.0.0.0``,
      ``*.local``) — always allowed.
    - ``ANIMUS_OFFLINE=1`` env — deny all non-loopback.
    - Otherwise — allow.
    """
    host = _extract_host(destination)
    if host in _LOOPBACK_HOSTS or host.endswith(".local"):
        return True
    if os.environ.get("ANIMUS_OFFLINE") == "1":
        return False
    return True
