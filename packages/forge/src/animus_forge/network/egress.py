"""Egress policy helper for Forge.

Originally a vendored copy of ``animus.network.egress`` to bridge the
Core→Forge cross-package barrier. As of Stage 3.D this module reads the
shared ``Sensitivity`` enum from ``animus-types``, so the type contract
is unified across packages even though the behavior implementations
remain separate (Core's copy has additional rules; this one mirrors the
ones Forge needs at the cloud-LLM call sites).

**Keep this in sync with ``animus.network.egress`` when the policy
changes.** Both copies are ~50 lines of pure logic with no heavy deps so
drift is detectable by inspection.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from animus_types import Sensitivity

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


def is_egress_allowed(
    destination: str,
    sensitivity: Sensitivity | None = None,
) -> bool:
    """Return True iff outbound traffic to ``destination`` is permitted.

    Rules (most-protective first):

    - Loopback (``localhost``, ``127.0.0.1``, ``::1``, ``0.0.0.0``,
      ``*.local``) — always allowed.
    - ``ANIMUS_OFFLINE=1`` env — deny all non-loopback.
    - ``sensitivity`` in {CONFIDENTIAL, SECRET} — deny non-loopback. These
      tiers must never leave the box, regardless of env flags.
    - ``sensitivity == PERSONAL`` + ``ANIMUS_LOCAL_ONLY=1`` — deny non-loopback.
    - Otherwise (PUBLIC, PERSONAL without LOCAL_ONLY, or no tier) — allow.

    The ``sensitivity`` parameter is optional for backward compat with
    pre-Stage-3.D callers that don't yet thread the request's tier. The
    cloud-LLM call-site adopters (5 sites in PR #57) will pass it via the
    request from PR #59 (Stage 3.D) onward.
    """
    host = _extract_host(destination)
    if host in _LOOPBACK_HOSTS or host.endswith(".local"):
        return True
    if os.environ.get("ANIMUS_OFFLINE") == "1":
        return False
    if sensitivity in (Sensitivity.CONFIDENTIAL, Sensitivity.SECRET):
        return False
    if sensitivity is Sensitivity.PERSONAL and os.environ.get("ANIMUS_LOCAL_ONLY") == "1":
        return False
    return True
