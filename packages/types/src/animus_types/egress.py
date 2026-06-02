"""Canonical compute-boundary egress policy for the Animus monorepo.

This is the single source of truth for "may outbound traffic to this
destination carry data of this sensitivity tier". It previously existed as
two hand-synced copies (``animus.network.egress`` and
``animus_forge.network.egress``) whose own docstrings admitted "keep this in
sync ... drift is detectable by inspection". Hand-synced security logic is a
latent divergence: this module removes that class of bug by giving both
packages one implementation to import.

``animus-types`` has zero production dependencies, and this module uses only
the standard library, so both Core and Forge can depend on it without a
cross-package cycle.

The function accepts the tier under either keyword for backward compatibility
with both historical call conventions:

- Core called it ``is_egress_allowed(dest, tier)`` (positional or ``tier=``).
- Forge called it ``is_egress_allowed(dest, sensitivity=...)`` (keyword).

Both continue to work; ``sensitivity`` wins if both are somehow supplied.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from animus_types.sensitivity import Sensitivity

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


class EgressDeniedError(RuntimeError):
    """Raised when a cloud call is attempted but egress policy blocks it.

    Defined here so a single class is shared across packages: code in Core
    can ``raise`` it and code in Forge can ``except`` it (and vice versa)
    with the identity check actually holding.
    """


def _extract_host(destination: str) -> str:
    """Strip scheme, port, and path from a destination string.

    Accepts ``api.anthropic.com``, ``https://api.anthropic.com``,
    ``https://api.anthropic.com/v1/messages``, ``localhost:11434``,
    bare ``::1`` IPv6, etc.
    """
    if destination in _LOOPBACK_HOSTS:
        return destination.lower()
    if "://" in destination:
        parsed = urlparse(destination)
        host = parsed.hostname or ""
    else:
        # Strip path first, then port. IPv6 bare addresses contain ``:`` so
        # only strip a trailing port when there's a single colon (i.e. not an
        # IPv6 literal).
        candidate = destination.split("/")[0]
        if candidate.count(":") <= 1:
            candidate = candidate.split(":")[0]
        host = candidate
    return host.lower()


def is_egress_allowed(
    destination: str,
    tier: Sensitivity | None = None,
    *,
    sensitivity: Sensitivity | None = None,
) -> bool:
    """Return True iff outbound traffic to ``destination`` is permitted.

    Rules (most-protective first):

    - Loopback (``localhost``, ``127.0.0.1``, ``::1``, ``0.0.0.0``,
      ``*.local``) is always allowed, so local Ollama / API servers work
      unconditionally.
    - ``ANIMUS_OFFLINE=1`` denies all non-loopback (kill-switch).
    - ``CONFIDENTIAL`` / ``SECRET`` deny non-loopback: these tiers must never
      leave the box, regardless of env flags.
    - ``PERSONAL`` denies non-loopback when ``ANIMUS_LOCAL_ONLY=1``.
    - Otherwise (``PUBLIC``, ``PERSONAL`` without LOCAL_ONLY, or no tier)
      allow.

    Args:
        destination: Hostname or URL. Loopback hosts always return True.
        tier: Sensitivity of the data being sent (positional/keyword, Core
            convention). ``None`` is treated as most-permissive.
        sensitivity: Same meaning under Forge's keyword convention. Takes
            precedence over ``tier`` if both are given.
    """
    effective = sensitivity if sensitivity is not None else tier

    host = _extract_host(destination)
    if host in _LOOPBACK_HOSTS or host.endswith(".local"):
        return True
    if os.environ.get("ANIMUS_OFFLINE") == "1":
        return False
    if effective in (Sensitivity.CONFIDENTIAL, Sensitivity.SECRET):
        return False
    if effective is Sensitivity.PERSONAL and os.environ.get("ANIMUS_LOCAL_ONLY") == "1":
        return False
    return True
