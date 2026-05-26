"""Compute-boundary egress policy (Stage 3.C).

``is_egress_allowed(destination_host, tier)`` enforces:

- ``ANIMUS_OFFLINE=1`` env → deny all non-loopback (kill-switch)
- ``Sensitivity.CONFIDENTIAL`` / ``Sensitivity.SECRET`` → deny non-loopback;
  these tiers must never leave the box.
- ``Sensitivity.PERSONAL`` → deny when ``ANIMUS_LOCAL_ONLY=1``, else allow
  (matches sovereignty-wizard intent without requiring its config plumbing).
- ``Sensitivity.PUBLIC`` → always allow.

Loopback destinations (``localhost``, ``127.0.0.1``, ``::1``) are always
allowed regardless of tier or env flags. Ollama / local API servers
therefore work unconditionally.

Callers that need to *enforce* (not just check) should catch
``EgressDeniedError`` from a helper wrapping their outbound HTTP, or call
``is_egress_allowed`` and raise themselves.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from animus.memory.types import Sensitivity

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


class EgressDeniedError(RuntimeError):
    """Raised when a cloud call is attempted for a tier blocked by policy."""


def _extract_host(destination: str) -> str:
    """Strip scheme, port, and path from a destination string.

    Accepts ``api.anthropic.com``, ``https://api.anthropic.com``,
    ``https://api.anthropic.com/v1/messages``, ``localhost:11434``,
    bare ``::1`` IPv6, etc.
    """
    # Exact loopback shorthand (no scheme, no port, no path)
    if destination in _LOOPBACK_HOSTS:
        return destination.lower()
    if "://" in destination:
        parsed = urlparse(destination)
        host = parsed.hostname or ""
    else:
        # Strip path first, then port. IPv6 bare addresses contain `:` so
        # only strip a trailing port if there's a digit-only segment after
        # the last colon. Conservative approach: only strip ":port" if the
        # destination starts with a non-IPv6 character.
        candidate = destination.split("/")[0]
        if candidate.count(":") <= 1:
            candidate = candidate.split(":")[0]
        host = candidate
    return host.lower()


def is_egress_allowed(destination: str, tier: Sensitivity | None = None) -> bool:
    """Return True if outbound traffic to ``destination`` is permitted for ``tier``.

    Args:
        destination: Hostname or URL. Loopback hosts always return True.
        tier: Sensitivity of the data being sent. ``None`` is treated as
            ``Sensitivity.PUBLIC`` (most permissive) for backward compat —
            but callers should pass an explicit tier.
    """
    from animus.memory.types import Sensitivity  # local import — avoids cycle

    host = _extract_host(destination)
    if host in _LOOPBACK_HOSTS or host.endswith(".local"):
        return True

    if os.environ.get("ANIMUS_OFFLINE") == "1":
        return False

    if tier in (Sensitivity.CONFIDENTIAL, Sensitivity.SECRET):
        return False

    if tier is Sensitivity.PERSONAL and os.environ.get("ANIMUS_LOCAL_ONLY") == "1":
        return False

    # tier is PUBLIC, None, or PERSONAL-with-no-restriction → allow
    return True
