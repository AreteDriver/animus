"""Compute-boundary egress policy (Stage 3.C).

The implementation now lives canonically in ``animus_types.egress`` so Core
and Forge share one source of truth instead of two hand-synced copies. This
module re-exports it to preserve the ``animus.network.egress`` import path and
the Core ``is_egress_allowed(destination, tier)`` call convention.

See ``animus_types.egress`` for the policy rules.
"""

from __future__ import annotations

from animus_types.egress import (
    _LOOPBACK_HOSTS,
    EgressDeniedError,
    _extract_host,
    is_egress_allowed,
)

__all__ = [
    "EgressDeniedError",
    "is_egress_allowed",
    "_extract_host",
    "_LOOPBACK_HOSTS",
]
