"""Egress policy helper for Forge.

The implementation now lives canonically in ``animus_types.egress`` so Core
and Forge share one source of truth. This module previously held a
hand-synced copy whose own docstring warned "keep this in sync ... drift is
detectable by inspection"; that drift risk is now gone. It re-exports the
canonical helpers to preserve the ``animus_forge.network.egress`` import path
and Forge's ``is_egress_allowed(destination, sensitivity=...)`` call
convention.

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
