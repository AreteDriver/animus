"""Network-boundary policy helpers (Stage 3.C)."""

from animus.network.egress import EgressDeniedError, is_egress_allowed

__all__ = ["EgressDeniedError", "is_egress_allowed"]
