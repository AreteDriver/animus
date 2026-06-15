"""Egress policy enforcement for Forge — vendored from animus core.

See ``egress.py`` for the rationale on why this is a duplicate of the
``animus.network`` helper instead of an import.
"""

from animus_kernel.network.egress import EgressDeniedError, is_egress_allowed

__all__ = ["EgressDeniedError", "is_egress_allowed"]
