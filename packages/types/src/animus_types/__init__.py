"""animus-types — shared type definitions for the Animus monorepo.

Zero production dependencies. Pure library. See README for rationale.
"""

from animus_types.egress import EgressDeniedError, is_egress_allowed
from animus_types.sensitivity import Sensitivity

__all__ = ["EgressDeniedError", "Sensitivity", "is_egress_allowed"]
__version__ = "0.1.0"
