"""Re-export of ``animus_kernel.budget.cost_audit`` for backward compatibility.

ADL-20260806-001 Phase 1+3 — see ``__init__.py`` for the deprecation notice.
"""

import warnings

warnings.warn(
    "animus_forge.budget.cost_audit is deprecated; import from "
    "animus_kernel.budget.cost_audit instead (ADL-20260806-001).",
    DeprecationWarning,
    stacklevel=2,
)

from animus_kernel.budget.cost_audit import *  # noqa: F401, F403
