"""Re-export of ``animus_kernel.budget.preflight`` for backward compatibility.

ADL-20260806-001 Phase 1+3 — see ``__init__.py`` for the deprecation notice.
"""

import warnings

warnings.warn(
    "animus_forge.budget.preflight is deprecated; import from "
    "animus_kernel.budget.preflight instead (ADL-20260806-001).",
    DeprecationWarning,
    stacklevel=2,
)

from animus_kernel.budget.preflight import *  # noqa: F401, F403
