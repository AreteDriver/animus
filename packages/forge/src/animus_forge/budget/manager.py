"""Re-export of ``animus_kernel.budget.manager`` for backward compatibility.

ADL-20260806-001 Phase 1+3 — see ``__init__.py`` for the deprecation notice.
"""

import warnings

warnings.warn(
    "animus_forge.budget.manager is deprecated; import from "
    "animus_kernel.budget.manager instead (ADL-20260806-001).",
    DeprecationWarning,
    stacklevel=2,
)

from animus_kernel.budget.manager import *  # noqa: F401, F403
