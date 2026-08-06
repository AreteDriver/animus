"""Re-export of ``animus_kernel.budget.manager`` for backward compatibility.

ADL-20260806-001 Phase 1 — the kernel package is the canonical home for
budget primitives; this module exists so ``from animus_forge.budget.manager
import BudgetManager`` keeps working unchanged. Remove in Phase 4.
"""

from animus_kernel.budget.manager import *  # noqa: F401, F403
