"""Cost and Token Budget Management.

Re-export surface for ``animus_kernel.budget`` (ADL-20260806-001 Phase 1+3).

The kernel package is the canonical home for budget primitives. This module
exists so ``from animus_forge.budget import BudgetManager`` (and every other
name in ``__all__``) keeps working unchanged.

Phase 3 emits a ``DeprecationWarning`` on every import path through this
package — please migrate to ``animus_kernel.budget``. The package will be
removed in Phase 4.
"""

import warnings

warnings.warn(
    "animus_forge.budget is deprecated; import from animus_kernel.budget "
    "instead (ADL-20260806-001). The forge-side package will be removed in "
    "the next minor release.",
    DeprecationWarning,
    stacklevel=2,
)

from animus_kernel.budget import *  # noqa: F401, F403
from animus_kernel.budget import get_budget_tracker, reset_budget_tracker

__all__ = [
    # In-memory budget tracking
    "BudgetManager",
    "BudgetConfig",
    "BudgetStatus",
    "UsageRecord",
    "effective_tokens",
    "DEFAULT_MODEL_MULTIPLIERS",
    "AllocationStrategy",
    "EqualAllocation",
    "PriorityAllocation",
    "AdaptiveAllocation",
    "PreflightValidator",
    "ValidationResult",
    "ValidationStatus",
    "WorkflowEstimate",
    "StepEstimate",
    "validate_workflow_budget",
    "get_budget_tracker",
    "reset_budget_tracker",
    # Persistent budget management
    "Budget",
    "BudgetCreate",
    "BudgetUpdate",
    "BudgetPeriod",
    "BudgetSummary",
    "PersistentBudgetManager",
]
