"""Arete Guard — pre-execution evidence gate for workflow steps.

Prevents a citizen from burning tokens on workflows whose recent eval
history shows they are demonstrably broken.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_forge.evaluation.store import EvalStore

logger = logging.getLogger(__name__)

# Environment variable: block | warn | log (default: warn)
_DEFAULT_GUARD_MODE = "warn"
_GUARD_THRESHOLD = 0.5


class AreteGuard:
    """Checks eval evidence before allowing workflow execution.

    Args:
        eval_store: Persistent eval store to query.
        mode: Behaviour when evidence is poor — ``block``, ``warn``, or ``log``.
            If omitted, falls back to ``ARETE_GUARD_MODE`` env var.
    """

    def __init__(self, eval_store: EvalStore, mode: str | None = None):
        self.eval_store = eval_store
        self.mode = (mode or os.environ.get("ARETE_GUARD_MODE", _DEFAULT_GUARD_MODE)).lower()

    def check(
        self,
        workflow_name: str,
        mission_id: str | None = None,
        *,
        agent_role: str = "research_citizen",
        lookback_minutes: int = 60,
    ) -> bool:
        """Return True if execution is allowed, False if blocked.

        Looks for the most recent eval run for this workflow / mission.
        If ``pass_rate < _GUARD_THRESHOLD`` the guard triggers according
        to ``self.mode``.
        """
        # Query recent eval runs for this agent_role + workflow
        runs = self.eval_store.query_runs(
            agent_role=agent_role,
            limit=5,
        )
        if not runs:
            return True

        # Filter to runs that mention this workflow in metadata
        recent_run = None
        for run in runs:
            meta = run.get("metadata") or {}
            if meta.get("workflow_id") == workflow_name or meta.get("mission_id") == mission_id:
                recent_run = run
                break

        if recent_run is None:
            # No evidence for this specific workflow — allow
            return True

        pass_rate = float(recent_run.get("pass_rate", 1.0))
        if pass_rate >= _GUARD_THRESHOLD:
            return True

        msg = (
            f"AreteGuard triggered for workflow '{workflow_name}' "
            f"(mission={mission_id}): pass_rate={pass_rate:.2f} < {_GUARD_THRESHOLD}"
        )

        if self.mode == "block":
            logger.warning("%s — BLOCKED", msg)
            raise AreteGuardError(msg)
        elif self.mode == "warn":
            logger.warning("%s — WARNING (execution continues)", msg)
        else:
            logger.info("%s — LOGGED", msg)

        return True  # warn / log modes never block


class AreteGuardError(RuntimeError):
    """Raised when AreteGuard is in ``block`` mode and evidence is poor."""
