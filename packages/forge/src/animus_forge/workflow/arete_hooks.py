"""Arete Tools runtime hooks for WorkflowExecutor.

Provides error callbacks and post-workflow hooks that connect Forge
execution events to Quorum scoring and Core memory bridges.

All imports are optional — graceful no-op when tools/packages not installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from convergent.arete_bridge import (
        leave_autopsy_marker,
        record_failure_outcome,
    )

    HAS_QUORUM_BRIDGE = True
except ImportError:
    HAS_QUORUM_BRIDGE = False

try:
    from animus.integrations.arete_bridge import auto_sync_verdicts

    HAS_CORE_BRIDGE = True
except ImportError:
    HAS_CORE_BRIDGE = False


class AreteHooks:
    """Runtime hooks connecting Forge workflow events to Arete Tool bridges.

    Usage:
        hooks = AreteHooks(phi_scorer=scorer, stigmergy_field=field)
        executor = WorkflowExecutor(
            error_callback=hooks.on_step_failure,
        )

    All operations are optional and no-op when dependencies are missing.
    """

    def __init__(
        self,
        phi_scorer: Any | None = None,
        stigmergy_field: Any | None = None,
        memory_layer: Any | None = None,
        default_agent_id: str = "forge",
    ):
        """Initialize hooks.

        Args:
            phi_scorer: Quorum PhiScorer instance for stability scoring
            stigmergy_field: Quorum StigmergyField for failure markers
            memory_layer: Core MemoryLayer for verdict→memory sync
            default_agent_id: Agent ID to use for scoring/markers
        """
        self._phi_scorer = phi_scorer
        self._stigmergy_field = stigmergy_field
        self._memory_layer = memory_layer
        self._agent_id = default_agent_id

    def on_step_failure(
        self,
        step_id: str,
        workflow_id: str,
        error: Exception,
    ) -> None:
        """Error callback for WorkflowExecutor.

        Records failure in Quorum phi scorer and leaves stigmergy marker.
        Matches the error_callback signature: (step_id, workflow_id, error).
        """
        if not HAS_QUORUM_BRIDGE:
            return

        error_text = str(error)
        failure_type = _classify_failure(error_text)
        domain = _extract_domain(step_id)

        if self._phi_scorer is not None:
            try:
                new_score = record_failure_outcome(
                    scorer=self._phi_scorer,
                    agent_id=self._agent_id,
                    failure_type=failure_type,
                    domain=domain,
                )
                logger.info(
                    "Arete hook: recorded failure %s for %s (phi=%.3f)",
                    failure_type,
                    step_id,
                    new_score,
                )
            except Exception:
                logger.warning("Arete hook: phi scoring failed for %s", step_id)

        if self._stigmergy_field is not None:
            try:
                leave_autopsy_marker(
                    field=self._stigmergy_field,
                    agent_id=self._agent_id,
                    failure_type=failure_type,
                    target=f"{workflow_id}/{step_id}",
                    details=error_text[:500],
                )
            except Exception:
                logger.warning("Arete hook: stigmergy marker failed for %s", step_id)

    def on_workflow_complete(self, workflow_id: str, status: str) -> None:
        """Post-workflow hook. Syncs verdicts to Core memory on success.

        Called from _finalize_workflow or execution_manager callback.
        """
        if status != "success":
            return

        if not HAS_CORE_BRIDGE or self._memory_layer is None:
            return

        try:
            count = auto_sync_verdicts(self._memory_layer)
            if count > 0:
                logger.info(
                    "Arete hook: synced %d verdicts after workflow %s",
                    count,
                    workflow_id,
                )
        except Exception:
            logger.warning("Arete hook: verdict sync failed after %s", workflow_id)


def _classify_failure(error_text: str) -> str:
    """Heuristic classification of error text into autopsy failure types."""
    text = error_text.lower()

    if any(k in text for k in ("loop", "infinite", "recursion", "max retries", "retry")):
        return "tool_loop"
    if any(k in text for k in ("hallucin", "not found", "no such", "does not exist")):
        return "tool_hallucination"
    if any(k in text for k in ("confidence", "certain", "threshold")):
        return "overconfidence"
    if any(k in text for k in ("conflict", "contradict", "incompatible")):
        return "goal_autoimmunity"
    if any(k in text for k in ("scope", "bloat", "unbounded", "grew")):
        return "goal_cancer"
    if any(k in text for k in ("stale", "abandoned", "orphan", "dead")):
        return "goal_necrosis"

    return "unknown"


def _extract_domain(step_id: str) -> str:
    """Extract a skill domain from step ID (best-effort)."""
    # step IDs like "security_review", "code_audit", etc.
    parts = step_id.split("_")
    if len(parts) >= 2:
        return parts[0]
    return "general"


def get_arete_hooks() -> AreteHooks | None:
    """Factory: build AreteHooks from available Quorum/Core primitives.

    Returns None when no Arete Tool bridges are installed.
    """
    phi_scorer = None
    stigmergy_field = None
    memory_layer = None

    try:
        from convergent.scoring import PhiScorer, ScoreStore

        phi_scorer = PhiScorer(store=ScoreStore())
    except Exception:
        logger.debug("PhiScorer not available for Arete hooks")

    try:
        from convergent.stigmergy import StigmergyField

        stigmergy_field = StigmergyField()
    except Exception:
        logger.debug("StigmergyField not available for Arete hooks")

    try:
        from pathlib import Path

        from animus.memory import MemoryLayer

        memory_layer = MemoryLayer(data_dir=Path.home() / ".animus" / "memory")
    except Exception:
        logger.debug("MemoryLayer not available for Arete hooks")

    if phi_scorer is None and stigmergy_field is None and memory_layer is None:
        return None

    return AreteHooks(
        phi_scorer=phi_scorer,
        stigmergy_field=stigmergy_field,
        memory_layer=memory_layer,
    )
