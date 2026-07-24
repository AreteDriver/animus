"""Bridge between Arete Tools (Autopsy) and Animus Quorum.

Maps autopsy failure types to stability score deltas via PhiScorer
and leaves stigmergy markers for failure pattern awareness.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import autopsy.analyzer  # noqa: F401

    HAS_AUTOPSY = True
except ImportError:
    HAS_AUTOPSY = False

__all__ = [
    "HAS_AUTOPSY",
    "FAILURE_SEVERITY",
    "autopsy",
    "record_failure_outcome",
    "leave_autopsy_marker",
]

# Maps autopsy failure_type → PhiScorer outcome
FAILURE_SEVERITY: dict[str, str] = {
    "goal_necrosis": "failed",
    "goal_cancer": "failed",
    "goal_autoimmunity": "failed",
    "tool_hallucination": "rejected",
    "tool_loop": "rejected",
    "overconfidence": "rejected",
    "unknown": "rejected",
}


def record_failure_outcome(
    scorer,
    agent_id: str,
    failure_type: str,
    domain: str,
) -> float:
    """Record an autopsy failure as a PhiScorer outcome.

    Args:
        scorer: PhiScorer instance
        agent_id: Agent that experienced the failure
        failure_type: Autopsy failure classification
        domain: Skill domain where failure occurred

    Returns:
        Updated phi score for the agent
    """
    outcome = FAILURE_SEVERITY.get(failure_type, "rejected")
    new_score = scorer.record_outcome(
        agent_id=agent_id,
        skill_domain=domain,
        outcome=outcome,
    )
    logger.info(
        "Recorded failure outcome for %s: %s → %s (phi=%.3f)",
        agent_id,
        failure_type,
        outcome,
        new_score,
    )
    return new_score


def leave_autopsy_marker(
    field,
    agent_id: str,
    failure_type: str,
    target: str,
    details: str,
) -> object:
    """Leave a stigmergy marker recording a failure pattern.

    Args:
        field: StigmergyField instance
        agent_id: Agent that experienced the failure
        failure_type: Autopsy failure classification
        target: What failed (workflow ID, step ID, etc.)
        details: Human-readable failure details

    Returns:
        StigmergyMarker object
    """
    severity = FAILURE_SEVERITY.get(failure_type, "rejected")
    content = f"[{failure_type}] {details}"
    strength = 1.0 if severity == "failed" else 0.7

    marker = field.leave_marker(
        agent_id=agent_id,
        marker_type="failure_pattern",
        target=target,
        content=content,
        strength=strength,
    )
    logger.info(
        "Left autopsy marker for %s on %s (type=%s, strength=%.1f)",
        agent_id,
        target,
        failure_type,
        strength,
    )
    return marker
