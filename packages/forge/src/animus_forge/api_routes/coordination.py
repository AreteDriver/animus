"""Coordination health, cycle detection, and event log API endpoints.

Exposes Convergent Phase 4 features (health dashboard, dependency cycles,
event log) via REST API. All endpoints degrade gracefully when Convergent
is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from animus_forge import api_state as state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coordination", tags=["coordination"])


def _get_bridge() -> Any:
    """Retrieve the coordination bridge from app state."""
    return state.coordination_bridge


@router.get("/health")
def coordination_health() -> dict:
    """Get coordination health report.

    Returns aggregated metrics from intent graph, stigmergy, phi scores,
    and voting subsystems with an A-F grade.
    """
    try:
        from animus_forge.agents.convergence import HAS_CONVERGENT, get_coordination_health

        if not HAS_CONVERGENT:
            return {"available": False, "reason": "convergent not installed"}

        # Try to get bridge from the module-level variable in api.py lifespan
        # The bridge is stored as a local in lifespan and passed to supervisors,
        # but not directly in api_state. We use the health checker on the store.
        bridge = _get_bridge()
        if bridge is None:
            return {"available": True, "reason": "no active coordination bridge"}

        health = get_coordination_health(bridge)
        if not health:
            return {"available": True, "reason": "health check returned no data"}

        return {"available": True, **health}
    except Exception:
        logger.warning("Coordination health endpoint failed", exc_info=True)
        return {"available": False, "error": "internal error"}


@router.get("/cycles")
def coordination_cycles() -> dict:
    """Check for dependency cycles in the intent graph.

    Returns any detected circular dependencies that could cause
    agent deadlocks.
    """
    try:
        from animus_forge.agents.convergence import HAS_CONVERGENT, check_dependency_cycles

        if not HAS_CONVERGENT:
            return {"available": False, "reason": "convergent not installed"}

        # Cycles require a resolver — create a fresh one for inspection
        from convergent import IntentResolver, PythonGraphBackend

        resolver = IntentResolver(backend=PythonGraphBackend())
        cycles = check_dependency_cycles(resolver)
        return {
            "available": True,
            "cycle_count": len(cycles),
            "cycles": cycles,
        }
    except Exception:
        logger.warning("Coordination cycles endpoint failed", exc_info=True)
        return {"available": False, "error": "internal error"}


@router.get("/events")
def coordination_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    agent: str | None = Query(None, description="Filter by agent ID"),
    limit: int = Query(50, ge=1, le=500, description="Max events to return"),
) -> dict:
    """Query coordination event log.

    Returns a timeline of coordination events (intent publishes, votes,
    decisions, markers, score updates).
    """
    try:
        from animus_forge.agents.convergence import HAS_CONVERGENT

        if not HAS_CONVERGENT:
            return {"available": False, "reason": "convergent not installed"}

        event_log = state.coordination_event_log
        if event_log is None:
            return {"available": True, "reason": "no active event log", "events": []}

        from convergent import EventType

        et = None
        if event_type is not None:
            try:
                et = EventType(event_type)
            except ValueError:
                return {
                    "available": True,
                    "error": f"Unknown event type: {event_type}",
                    "valid_types": [e.value for e in EventType],
                }

        events = event_log.query(event_type=et, agent_id=agent, limit=limit)
        return {
            "available": True,
            "count": len(events),
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "agent_id": e.agent_id,
                    "timestamp": e.timestamp,
                    "payload": e.payload,
                    "correlation_id": e.correlation_id,
                }
                for e in events
            ],
        }
    except Exception:
        logger.warning("Coordination events endpoint failed", exc_info=True)
        return {"available": False, "error": "internal error"}
