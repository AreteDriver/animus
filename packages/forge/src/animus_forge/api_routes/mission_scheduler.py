"""Mission scheduler control endpoints.

Provides start/stop/status for the continuous autonomous mission scheduler.
"""

from __future__ import annotations

from fastapi import APIRouter, Header

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, bad_request, not_found
from animus_forge.api_routes.auth import verify_auth

router = APIRouter()


@router.post("/scheduler/start", responses=AUTH_RESPONSES)
async def start_scheduler(authorization: str | None = Header(None)):
    """Start the mission scheduler if it is not already running."""
    verify_auth(authorization)

    if state.mission_scheduler is None:
        raise bad_request("Mission scheduler not initialized")

    if state.mission_scheduler._stopped.is_set() is False:
        return {"status": "already_running"}

    await state.mission_scheduler.start()
    return {"status": "started"}


@router.post("/scheduler/stop", responses=AUTH_RESPONSES)
async def stop_scheduler(authorization: str | None = Header(None)):
    """Stop the mission scheduler gracefully."""
    verify_auth(authorization)

    if state.mission_scheduler is None:
        raise bad_request("Mission scheduler not initialized")

    if state.mission_scheduler._stopped.is_set():
        return {"status": "already_stopped"}

    await state.mission_scheduler.stop()
    return {"status": "stopped"}


@router.get("/scheduler/status", responses=AUTH_RESPONSES)
def get_scheduler_status(authorization: str | None = Header(None)):
    """Return the current scheduler status snapshot."""
    verify_auth(authorization)

    if state.mission_scheduler is None:
        raise bad_request("Mission scheduler not initialized")

    return state.mission_scheduler.status()


@router.get("/scheduler/metrics", responses=AUTH_RESPONSES)
def get_scheduler_metrics(
    mission_id: str | None = None,
    authorization: str | None = Header(None),
):
    """Return scheduler metrics.

    Optionally filter by mission_id to get per-mission events.
    """
    verify_auth(authorization)

    if state.mission_scheduler is None:
        raise bad_request("Mission scheduler not initialized")

    if state.mission_scheduler.metrics is None:
        raise bad_request("Metrics not initialized")

    if mission_id:
        return {
            "mission_id": mission_id,
            "events": state.mission_scheduler.metrics.by_mission(mission_id),
        }

    return state.mission_scheduler.metrics.summary()
