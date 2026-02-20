"""Schedule management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header

from animus_forge import api_state as state
from animus_forge.api_errors import (
    AUTH_RESPONSES,
    CRUD_RESPONSES,
    bad_request,
    internal_error,
    not_found,
)
from animus_forge.api_routes.auth import verify_auth
from animus_forge.scheduler import WorkflowSchedule

router = APIRouter()


@router.get("/schedules", responses=AUTH_RESPONSES)
def list_schedules(authorization: str | None = Header(None)):
    """List all schedules."""
    verify_auth(authorization)
    return state.schedule_manager.list_schedules()


@router.get("/schedules/{schedule_id}", responses=CRUD_RESPONSES)
def get_schedule(schedule_id: str, authorization: str | None = Header(None)):
    """Get a specific schedule."""
    verify_auth(authorization)
    schedule = state.schedule_manager.get_schedule(schedule_id)

    if not schedule:
        raise not_found("Schedule", schedule_id)

    return schedule


@router.post("/schedules", responses=CRUD_RESPONSES)
def create_schedule(schedule: WorkflowSchedule, authorization: str | None = Header(None)):
    """Create a new schedule."""
    verify_auth(authorization)

    try:
        if state.schedule_manager.create_schedule(schedule):
            return {"status": "success", "schedule_id": schedule.id}
        raise internal_error("Failed to save schedule")
    except ValueError as e:
        raise bad_request(str(e))


@router.put("/schedules/{schedule_id}", responses=CRUD_RESPONSES)
def update_schedule(
    schedule_id: str,
    schedule: WorkflowSchedule,
    authorization: str | None = Header(None),
):
    """Update an existing schedule."""
    verify_auth(authorization)

    if schedule.id != schedule_id:
        raise bad_request("Schedule ID mismatch", {"expected": schedule_id, "got": schedule.id})

    try:
        if state.schedule_manager.update_schedule(schedule):
            return {"status": "success", "schedule_id": schedule.id}
        raise internal_error("Failed to update schedule")
    except ValueError:
        raise not_found("Schedule", schedule_id)


@router.delete("/schedules/{schedule_id}", responses=CRUD_RESPONSES)
def delete_schedule(schedule_id: str, authorization: str | None = Header(None)):
    """Delete a schedule."""
    verify_auth(authorization)

    if state.schedule_manager.delete_schedule(schedule_id):
        return {"status": "success"}

    raise not_found("Schedule", schedule_id)


@router.post("/schedules/{schedule_id}/pause", responses=CRUD_RESPONSES)
def pause_schedule(schedule_id: str, authorization: str | None = Header(None)):
    """Pause a schedule."""
    verify_auth(authorization)

    if state.schedule_manager.pause_schedule(schedule_id):
        return {"status": "success", "message": "Schedule paused"}

    raise not_found("Schedule", schedule_id)


@router.post("/schedules/{schedule_id}/resume", responses=CRUD_RESPONSES)
def resume_schedule(schedule_id: str, authorization: str | None = Header(None)):
    """Resume a paused schedule."""
    verify_auth(authorization)

    if state.schedule_manager.resume_schedule(schedule_id):
        return {"status": "success", "message": "Schedule resumed"}

    raise not_found("Schedule", schedule_id)


@router.post("/schedules/{schedule_id}/trigger", responses=CRUD_RESPONSES)
def trigger_schedule(schedule_id: str, authorization: str | None = Header(None)):
    """Manually trigger a scheduled workflow immediately."""
    verify_auth(authorization)

    if state.schedule_manager.trigger_now(schedule_id):
        return {"status": "success", "message": "Workflow triggered"}

    raise not_found("Schedule", schedule_id)


@router.get("/schedules/{schedule_id}/history", responses=CRUD_RESPONSES)
def get_schedule_history(
    schedule_id: str,
    limit: int = 10,
    authorization: str | None = Header(None),
):
    """Get execution history for a schedule."""
    verify_auth(authorization)

    schedule = state.schedule_manager.get_schedule(schedule_id)
    if not schedule:
        raise not_found("Schedule", schedule_id)

    history = state.schedule_manager.get_execution_history(schedule_id, limit)
    return [h.model_dump(mode="json") for h in history]
