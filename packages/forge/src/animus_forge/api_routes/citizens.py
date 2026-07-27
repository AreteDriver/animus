"""API routes for Research Citizen missions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, CRUD_RESPONSES, bad_request, not_found
from animus_forge.api_routes.auth import verify_auth
from animus_forge.citizens.mission import MissionConfig

router = APIRouter()


@router.post("/citizens/research/commission", responses=AUTH_RESPONSES)
def commission_research_mission(
    request: dict[str, Any],
    authorization: str | None = Header(None),
):
    """Commission a new research mission.

    Body must include at least ``objective``, ``eval_suite``, and
    ``workflow_template``.  Optional fields: ``max_iterations``,
    ``min_pass_rate``, ``max_variance``.
    """
    verify_auth(authorization)

    if state.citizen_commissioner is None:
        raise bad_request("Citizen commissioner not initialized")

    required = {"objective", "eval_suite", "workflow_template"}
    missing = required - set(request.keys())
    if missing:
        raise bad_request(f"Missing required fields: {', '.join(missing)}")

    config = MissionConfig(
        objective=request["objective"],
        eval_suite=request["eval_suite"],
        workflow_template=request["workflow_template"],
        max_iterations=request.get("max_iterations", 3),
        min_pass_rate=request.get("min_pass_rate", 0.9),
        max_variance=request.get("max_variance", 0.1),
        metadata=request.get("metadata", {}),
    )

    mission_id = state.citizen_commissioner.commission(config)
    return {"status": "commissioned", "mission_id": mission_id}


@router.get("/citizens/research/{mission_id}", responses=AUTH_RESPONSES)
def get_research_mission(
    mission_id: str,
    authorization: str | None = Header(None),
):
    """Get the current status of a research mission."""
    verify_auth(authorization)

    if state.citizen_commissioner is None:
        raise bad_request("Citizen commissioner not initialized")

    status = state.citizen_commissioner.status(mission_id)
    if status is None:
        raise not_found("Mission", mission_id)
    return status


@router.post("/citizens/research/{mission_id}/run", responses=AUTH_RESPONSES)
def run_research_mission_iteration(
    mission_id: str,
    authorization: str | None = Header(None),
):
    """Run a single iteration of a research mission."""
    verify_auth(authorization)

    if state.citizen_commissioner is None:
        raise bad_request("Citizen commissioner not initialized")

    result = state.citizen_commissioner.run(mission_id)
    if result is None:
        raise not_found("Mission", mission_id)
    return result


@router.get("/citizens/research", responses=AUTH_RESPONSES)
def list_research_missions(
    state_filter: str | None = None,
    limit: int = 20,
    authorization: str | None = Header(None),
):
    """List research missions, optionally filtered by state."""
    verify_auth(authorization)

    if state.citizen_commissioner is None:
        raise bad_request("Citizen commissioner not initialized")

    missions = state.citizen_commissioner.list(state=state_filter, limit=limit)
    return {"missions": missions, "count": len(missions)}
