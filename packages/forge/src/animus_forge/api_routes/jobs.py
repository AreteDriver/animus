"""Job management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, CRUD_RESPONSES, bad_request, not_found
from animus_forge.api_models import WorkflowExecuteRequest
from animus_forge.api_routes.auth import verify_auth
from animus_forge.jobs import JobStatus

router = APIRouter()


@router.post("/jobs", responses=CRUD_RESPONSES)
@state.limiter.limit("20/minute")
def submit_job(
    request_obj: Request,
    request: WorkflowExecuteRequest,
    authorization: str | None = Header(None),
):
    """Submit a workflow for async execution. Rate limited to 20 jobs/minute per IP."""
    verify_auth(authorization)

    try:
        job = state.job_manager.submit(request.workflow_id, request.variables)
        return {
            "status": "submitted",
            "job_id": job.id,
            "workflow_id": job.workflow_id,
            "poll_url": f"/jobs/{job.id}",
        }
    except ValueError as e:
        raise bad_request(str(e))


@router.get("/jobs", responses=AUTH_RESPONSES)
def list_jobs(
    status: str | None = None,
    workflow_id: str | None = None,
    limit: int = 50,
    authorization: str | None = Header(None),
):
    """List jobs with optional filtering."""
    verify_auth(authorization)

    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise bad_request(
                f"Invalid status: {status}",
                {"valid_statuses": [s.value for s in JobStatus]},
            )

    jobs = state.job_manager.list_jobs(status=status_filter, workflow_id=workflow_id, limit=limit)
    return [j.model_dump(mode="json") for j in jobs]


@router.get("/jobs/stats", responses=AUTH_RESPONSES)
def get_job_stats(authorization: str | None = Header(None)):
    """Get job statistics."""
    verify_auth(authorization)
    return state.job_manager.get_stats()


@router.get("/jobs/{job_id}", responses=CRUD_RESPONSES)
def get_job(job_id: str, authorization: str | None = Header(None)):
    """Get job status and result."""
    verify_auth(authorization)

    job = state.job_manager.get_job(job_id)
    if not job:
        raise not_found("Job", job_id)

    return job.model_dump(mode="json")


@router.post("/jobs/{job_id}/cancel", responses=CRUD_RESPONSES)
def cancel_job(job_id: str, authorization: str | None = Header(None)):
    """Cancel a pending or running job."""
    verify_auth(authorization)

    if state.job_manager.cancel(job_id):
        return {"status": "success", "message": "Job cancelled"}

    job = state.job_manager.get_job(job_id)
    if not job:
        raise not_found("Job", job_id)

    raise bad_request(
        f"Cannot cancel job in {job.status.value} status",
        {"job_id": job_id, "current_status": job.status.value},
    )


@router.delete("/jobs/{job_id}", responses=CRUD_RESPONSES)
def delete_job(job_id: str, authorization: str | None = Header(None)):
    """Delete a completed/failed/cancelled job."""
    verify_auth(authorization)

    if state.job_manager.delete_job(job_id):
        return {"status": "success"}

    job = state.job_manager.get_job(job_id)
    if not job:
        raise not_found("Job", job_id)

    raise bad_request("Cannot delete running job", {"job_id": job_id, "status": job.status.value})


@router.post("/jobs/cleanup")
def cleanup_jobs(max_age_hours: int = 24, authorization: str | None = Header(None)):
    """Remove old completed/failed/cancelled jobs."""
    verify_auth(authorization)
    deleted = state.job_manager.cleanup_old_jobs(max_age_hours)
    return {"status": "success", "deleted": deleted}
