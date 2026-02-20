"""Execution management endpoints."""

from __future__ import annotations

import asyncio
import json as json_mod

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from animus_forge import api_state as state
from animus_forge.api_errors import (
    AUTH_RESPONSES,
    CRUD_RESPONSES,
    bad_request,
    internal_error,
    not_found,
)
from animus_forge.api_routes.auth import verify_auth

router = APIRouter()


@router.get("/executions", responses=AUTH_RESPONSES)
def list_executions(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    workflow_id: str | None = None,
    authorization: str | None = Header(None),
):
    """List workflow executions with pagination."""
    verify_auth(authorization)

    from animus_forge.executions import ExecutionStatus

    status_filter = None
    if status:
        try:
            status_filter = ExecutionStatus(status)
        except ValueError:
            raise bad_request(
                f"Invalid status: {status}",
                {"valid_statuses": [s.value for s in ExecutionStatus]},
            )

    result = state.execution_manager.list_executions(
        page=page,
        page_size=page_size,
        status=status_filter,
        workflow_id=workflow_id,
    )
    return {
        "data": [e.model_dump(mode="json") for e in result.data],
        "total": result.total,
        "page": result.page,
        "page_size": result.page_size,
        "total_pages": result.total_pages,
    }


@router.get("/executions/{execution_id}", responses=CRUD_RESPONSES)
def get_execution(execution_id: str, authorization: str | None = Header(None)):
    """Get a specific execution by ID."""
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    execution.logs = state.execution_manager.get_logs(execution_id, limit=50)
    execution.metrics = state.execution_manager.get_metrics(execution_id)

    return execution.model_dump(mode="json")


@router.get("/executions/{execution_id}/logs", responses=CRUD_RESPONSES)
def get_execution_logs(
    execution_id: str,
    limit: int = 100,
    level: str | None = None,
    authorization: str | None = Header(None),
):
    """Get logs for an execution."""
    verify_auth(authorization)

    from animus_forge.executions import LogLevel

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    level_filter = None
    if level:
        try:
            level_filter = LogLevel(level)
        except ValueError:
            raise bad_request(
                f"Invalid level: {level}",
                {"valid_levels": [lvl.value for lvl in LogLevel]},
            )

    logs = state.execution_manager.get_logs(execution_id, limit=limit, level=level_filter)
    return [log.model_dump(mode="json") for log in logs]


@router.get("/executions/{execution_id}/stream")
async def stream_execution(
    execution_id: str,
    request: Request,
    authorization: str | None = Header(None),
):
    """Stream execution events via Server-Sent Events (SSE).

    Sends an initial snapshot, then real-time log/progress/metrics/status events.
    Completes with an 'event: done' when the execution finishes.
    """
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _callback(event_type: str, eid: str, **kwargs):
        """Push events from ExecutionManager into the async queue."""
        if eid != execution_id:
            return
        loop.call_soon_threadsafe(queue.put_nowait, {"event": event_type, **kwargs})

    state.execution_manager.register_callback(_callback)

    async def _event_generator():
        try:
            # Initial snapshot
            snap = execution.model_dump(mode="json")
            snap["logs"] = [
                log.model_dump(mode="json")
                for log in state.execution_manager.get_logs(execution_id, limit=50)
            ]
            metrics = state.execution_manager.get_metrics(execution_id)
            if metrics:
                snap["metrics"] = metrics.model_dump(mode="json")
            yield f"event: snapshot\ndata: {json_mod.dumps(snap)}\n\n"

            # Check if already terminal
            from animus_forge.executions import ExecutionStatus

            terminal = {
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
            if execution.status in terminal:
                yield "event: done\ndata: {}\n\n"
                return

            # Stream live events
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    event_type = event.pop("event", "update")
                    yield f"event: {event_type}\ndata: {json_mod.dumps(event, default=str)}\n\n"

                    # If status event indicates terminal state, send done
                    if event_type == "status" and event.get("status") in {
                        s.value for s in terminal
                    }:
                        yield "event: done\ndata: {}\n\n"
                        break
                except TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        finally:
            state.execution_manager.unregister_callback(_callback)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/executions/{execution_id}/pause", responses=CRUD_RESPONSES)
def pause_execution(execution_id: str, authorization: str | None = Header(None)):
    """Pause a running execution."""
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    from animus_forge.executions import ExecutionStatus

    if execution.status != ExecutionStatus.RUNNING:
        raise bad_request(
            f"Cannot pause execution in {execution.status.value} status",
            {"execution_id": execution_id, "current_status": execution.status.value},
        )

    updated = state.execution_manager.pause_execution(execution_id)
    return {
        "status": "success",
        "execution_id": execution_id,
        "execution_status": updated.status.value if updated else "unknown",
    }


@router.post("/executions/{execution_id}/resume", responses=CRUD_RESPONSES)
def resume_execution(
    execution_id: str,
    body: dict | None = None,
    authorization: str | None = Header(None),
):
    """Resume a paused or approval-awaiting execution.

    For approval-gated executions, provide a token in the body:
    {"token": "...", "approve": true}
    """
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    from animus_forge.executions import ExecutionStatus

    # Handle approval token resume
    if execution.status == ExecutionStatus.AWAITING_APPROVAL:
        if not body or not body.get("token"):
            raise bad_request(
                "Resume token required for approval-gated execution",
                {"execution_id": execution_id},
            )

        from animus_forge.workflow.approval_store import get_approval_store

        store = get_approval_store()
        token_data = store.get_by_token(body["token"])

        if not token_data or token_data["execution_id"] != execution_id:
            raise bad_request(
                "Invalid or expired resume token",
                {"execution_id": execution_id},
            )

        approve = body.get("approve", True)
        approved_by = body.get("approved_by", "api")

        if not approve:
            store.reject(body["token"], rejected_by=approved_by)
            state.execution_manager.complete_execution(execution_id, error="Approval rejected")
            return {
                "status": "rejected",
                "execution_id": execution_id,
                "reason": body.get("reason", ""),
            }

        # Approve and resume
        store.approve(body["token"], approved_by=approved_by)
        context = token_data.get("context", {})
        next_step_id = token_data.get("next_step_id", "")

        # Re-execute workflow from the step after approval
        try:
            # Mark execution as running again
            state.execution_manager.resume_execution(execution_id)

            return {
                "status": "approved",
                "execution_id": execution_id,
                "resume_from": next_step_id,
                "context_keys": list(context.keys()) if isinstance(context, dict) else [],
            }
        except Exception as e:
            raise internal_error(f"Failed to resume workflow: {e}")

    # Standard PAUSED resume
    if execution.status != ExecutionStatus.PAUSED:
        raise bad_request(
            f"Cannot resume execution in {execution.status.value} status",
            {"execution_id": execution_id, "current_status": execution.status.value},
        )

    updated = state.execution_manager.resume_execution(execution_id)
    return {
        "status": "success",
        "execution_id": execution_id,
        "execution_status": updated.status.value if updated else "unknown",
    }


@router.get("/executions/{execution_id}/approval", responses=CRUD_RESPONSES)
def get_approval_status(execution_id: str, authorization: str | None = Header(None)):
    """Get pending approval details for an execution."""
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    from animus_forge.workflow.approval_store import get_approval_store

    tokens = get_approval_store().get_by_execution(execution_id)
    pending = [
        {
            "token": t["token"],
            "step_id": t["step_id"],
            "prompt": t.get("prompt", ""),
            "preview": t.get("preview", {}),
            "timeout_at": t.get("timeout_at"),
            "created_at": t.get("created_at"),
            "status": t["status"],
        }
        for t in tokens
        if t["status"] == "pending"
    ]

    return {
        "execution_id": execution_id,
        "pending_approvals": pending,
        "total_tokens": len(tokens),
    }


@router.post("/executions/{execution_id}/cancel", responses=CRUD_RESPONSES)
def cancel_execution(execution_id: str, authorization: str | None = Header(None)):
    """Cancel a running or paused execution."""
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    if state.execution_manager.cancel_execution(execution_id):
        return {"status": "success", "message": "Execution cancelled"}

    raise bad_request(
        f"Cannot cancel execution in {execution.status.value} status",
        {"execution_id": execution_id, "current_status": execution.status.value},
    )


@router.delete("/executions/{execution_id}", responses=CRUD_RESPONSES)
def delete_execution(execution_id: str, authorization: str | None = Header(None)):
    """Delete an execution (must be completed/failed/cancelled)."""
    verify_auth(authorization)

    execution = state.execution_manager.get_execution(execution_id)
    if not execution:
        raise not_found("Execution", execution_id)

    from animus_forge.executions import ExecutionStatus

    if execution.status in (
        ExecutionStatus.PENDING,
        ExecutionStatus.RUNNING,
        ExecutionStatus.PAUSED,
    ):
        raise bad_request(
            f"Cannot delete execution in {execution.status.value} status",
            {"execution_id": execution_id, "current_status": execution.status.value},
        )

    if state.execution_manager.delete_execution(execution_id):
        return {"status": "success"}

    raise internal_error("Failed to delete execution")


@router.post("/executions/cleanup")
def cleanup_executions(max_age_hours: int = 168, authorization: str | None = Header(None)):
    """Remove old completed/failed/cancelled executions (default 7 days)."""
    verify_auth(authorization)
    deleted = state.execution_manager.cleanup_old_executions(max_age_hours)
    return {"status": "success", "deleted": deleted}
