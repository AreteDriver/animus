"""Agent infrastructure endpoints.

Exposes process registry, autonomy status, and agent memory via REST API.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, Query

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, not_found
from animus_forge.api_routes.auth import verify_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------


@router.get("/processes", responses=AUTH_RESPONSES)
def list_processes(
    process_type: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, le=200),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """List active and recent processes from the registry."""
    verify_auth(authorization)

    registry = _get_process_registry()
    if registry is None:
        return {"processes": [], "total": 0, "message": "Process registry not available"}

    try:
        from animus_forge.agents.process_registry import ProcessState, ProcessType

        ptype = ProcessType(process_type) if process_type else None
        pstate = ProcessState(status) if status else None
        processes = registry.list_all(process_type=ptype, state=pstate)
        entries = [p.to_dict() for p in processes[:limit]]
        return {"processes": entries, "total": len(entries)}
    except (ValueError, KeyError) as e:
        return {"processes": [], "total": 0, "error": str(e)}


@router.get("/processes/{process_id}", responses=AUTH_RESPONSES)
def get_process(
    process_id: str,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Get a specific process by ID."""
    verify_auth(authorization)

    registry = _get_process_registry()
    if registry is None:
        raise not_found("Process registry", "unavailable")

    entry = registry.get(process_id)
    if entry is None:
        raise not_found("Process", process_id)

    return entry.to_dict()


# ---------------------------------------------------------------------------
# Agent memory
# ---------------------------------------------------------------------------


@router.get("/memory/{agent_id}", responses=AUTH_RESPONSES)
def get_agent_memory(
    agent_id: str,
    memory_type: str | None = None,
    limit: int = Query(default=20, le=100),
    min_importance: float = Query(default=0.0, ge=0.0, le=1.0),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Retrieve memories for a specific agent."""
    verify_auth(authorization)

    memory = _get_agent_memory()
    if memory is None:
        return {"agent_id": agent_id, "memories": [], "message": "Agent memory not available"}

    entries = memory.recall(
        agent_id=agent_id,
        memory_type=memory_type,
        limit=limit,
        min_importance=min_importance,
    )
    return {
        "agent_id": agent_id,
        "memories": [_memory_to_dict(m) for m in entries],
        "total": len(entries),
    }


@router.get("/memory/{agent_id}/stats", responses=AUTH_RESPONSES)
def get_agent_memory_stats(
    agent_id: str,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Get memory statistics for an agent."""
    verify_auth(authorization)

    memory = _get_agent_memory()
    if memory is None:
        return {"agent_id": agent_id, "stats": {}, "message": "Agent memory not available"}

    stats = memory.get_stats(agent_id)
    return {"agent_id": agent_id, "stats": stats}


@router.post("/memory/{agent_id}", responses=AUTH_RESPONSES)
def store_agent_memory(
    agent_id: str,
    content: str = Query(...),
    memory_type: str = Query(default="fact"),
    importance: float = Query(default=0.5, ge=0.0, le=1.0),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Store a new memory for an agent."""
    verify_auth(authorization)

    memory = _get_agent_memory()
    if memory is None:
        raise not_found("Agent memory", "unavailable")

    memory_id = memory.store(
        agent_id=agent_id,
        content=content,
        memory_type=memory_type,
        importance=importance,
    )
    return {"agent_id": agent_id, "memory_id": memory_id, "stored": True}


@router.delete("/memory/{agent_id}", responses=AUTH_RESPONSES)
def forget_agent_memory(
    agent_id: str,
    memory_id: int | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Remove memories for an agent."""
    verify_auth(authorization)

    memory = _get_agent_memory()
    if memory is None:
        raise not_found("Agent memory", "unavailable")

    removed = memory.forget(
        agent_id=agent_id,
        memory_id=memory_id,
        memory_type=memory_type,
    )
    return {"agent_id": agent_id, "removed": removed}


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


@router.post("/run", responses=AUTH_RESPONSES)
async def run_agent_task(
    agent: str = Query(..., description="Agent role (builder, tester, etc.)"),
    task: str = Query(..., description="Task description"),
    use_tools: bool = Query(default=True),
    provider: str = Query(default="ollama"),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Submit and execute an agent task. Returns the result."""
    verify_auth(authorization)

    runner = _get_task_runner()
    if runner is None:
        # Create an ad-hoc runner
        try:
            from animus_forge.agents.provider_wrapper import create_agent_provider
            from animus_forge.agents.task_runner import AgentTaskRunner

            agent_provider = create_agent_provider(provider)
            tool_registry = None
            if use_tools:
                try:
                    from animus_forge.tools.registry import ForgeToolRegistry

                    tool_registry = ForgeToolRegistry()
                except Exception:
                    pass

            runner = AgentTaskRunner(
                provider=agent_provider,
                tool_registry=tool_registry,
                agent_memory=_get_agent_memory(),
            )
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    result = await runner.run(agent, task, use_tools=use_tools)
    return result.to_dict()


@router.get("/runs", responses=AUTH_RESPONSES)
def list_agent_runs(
    status: str | None = None,
    limit: int = Query(default=50, le=200),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """List recent agent runs from SubAgentManager."""
    verify_auth(authorization)

    sam = _get_subagent_manager()
    if sam is None:
        return {"runs": [], "total": 0, "message": "SubAgentManager not available"}

    try:
        from animus_forge.agents.subagent_manager import RunStatus

        status_filter = RunStatus(status) if status else None
        runs = sam.list_runs(status=status_filter)
        entries = [r.to_dict() for r in runs[:limit]]
        return {"runs": entries, "total": len(entries)}
    except (ValueError, KeyError) as e:
        return {"runs": [], "total": 0, "error": str(e)}


@router.get("/runs/{run_id}", responses=AUTH_RESPONSES)
def get_agent_run(
    run_id: str,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Get a specific agent run by ID."""
    verify_auth(authorization)

    sam = _get_subagent_manager()
    if sam is None:
        raise not_found("SubAgentManager", "unavailable")

    run = sam.get_run(run_id)
    if run is None:
        raise not_found("Agent run", run_id)

    return run.to_dict()


@router.post("/runs/{run_id}/cancel", responses=AUTH_RESPONSES)
async def cancel_agent_run(
    run_id: str,
    cascade: bool = Query(default=True),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Cancel a running agent."""
    verify_auth(authorization)

    sam = _get_subagent_manager()
    if sam is None:
        raise not_found("SubAgentManager", "unavailable")

    cancelled = await sam.cancel(run_id, cascade=cascade)
    return {"run_id": run_id, "cancelled": cancelled}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_process_registry():
    """Get process registry from api_state, if available."""
    return getattr(state, "process_registry", None)


def _get_agent_memory():
    """Get agent memory from api_state, if available."""
    return getattr(state, "agent_memory", None)


def _get_subagent_manager():
    """Get subagent manager from api_state, if available."""
    return getattr(state, "subagent_manager", None)


def _get_task_runner():
    """Get task runner from api_state, if available."""
    return getattr(state, "task_runner", None)


def _memory_to_dict(entry) -> dict[str, Any]:
    """Convert a MemoryEntry to a JSON-safe dict."""
    return {
        "id": entry.id,
        "agent_id": entry.agent_id,
        "workflow_id": entry.workflow_id,
        "memory_type": entry.memory_type,
        "content": entry.content,
        "metadata": entry.metadata,
        "importance": entry.importance,
        "created_at": str(entry.created_at) if entry.created_at else None,
        "accessed_at": str(entry.accessed_at) if entry.accessed_at else None,
        "access_count": entry.access_count,
    }
