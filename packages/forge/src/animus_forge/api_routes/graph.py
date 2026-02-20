"""Graph workflow execution endpoints."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, CRUD_RESPONSES, bad_request, not_found
from animus_forge.api_routes.auth import verify_auth

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class GraphNodeInput(BaseModel):
    """A node in the workflow graph."""

    id: str
    type: str = "agent"
    data: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})


class GraphEdgeInput(BaseModel):
    """An edge connecting two nodes."""

    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None
    label: str | None = None


class GraphExecuteRequest(BaseModel):
    """Request to execute a graph workflow."""

    id: str | None = None
    name: str = "Untitled Workflow"
    nodes: list[GraphNodeInput]
    edges: list[GraphEdgeInput] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    version: str = "1.0"


class GraphExecuteVariablesRequest(BaseModel):
    """Request body with graph and optional variables override."""

    graph: GraphExecuteRequest
    variables: dict[str, Any] | None = None


class NodeResultResponse(BaseModel):
    """Result of a single node execution."""

    node_id: str
    status: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0
    tokens_used: int = 0


class GraphExecutionResponse(BaseModel):
    """Response from graph execution."""

    execution_id: str
    workflow_id: str
    status: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    node_results: dict[str, NodeResultResponse] = Field(default_factory=dict)
    total_duration_ms: int = 0
    total_tokens: int = 0
    error: str | None = None


class GraphValidationIssue(BaseModel):
    """A single validation issue found in a graph."""

    severity: str  # "error" or "warning"
    message: str
    node_id: str | None = None


class GraphValidationResponse(BaseModel):
    """Response from graph validation."""

    valid: bool
    issues: list[GraphValidationIssue] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


# ---------------------------------------------------------------------------
# In-memory async execution store (keyed by execution_id)
# ---------------------------------------------------------------------------

_async_executions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_workflow_graph(data: GraphExecuteRequest):
    """Convert API request model into a WorkflowGraph dataclass."""
    from animus_forge.workflow.graph_models import WorkflowGraph

    graph_dict = {
        "id": data.id or str(uuid.uuid4()),
        "name": data.name,
        "nodes": [n.model_dump() for n in data.nodes],
        "edges": [e.model_dump() for e in data.edges],
        "variables": data.variables,
        "description": data.description,
        "version": data.version,
    }
    return WorkflowGraph.from_dict(graph_dict)


def _execution_result_to_response(result) -> dict:
    """Convert an ExecutionResult dataclass to a JSON-serialisable dict."""
    node_results = {}
    for nid, nr in result.node_results.items():
        node_results[nid] = {
            "node_id": nr.node_id,
            "status": nr.status.value if hasattr(nr.status, "value") else str(nr.status),
            "outputs": nr.outputs,
            "error": nr.error,
            "duration_ms": nr.duration_ms,
            "tokens_used": nr.tokens_used,
        }

    return {
        "execution_id": result.execution_id,
        "workflow_id": result.workflow_id,
        "status": result.status,
        "outputs": result.outputs,
        "node_results": node_results,
        "total_duration_ms": result.total_duration_ms,
        "total_tokens": result.total_tokens,
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/graph/execute", responses=CRUD_RESPONSES)
async def execute_graph(
    body: GraphExecuteVariablesRequest,
    authorization: str | None = Header(None),
):
    """Execute a graph workflow synchronously and return results."""
    verify_auth(authorization)

    try:
        graph = _build_workflow_graph(body.graph)
    except Exception as e:
        raise bad_request(f"Invalid graph: {e}")

    try:
        from animus_forge.workflow.graph_executor import ReactFlowExecutor

        executor = ReactFlowExecutor(
            execution_manager=state.execution_manager,
        )
        result = await executor.execute_async(graph, body.variables)
    except Exception as e:
        logger.exception("Graph execution failed: %s", e)
        raise bad_request(f"Execution failed: {e}")

    return _execution_result_to_response(result)


@router.post("/graph/execute/async", responses=CRUD_RESPONSES)
async def execute_graph_async(
    body: GraphExecuteVariablesRequest,
    authorization: str | None = Header(None),
):
    """Start an asynchronous graph execution.

    Returns immediately with an execution_id that can be polled
    via ``GET /graph/executions/{id}``.
    """
    verify_auth(authorization)

    try:
        graph = _build_workflow_graph(body.graph)
    except Exception as e:
        raise bad_request(f"Invalid graph: {e}")

    import asyncio

    execution_id = f"graph-{uuid.uuid4().hex[:12]}"

    # Store initial status
    _async_executions[execution_id] = {
        "execution_id": execution_id,
        "workflow_id": graph.id,
        "status": "running",
        "outputs": {},
        "node_results": {},
        "total_duration_ms": 0,
        "total_tokens": 0,
        "error": None,
    }

    async def _run():
        try:
            from animus_forge.workflow.graph_executor import ReactFlowExecutor

            executor = ReactFlowExecutor(
                execution_manager=state.execution_manager,
            )
            result = await executor.execute_async(graph, body.variables, execution_id)
            _async_executions[execution_id] = _execution_result_to_response(result)
        except Exception as exc:
            _async_executions[execution_id]["status"] = "failed"
            _async_executions[execution_id]["error"] = str(exc)

    asyncio.ensure_future(_run())

    return {
        "execution_id": execution_id,
        "status": "running",
        "poll_url": f"/v1/graph/executions/{execution_id}",
    }


@router.get("/graph/executions/{execution_id}", responses=CRUD_RESPONSES)
def get_graph_execution(
    execution_id: str,
    authorization: str | None = Header(None),
):
    """Get the status and results of a graph execution."""
    verify_auth(authorization)

    entry = _async_executions.get(execution_id)
    if not entry:
        raise not_found("Graph Execution", execution_id)
    return entry


@router.post("/graph/executions/{execution_id}/pause", responses=CRUD_RESPONSES)
def pause_graph_execution(
    execution_id: str,
    authorization: str | None = Header(None),
):
    """Pause a running graph execution."""
    verify_auth(authorization)

    entry = _async_executions.get(execution_id)
    if not entry:
        raise not_found("Graph Execution", execution_id)

    if entry["status"] != "running":
        raise bad_request(
            f"Cannot pause execution in '{entry['status']}' state",
            {"execution_id": execution_id, "current_status": entry["status"]},
        )

    try:
        from animus_forge.workflow.graph_executor import ReactFlowExecutor

        executor = ReactFlowExecutor(
            execution_manager=state.execution_manager,
        )
        paused = executor.pause(execution_id)
        if paused:
            entry["status"] = "paused"
            return {"status": "success", "message": "Execution paused"}
        raise bad_request("Failed to pause execution (no execution manager)")
    except bad_request.__class__:
        raise
    except Exception as e:
        raise bad_request(f"Failed to pause: {e}")


@router.post("/graph/executions/{execution_id}/resume", responses=CRUD_RESPONSES)
async def resume_graph_execution(
    execution_id: str,
    body: GraphExecuteRequest,
    authorization: str | None = Header(None),
):
    """Resume a paused graph execution.

    Requires the original graph definition to continue execution.
    """
    verify_auth(authorization)

    entry = _async_executions.get(execution_id)
    if not entry:
        raise not_found("Graph Execution", execution_id)

    if entry["status"] != "paused":
        raise bad_request(
            f"Cannot resume execution in '{entry['status']}' state",
            {"execution_id": execution_id, "current_status": entry["status"]},
        )

    try:
        graph = _build_workflow_graph(body)
    except Exception as e:
        raise bad_request(f"Invalid graph: {e}")

    try:
        from animus_forge.workflow.graph_executor import ReactFlowExecutor

        executor = ReactFlowExecutor(
            execution_manager=state.execution_manager,
        )
        entry["status"] = "running"
        result = executor.resume(execution_id, graph)
        _async_executions[execution_id] = _execution_result_to_response(result)
        return _async_executions[execution_id]
    except Exception as e:
        entry["status"] = "failed"
        entry["error"] = str(e)
        raise bad_request(f"Resume failed: {e}")


@router.post("/graph/validate", responses=AUTH_RESPONSES)
def validate_graph(
    body: GraphExecuteRequest,
    authorization: str | None = Header(None),
):
    """Validate a graph for structural issues (cycles, missing connections)."""
    verify_auth(authorization)

    issues: list[dict] = []

    try:
        graph = _build_workflow_graph(body)
    except Exception as e:
        return GraphValidationResponse(
            valid=False,
            issues=[
                GraphValidationIssue(
                    severity="error",
                    message=f"Failed to parse graph: {e}",
                )
            ],
        ).model_dump()

    from animus_forge.workflow.graph_walker import GraphWalker

    walker = GraphWalker(graph)

    # Check for cycles (non-loop cycles are errors)
    cycles = walker.detect_cycles()
    for cycle in cycles:
        cycle_nodes = [graph.get_node(nid) for nid in cycle]
        if not any(n and n.type == "loop" for n in cycle_nodes):
            issues.append(
                {
                    "severity": "error",
                    "message": f"Non-loop cycle detected: {' -> '.join(cycle)}",
                }
            )

    # Check for disconnected nodes (no incoming or outgoing edges)
    all_connected = {e.source for e in graph.edges} | {e.target for e in graph.edges}
    for node in graph.nodes:
        if node.id not in all_connected and len(graph.nodes) > 1:
            issues.append(
                {
                    "severity": "warning",
                    "message": f"Node '{node.id}' is disconnected",
                    "node_id": node.id,
                }
            )

    # Check for edges referencing nonexistent nodes
    node_ids = {n.id for n in graph.nodes}
    for edge in graph.edges:
        if edge.source not in node_ids:
            issues.append(
                {
                    "severity": "error",
                    "message": f"Edge '{edge.id}' references missing source node '{edge.source}'",
                }
            )
        if edge.target not in node_ids:
            issues.append(
                {
                    "severity": "error",
                    "message": f"Edge '{edge.id}' references missing target node '{edge.target}'",
                }
            )

    # Check for missing start/end nodes
    start_nodes = [n for n in graph.nodes if n.type == "start"]
    end_nodes = [n for n in graph.nodes if n.type == "end"]
    if not start_nodes and graph.nodes:
        issues.append(
            {
                "severity": "warning",
                "message": "No explicit start node found",
            }
        )
    if not end_nodes and graph.nodes:
        issues.append(
            {
                "severity": "warning",
                "message": "No explicit end node found",
            }
        )

    has_errors = any(i.get("severity") == "error" for i in issues)

    return GraphValidationResponse(
        valid=not has_errors,
        issues=[GraphValidationIssue(**i) for i in issues],
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    ).model_dump()
