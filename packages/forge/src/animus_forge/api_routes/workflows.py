"""Workflow CRUD, YAML workflows, and version endpoints."""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import APIRouter, Header, Request

from animus_forge import api_state as state
from animus_forge.api_errors import (
    AUTH_RESPONSES,
    CRUD_RESPONSES,
    WORKFLOW_RESPONSES,
    bad_request,
    internal_error,
    not_found,
)
from animus_forge.api_models import (
    ExecutionStartRequest,
    WorkflowExecuteRequest,
    WorkflowVersionRequest,
    YAMLWorkflowExecuteRequest,
)
from animus_forge.api_routes.auth import verify_auth
from animus_forge.orchestrator import Workflow
from animus_forge.workflow.loader import (
    list_workflows as list_yaml_workflows,
)
from animus_forge.workflow.loader import (
    load_workflow as load_yaml_workflow,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# JSON Workflow CRUD
# ---------------------------------------------------------------------------


@router.get("/workflows", responses=AUTH_RESPONSES)
def list_workflows(authorization: str | None = Header(None)):
    """List all workflows."""
    verify_auth(authorization)
    return state.workflow_engine.list_workflows()


@router.get("/workflows/{workflow_id}", responses=CRUD_RESPONSES)
def get_workflow(workflow_id: str, authorization: str | None = Header(None)):
    """Get a specific workflow."""
    verify_auth(authorization)
    workflow = state.workflow_engine.load_workflow(workflow_id)

    if not workflow:
        raise not_found("Workflow", workflow_id)

    return workflow


@router.post("/workflows", responses=CRUD_RESPONSES)
def create_workflow(workflow: Workflow, authorization: str | None = Header(None)):
    """Create a new workflow."""
    verify_auth(authorization)

    if state.workflow_engine.save_workflow(workflow):
        return {"status": "success", "workflow_id": workflow.id}

    raise internal_error("Failed to save workflow")


@router.post("/workflows/execute", responses=WORKFLOW_RESPONSES)
@state.limiter.limit("10/minute")
def execute_workflow(
    request_obj: Request,
    request: WorkflowExecuteRequest,
    authorization: str | None = Header(None),
):
    """Execute a workflow. Rate limited to 10 executions/minute per IP."""
    verify_auth(authorization)

    workflow = state.workflow_engine.load_workflow(request.workflow_id)
    if not workflow:
        raise not_found("Workflow", request.workflow_id)

    if request.variables:
        workflow.variables.update(request.variables)

    result = state.workflow_engine.execute_workflow(workflow)
    return result


# ---------------------------------------------------------------------------
# YAML Workflows
# ---------------------------------------------------------------------------


@router.get("/yaml-workflows", responses=AUTH_RESPONSES)
def list_yaml_workflow_definitions(authorization: str | None = Header(None)):
    """List all YAML workflow definitions."""
    verify_auth(authorization)
    try:
        workflows = list_yaml_workflows(str(state.YAML_WORKFLOWS_DIR))
        return {
            "workflows": [
                {
                    "id": w.get("name", "").lower().replace(" ", "-"),
                    "name": w.get("name"),
                    "description": w.get("description"),
                    "version": w.get("version"),
                    "path": w.get("path"),
                }
                for w in workflows
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list YAML workflows: {e}")
        return {"workflows": []}


@router.get("/yaml-workflows/{workflow_id}", responses=CRUD_RESPONSES)
def get_yaml_workflow_definition(workflow_id: str, authorization: str | None = Header(None)):
    """Get a specific YAML workflow definition."""
    verify_auth(authorization)

    workflow_id = re.sub(r"[^\w\-]", "", workflow_id)
    if not workflow_id:
        raise not_found("YAML Workflow", workflow_id)

    yaml_file = state.YAML_WORKFLOWS_DIR / f"{workflow_id}.yaml"
    yml_file = state.YAML_WORKFLOWS_DIR / f"{workflow_id}.yml"

    workflow_path = yaml_file if yaml_file.exists() else yml_file if yml_file.exists() else None

    if not workflow_path:
        raise not_found("YAML Workflow", workflow_id)

    try:
        workflow = load_yaml_workflow(str(workflow_path), str(state.YAML_WORKFLOWS_DIR))
        return {
            "id": workflow_id,
            "name": workflow.name,
            "description": getattr(workflow, "description", ""),
            "version": getattr(workflow, "version", "1.0"),
            "inputs": getattr(workflow, "inputs", {}),
            "outputs": getattr(workflow, "outputs", []),
            "steps": [
                {"id": step.id, "type": step.type, "params": step.params} for step in workflow.steps
            ],
        }
    except Exception as e:
        logger.error("Failed to load YAML workflow: %s", type(e).__name__)
        raise internal_error("Failed to load workflow")


@router.post("/yaml-workflows/execute", responses=WORKFLOW_RESPONSES)
@state.limiter.limit("10/minute")
def execute_yaml_workflow(
    request: Request,
    body: YAMLWorkflowExecuteRequest,
    authorization: str | None = Header(None),
):
    """Execute a YAML workflow. Rate limited to 10 executions/minute per IP."""
    verify_auth(authorization)

    yaml_file = state.YAML_WORKFLOWS_DIR / f"{body.workflow_id}.yaml"
    yml_file = state.YAML_WORKFLOWS_DIR / f"{body.workflow_id}.yml"

    workflow_path = yaml_file if yaml_file.exists() else yml_file if yml_file.exists() else None

    if not workflow_path:
        raise not_found("YAML Workflow", body.workflow_id)

    try:
        workflow = load_yaml_workflow(str(workflow_path), str(state.YAML_WORKFLOWS_DIR))
        inputs = body.inputs or {}
        result = state.yaml_workflow_executor.execute(workflow, inputs=inputs)

        return {
            "id": str(uuid.uuid4()),
            "workflow_id": body.workflow_id,
            "workflow_name": workflow.name,
            "status": result.status,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "total_duration_ms": result.total_duration_ms,
            "total_tokens": result.total_tokens,
            "outputs": result.outputs,
            "steps": [
                {
                    "step_id": step.step_id,
                    "status": step.status.value
                    if hasattr(step.status, "value")
                    else str(step.status),
                    "duration_ms": step.duration_ms,
                    "tokens_used": step.tokens_used,
                }
                for step in result.steps
            ],
            "error": result.error,
        }
    except Exception as e:
        logger.error("Failed to execute YAML workflow %s: %s", body.workflow_id, e)
        raise internal_error("Workflow execution failed")


# ---------------------------------------------------------------------------
# Workflow Versions
# ---------------------------------------------------------------------------


@router.get("/workflows/{workflow_name}/versions", responses=AUTH_RESPONSES)
def list_workflow_versions(
    workflow_name: str,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(None),
):
    """List all versions of a workflow."""
    verify_auth(authorization)
    versions = state.version_manager.list_versions(workflow_name, limit=limit, offset=offset)
    return [v.model_dump(mode="json") for v in versions]


@router.get("/workflows/{workflow_name}/versions/compare", responses=CRUD_RESPONSES)
def compare_workflow_versions(
    workflow_name: str,
    from_version: str,
    to_version: str,
    authorization: str | None = Header(None),
):
    """Compare two workflow versions."""
    verify_auth(authorization)

    try:
        diff = state.version_manager.compare_versions(workflow_name, from_version, to_version)
        return {
            "workflow_name": workflow_name,
            "from_version": diff.from_version,
            "to_version": diff.to_version,
            "has_changes": diff.has_changes,
            "added_lines": diff.added_lines,
            "removed_lines": diff.removed_lines,
            "changed_sections": diff.changed_sections,
            "unified_diff": diff.unified_diff,
        }
    except ValueError as e:
        raise bad_request(str(e))


@router.get("/workflows/{workflow_name}/versions/{version}", responses=CRUD_RESPONSES)
def get_workflow_version(
    workflow_name: str,
    version: str,
    authorization: str | None = Header(None),
):
    """Get a specific workflow version."""
    verify_auth(authorization)
    wv = state.version_manager.get_version(workflow_name, version)
    if not wv:
        raise not_found("Version", f"{workflow_name}@{version}")
    return wv.model_dump(mode="json")


@router.post("/workflows/{workflow_name}/versions", responses=CRUD_RESPONSES)
def save_workflow_version(
    workflow_name: str,
    request: WorkflowVersionRequest,
    authorization: str | None = Header(None),
):
    """Save a new workflow version."""
    verify_auth(authorization)

    try:
        wv = state.version_manager.save_version(
            workflow_name=workflow_name,
            content=request.content,
            version=request.version,
            description=request.description,
            author=request.author,
            activate=request.activate,
        )
        return {
            "status": "success",
            "workflow_name": wv.workflow_name,
            "version": wv.version,
            "is_active": wv.is_active,
        }
    except ValueError as e:
        raise bad_request(str(e))


@router.post("/workflows/{workflow_name}/versions/{version}/activate", responses=CRUD_RESPONSES)
def activate_workflow_version(
    workflow_name: str,
    version: str,
    authorization: str | None = Header(None),
):
    """Activate a specific workflow version."""
    verify_auth(authorization)

    try:
        state.version_manager.set_active(workflow_name, version)
        return {
            "status": "success",
            "workflow_name": workflow_name,
            "active_version": version,
        }
    except ValueError:
        raise not_found("Version", f"{workflow_name}@{version}")


@router.post("/workflows/{workflow_name}/rollback", responses=CRUD_RESPONSES)
def rollback_workflow(
    workflow_name: str,
    authorization: str | None = Header(None),
):
    """Rollback to the previous workflow version."""
    verify_auth(authorization)

    wv = state.version_manager.rollback(workflow_name)
    if not wv:
        raise bad_request(
            "No previous version to rollback to",
            {"workflow_name": workflow_name},
        )

    return {
        "status": "success",
        "workflow_name": workflow_name,
        "rolled_back_to": wv.version,
    }


@router.delete("/workflows/{workflow_name}/versions/{version}", responses=CRUD_RESPONSES)
def delete_workflow_version(
    workflow_name: str,
    version: str,
    authorization: str | None = Header(None),
):
    """Delete a workflow version (cannot delete active version)."""
    verify_auth(authorization)

    try:
        state.version_manager.delete_version(workflow_name, version)
        return {"status": "success"}
    except ValueError as e:
        raise bad_request(str(e))


@router.get("/workflow-versions", responses=AUTH_RESPONSES)
def list_versioned_workflows(authorization: str | None = Header(None)):
    """List all workflows with version information."""
    verify_auth(authorization)
    return state.version_manager.list_workflows()


# ---------------------------------------------------------------------------
# Start Workflow Execution (creates execution record)
# ---------------------------------------------------------------------------


@router.post("/workflows/{workflow_id}/execute", responses=WORKFLOW_RESPONSES)
@state.limiter.limit("10/minute")
def start_workflow_execution(
    request: Request,
    workflow_id: str,
    body: ExecutionStartRequest,
    authorization: str | None = Header(None),
):
    """Start a new workflow execution. Rate limited to 10 executions/minute per IP."""
    verify_auth(authorization)

    workflow = state.workflow_engine.load_workflow(workflow_id)
    workflow_name = workflow_id

    if workflow:
        workflow_name = getattr(workflow, "name", workflow_id)
    else:
        yaml_file = state.YAML_WORKFLOWS_DIR / f"{workflow_id}.yaml"
        yml_file = state.YAML_WORKFLOWS_DIR / f"{workflow_id}.yml"
        workflow_path = yaml_file if yaml_file.exists() else yml_file if yml_file.exists() else None
        if workflow_path:
            try:
                yaml_workflow = load_yaml_workflow(
                    str(workflow_path), str(state.YAML_WORKFLOWS_DIR)
                )
                workflow_name = yaml_workflow.name
            except Exception:
                logger.warning("Failed to load YAML workflow, using ID as name")
        else:
            raise not_found("Workflow", workflow_id)

    execution = state.execution_manager.create_execution(
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        variables=body.variables,
    )
    state.execution_manager.start_execution(execution.id)

    return {
        "execution_id": execution.id,
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "status": "running",
        "poll_url": f"/v1/executions/{execution.id}",
    }
