"""Adapter for migrating from WorkflowEngine to WorkflowExecutor.

This module provides backward compatibility by converting the old Workflow
objects to the new WorkflowConfig format used by WorkflowExecutor.
"""

from __future__ import annotations

import json
import logging

from animus_forge.config import get_settings
from animus_forge.workflow import (
    ExecutionResult,
    StepConfig,
    WorkflowConfig,
    WorkflowExecutor,
)

from .workflow_engine import StepType, Workflow, WorkflowResult, WorkflowStep

logger = logging.getLogger(__name__)


def convert_step_type(step_type: StepType) -> str:
    """Convert old StepType enum to new step type string."""
    mapping = {
        StepType.OPENAI: "openai",
        StepType.CLAUDE_CODE: "claude_code",
        StepType.GITHUB: "shell",  # GitHub actions mapped to shell
        StepType.NOTION: "shell",  # Notion actions mapped to shell
        StepType.GMAIL: "shell",  # Gmail actions mapped to shell
        StepType.TRANSFORM: "shell",  # Transform mapped to shell
    }
    return mapping.get(step_type, "shell")


def convert_workflow_step(step: WorkflowStep) -> dict:
    """Convert a WorkflowStep to StepConfig dict format."""
    step_type = convert_step_type(step.type)

    # Build params based on step type
    params = step.params.copy()

    if step_type == "openai":
        # Map action to prompt if not present
        if "prompt" not in params and step.action:
            params["prompt"] = step.action
    elif step_type == "claude_code":
        # Map action to prompt if not present
        if "prompt" not in params and step.action:
            params["prompt"] = step.action
        if "role" not in params:
            params["role"] = "builder"
    elif step_type == "shell":
        # For shell steps, use action as command
        if "command" not in params:
            params["command"] = f"echo 'Step: {step.action}'"

    return {
        "id": step.id,
        "type": step_type,
        "params": params,
    }


def convert_workflow(workflow: Workflow) -> WorkflowConfig:
    """Convert old Workflow to new WorkflowConfig format.

    Args:
        workflow: Old-style Workflow object

    Returns:
        New-style WorkflowConfig object
    """
    steps = [StepConfig.from_dict(convert_workflow_step(step)) for step in workflow.steps]

    return WorkflowConfig(
        name=workflow.name,
        description=workflow.description,
        version="1.0.0",  # Default version
        steps=steps,
        inputs={
            name: {"type": "string", "default": value} for name, value in workflow.variables.items()
        },
    )


def convert_execution_result(result: ExecutionResult, workflow_id: str) -> WorkflowResult:
    """Convert ExecutionResult to old WorkflowResult format.

    Args:
        result: New-style ExecutionResult
        workflow_id: Original workflow ID

    Returns:
        Old-style WorkflowResult for backward compatibility
    """
    return WorkflowResult(
        workflow_id=workflow_id,
        status=result.status,
        started_at=result.started_at,
        completed_at=result.completed_at,
        steps_executed=[s.step_id for s in result.steps],
        outputs=result.outputs,
        errors=[result.error] if result.error else [],
    )


class WorkflowEngineAdapter:
    """Adapter that provides WorkflowEngine interface using WorkflowExecutor.

    This allows existing code using WorkflowEngine to migrate incrementally
    to WorkflowExecutor while maintaining backward compatibility.

    Usage:
        # Old code
        engine = WorkflowEngine()
        result = engine.execute_workflow(workflow)

        # New code with adapter (same interface)
        engine = WorkflowEngineAdapter()
        result = engine.execute_workflow(workflow)

        # Or migrate directly to WorkflowExecutor
        executor = WorkflowExecutor()
        config = convert_workflow(workflow)
        result = executor.execute(config)
    """

    def __init__(
        self,
        checkpoint_manager=None,
        contract_validator=None,
        budget_manager=None,
        dry_run: bool = False,
        execution_manager=None,
    ):
        """Initialize adapter with WorkflowExecutor.

        Args:
            checkpoint_manager: Optional checkpoint manager
            contract_validator: Optional contract validator
            budget_manager: Optional budget manager
            dry_run: If True, use mock responses
            execution_manager: Optional ExecutionManager for streaming logs
        """
        self._executor = WorkflowExecutor(
            checkpoint_manager=checkpoint_manager,
            contract_validator=contract_validator,
            budget_manager=budget_manager,
            dry_run=dry_run,
            execution_manager=execution_manager,
        )
        logger.info("Using WorkflowEngineAdapter - consider migrating to WorkflowExecutor directly")

    def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
        """Execute a workflow using the new WorkflowExecutor.

        Args:
            workflow: Old-style Workflow object

        Returns:
            Old-style WorkflowResult for backward compatibility
        """
        # Convert to new format
        config = convert_workflow(workflow)

        # Execute with new executor
        result = self._executor.execute(config, inputs=workflow.variables)

        # Convert result back to old format
        return convert_execution_result(result, workflow.id)

    async def execute_workflow_async(self, workflow: Workflow) -> WorkflowResult:
        """Execute a workflow asynchronously.

        Args:
            workflow: Old-style Workflow object

        Returns:
            Old-style WorkflowResult for backward compatibility
        """
        config = convert_workflow(workflow)
        result = await self._executor.execute_async(config, inputs=workflow.variables)
        return convert_execution_result(result, workflow.id)

    @property
    def settings(self):
        """Get settings for compatibility with code expecting settings attribute."""
        return get_settings()

    def save_workflow(self, workflow: Workflow) -> bool:
        """Save a workflow definition to JSON file.

        Args:
            workflow: Workflow to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            settings = get_settings()
            settings.workflows_dir.mkdir(parents=True, exist_ok=True)
            file_path = settings.workflows_dir / f"{workflow.id}.json"
            with open(file_path, "w") as f:
                json.dump(workflow.model_dump(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save workflow {workflow.id}: {e}")
            return False

    def load_workflow(self, workflow_id: str) -> Workflow | None:
        """Load a workflow definition from JSON file.

        Args:
            workflow_id: ID of workflow to load

        Returns:
            Workflow if found, None otherwise
        """
        try:
            settings = get_settings()
            file_path = settings.workflows_dir / f"{workflow_id}.json"
            if not file_path.exists():
                return None
            with open(file_path) as f:
                data = json.load(f)
            return Workflow(**data)
        except Exception as e:
            logger.error(f"Failed to load workflow {workflow_id}: {e}")
            return None

    def list_workflows(self) -> list[dict]:
        """List all available workflows.

        Returns:
            List of workflow metadata dicts (id, name, description)
        """
        workflows = []
        settings = get_settings()
        if not settings.workflows_dir.exists():
            return workflows

        for file_path in settings.workflows_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                workflows.append(
                    {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read workflow file {file_path}: {e}")
                continue
        return workflows
