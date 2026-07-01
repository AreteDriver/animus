"""Checkpoint Manager for automatic workflow state management."""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .backends import DatabaseBackend
from .persistence import StatePersistence, WorkflowStatus


@dataclass
class StageContext:
    """Context for a workflow stage execution."""

    workflow_id: str
    stage: str
    started_at: float
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    tokens_used: int = 0
    status: str = "running"
    error: str | None = None


class CheckpointManager:
    """Manages workflow checkpoints with automatic state capture.

    Provides context managers for easy checkpoint creation during
    workflow execution.
    """

    def __init__(
        self,
        persistence: StatePersistence = None,
        backend: DatabaseBackend = None,
        db_path: str = None,
    ):
        """Initialize checkpoint manager.

        Args:
            persistence: Optional StatePersistence instance
            backend: Optional DatabaseBackend (used if persistence not provided)
            db_path: Path to database (used if neither persistence nor backend provided)
        """
        if persistence:
            self.persistence = persistence
        elif backend:
            self.persistence = StatePersistence(backend=backend)
        else:
            self.persistence = StatePersistence(db_path=db_path or "gorgon-state.db")

        self._current_workflow: str | None = None
        self._current_stage: StageContext | None = None

    def start_workflow(
        self,
        name: str,
        config: dict = None,
        workflow_id: str = None,
    ) -> str:
        """Start a new workflow.

        Args:
            name: Workflow name
            config: Optional configuration
            workflow_id: Optional custom ID (generated if not provided)

        Returns:
            Workflow ID
        """
        if workflow_id is None:
            workflow_id = f"wf-{uuid.uuid4().hex[:12]}"

        self.persistence.create_workflow(workflow_id, name, config)
        self.persistence.update_status(workflow_id, WorkflowStatus.RUNNING)
        self._current_workflow = workflow_id

        return workflow_id

    def complete_workflow(self, workflow_id: str = None) -> None:
        """Mark workflow as completed.

        Args:
            workflow_id: Workflow ID (uses current if not provided)
        """
        wf_id = workflow_id or self._current_workflow
        if wf_id:
            self.persistence.mark_complete(wf_id)
            if wf_id == self._current_workflow:
                self._current_workflow = None

    def fail_workflow(self, error: str, workflow_id: str = None) -> None:
        """Mark workflow as failed.

        Args:
            error: Error message
            workflow_id: Workflow ID (uses current if not provided)
        """
        wf_id = workflow_id or self._current_workflow
        if wf_id:
            self.persistence.mark_failed(wf_id, error)
            if wf_id == self._current_workflow:
                self._current_workflow = None

    @contextmanager
    def workflow(
        self,
        name: str,
        config: dict = None,
    ) -> Generator[str, None, None]:
        """Context manager for workflow execution.

        Automatically handles workflow creation and completion/failure.

        Usage:
            with checkpoint_manager.workflow("my_workflow") as wf_id:
                # Execute workflow stages
                pass
        """
        workflow_id = self.start_workflow(name, config)
        try:
            yield workflow_id
            self.complete_workflow(workflow_id)
        except Exception as e:
            self.fail_workflow(str(e), workflow_id)
            raise

    @contextmanager
    def stage(
        self,
        stage_name: str,
        input_data: dict = None,
        workflow_id: str = None,
    ) -> Generator[StageContext, None, None]:
        """Context manager for stage execution with automatic checkpointing.

        Automatically captures timing and checkpoints on success or failure.

        Usage:
            with checkpoint_manager.stage("building", input_data) as ctx:
                # Execute stage
                ctx.output_data = {"result": "..."}
                ctx.tokens_used = 100
        """
        wf_id = workflow_id or self._current_workflow
        if not wf_id:
            raise ValueError("No active workflow. Call start_workflow() first.")

        ctx = StageContext(
            workflow_id=wf_id,
            stage=stage_name,
            started_at=time.time(),
            input_data=input_data or {},
        )
        self._current_stage = ctx

        try:
            yield ctx
            ctx.status = "success"
        except Exception as e:
            ctx.status = "failed"
            ctx.error = str(e)
            raise
        finally:
            duration_ms = int((time.time() - ctx.started_at) * 1000)

            self.persistence.checkpoint(
                workflow_id=wf_id,
                stage=stage_name,
                status=ctx.status,
                input_data=ctx.input_data,
                output_data=ctx.output_data,
                tokens_used=ctx.tokens_used,
                duration_ms=duration_ms,
            )
            self._current_stage = None

    def checkpoint_now(
        self,
        stage: str,
        status: str,
        input_data: dict = None,
        output_data: dict = None,
        tokens_used: int = 0,
        duration_ms: int = 0,
        workflow_id: str = None,
    ) -> int:
        """Create a checkpoint immediately.

        For cases where the context manager isn't suitable.

        Args:
            stage: Stage name
            status: Stage status
            input_data: Input data
            output_data: Output data
            tokens_used: Tokens consumed
            duration_ms: Duration in milliseconds
            workflow_id: Workflow ID (uses current if not provided)

        Returns:
            Checkpoint ID
        """
        wf_id = workflow_id or self._current_workflow
        if not wf_id:
            raise ValueError("No active workflow")

        return self.persistence.checkpoint(
            workflow_id=wf_id,
            stage=stage,
            status=status,
            input_data=input_data,
            output_data=output_data,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
        )

    def resume(self, workflow_id: str) -> dict:
        """Resume a workflow from its last checkpoint.

        Args:
            workflow_id: Workflow to resume

        Returns:
            Last checkpoint data with workflow info
        """
        workflow = self.persistence.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        checkpoint = self.persistence.resume_from_checkpoint(workflow_id)
        self._current_workflow = workflow_id

        return {
            "workflow": workflow,
            "checkpoint": checkpoint,
            "resume_from_stage": checkpoint["stage"],
        }

    def get_progress(self, workflow_id: str = None) -> dict:
        """Get workflow progress information.

        Args:
            workflow_id: Workflow ID (uses current if not provided)

        Returns:
            Progress information including completed stages
        """
        wf_id = workflow_id or self._current_workflow
        if not wf_id:
            raise ValueError("No workflow specified")

        workflow = self.persistence.get_workflow(wf_id)
        checkpoints = self.persistence.get_all_checkpoints(wf_id)

        total_tokens = sum(cp["tokens_used"] for cp in checkpoints)
        total_duration = sum(cp["duration_ms"] for cp in checkpoints)
        stages_completed = [cp["stage"] for cp in checkpoints if cp["status"] == "success"]
        stages_failed = [cp["stage"] for cp in checkpoints if cp["status"] == "failed"]

        return {
            "workflow_id": wf_id,
            "workflow_name": workflow["name"] if workflow else None,
            "status": workflow["status"] if workflow else None,
            "current_stage": workflow["current_stage"] if workflow else None,
            "stages_completed": stages_completed,
            "stages_failed": stages_failed,
            "total_checkpoints": len(checkpoints),
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration,
        }

    @property
    def current_workflow_id(self) -> str | None:
        """Get the current workflow ID."""
        return self._current_workflow

    @property
    def current_stage(self) -> StageContext | None:
        """Get the current stage context."""
        return self._current_stage
