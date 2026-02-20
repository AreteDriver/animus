"""Workflow models for orchestration.

These models define the structure of workflows, steps, and results.
For execution, use WorkflowEngineAdapter or WorkflowExecutor.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types of workflow steps."""

    OPENAI = "openai"
    GITHUB = "github"
    NOTION = "notion"
    GMAIL = "gmail"
    TRANSFORM = "transform"
    CLAUDE_CODE = "claude_code"


class WorkflowStep(BaseModel):
    """A single step in a workflow."""

    id: str = Field(..., description="Step identifier")
    type: StepType = Field(..., description="Step type")
    action: str = Field(..., description="Action to perform")
    params: dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    next_step: str | None = Field(None, description="Next step ID")


class Workflow(BaseModel):
    """A workflow definition."""

    id: str = Field(..., description="Workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: list[WorkflowStep] = Field(default_factory=list, description="Workflow steps")
    variables: dict[str, Any] = Field(default_factory=dict, description="Workflow variables")


class WorkflowResult(BaseModel):
    """Result of a workflow execution."""

    workflow_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    steps_executed: list[str] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
