"""Pydantic request/response models for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    """Login request."""

    user_id: str = Field(..., max_length=128, pattern=r"^[\w@.\-]+$")
    password: str


class LoginResponse(BaseModel):
    """Login response."""

    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow."""

    workflow_id: str
    variables: dict | None = None


class ExecutionStartRequest(BaseModel):
    """Request to start a workflow execution."""

    variables: dict | None = None


class ApprovalResumeRequest(BaseModel):
    """Request to resume an execution with an approval token."""

    token: str
    approve: bool = True
    approved_by: str | None = None
    reason: str | None = None


class YAMLWorkflowExecuteRequest(BaseModel):
    """Request to execute a YAML workflow."""

    workflow_id: str = Field(..., pattern=r"^[\w\-]+$")
    inputs: dict | None = None


# ---------------------------------------------------------------------------
# Workflow versioning
# ---------------------------------------------------------------------------


class WorkflowVersionRequest(BaseModel):
    """Request to save a workflow version."""

    content: str
    version: str | None = None
    description: str | None = None
    author: str | None = None
    activate: bool = True


class VersionCompareRequest(BaseModel):
    """Request to compare two versions."""

    from_version: str
    to_version: str


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class PreferencesUpdateRequest(BaseModel):
    """Request to update user preferences."""

    theme: str | None = None
    compact_view: bool | None = None
    show_costs: bool | None = None
    default_page_size: int | None = None
    notifications: dict | None = None


class APIKeyCreateRequest(BaseModel):
    """Request to create/update an API key."""

    provider: str
    key: str


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class BudgetCreateRequest(BaseModel):
    """Request to create a budget."""

    name: str
    total_amount: float
    period: str = "monthly"
    agent_id: str | None = None


class BudgetUpdateRequest(BaseModel):
    """Request to update a budget."""

    name: str | None = None
    total_amount: float | None = None
    used_amount: float | None = None
    period: str | None = None
    agent_id: str | None = None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class DashboardStats(BaseModel):
    """Dashboard statistics response."""

    totalWorkflows: int
    activeExecutions: int
    completedToday: int
    failedToday: int
    totalTokensToday: int
    totalCostToday: float


class RecentExecution(BaseModel):
    """Recent execution summary for dashboard."""

    id: str
    name: str
    status: str
    time: str


class DailyUsage(BaseModel):
    """Daily usage data point."""

    date: str
    tokens: int
    cost: float


class AgentUsage(BaseModel):
    """Per-agent usage data point."""

    agent: str
    tokens: int


class BudgetStatus(BaseModel):
    """Budget status for an agent."""

    agent: str
    used: float
    limit: float


class DashboardBudget(BaseModel):
    """Dashboard budget summary."""

    totalBudget: float
    totalUsed: float
    percentUsed: float
    byAgent: list[BudgetStatus]
    alert: str | None = None


class AgentDefinitionResponse(BaseModel):
    """Response model for agent definition."""

    id: str
    name: str
    description: str
    capabilities: list[str]
    icon: str
    color: str
