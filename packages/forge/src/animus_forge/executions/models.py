"""Pydantic models for workflow execution tracking."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class LogLevel(str, Enum):
    """Log level for execution logs."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ExecutionLog(BaseModel):
    """A single log entry for an execution."""

    id: int | None = None
    execution_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    level: LogLevel
    message: str
    step_id: str | None = None
    metadata: dict[str, Any] | None = None


class ExecutionMetrics(BaseModel):
    """Metrics for a workflow execution."""

    execution_id: str
    total_tokens: int = 0
    total_cost_cents: int = 0
    duration_ms: int = 0
    steps_completed: int = 0
    steps_failed: int = 0


class Execution(BaseModel):
    """A workflow execution instance."""

    id: str
    workflow_id: str
    workflow_name: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_step: str | None = None
    progress: int = 0
    checkpoint_id: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    logs: list[ExecutionLog] = Field(default_factory=list)
    metrics: ExecutionMetrics | None = None


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response container."""

    data: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def create(cls, data: list[T], total: int, page: int, page_size: int) -> PaginatedResponse[T]:
        """Create a paginated response with calculated total_pages."""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
