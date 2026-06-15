"""Gorgon Error Hierarchy.

Structured exception types for the orchestration framework.
"""

from __future__ import annotations


class GorgonError(Exception):
    """Base error for all Gorgon exceptions."""

    code = "GORGON_ERROR"

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# Agent Errors
class AgentError(GorgonError):
    """Base error for agent execution failures."""

    code = "AGENT_ERROR"


class AgentTimeoutError(AgentError):
    """Agent execution exceeded timeout."""

    code = "TIMEOUT"


class TokenLimitError(AgentError):
    """Agent exceeded token limit."""

    code = "TOKEN_LIMIT"

    def __init__(self, message: str, requested: int = 0, available: int = 0):
        super().__init__(message, {"requested": requested, "available": available})
        self.requested = requested
        self.available = available


class ContractViolationError(AgentError):
    """Agent input/output violated contract."""

    code = "CONTRACT_VIOLATION"

    def __init__(self, message: str, role: str = None, field: str = None):
        super().__init__(message, {"role": role, "field": field})
        self.role = role
        self.field = field


class APIError(AgentError):
    """External API call failed."""

    code = "API_ERROR"

    def __init__(self, message: str, provider: str = None, status_code: int = None):
        super().__init__(message, {"provider": provider, "status_code": status_code})
        self.provider = provider
        self.status_code = status_code


class ValidationError(AgentError):
    """Data validation failed."""

    code = "VALIDATION"


# Budget Errors
class BudgetExceededError(GorgonError):
    """Token budget has been exceeded."""

    code = "BUDGET_EXCEEDED"

    def __init__(self, message: str, budget: int = 0, used: int = 0, agent: str = None):
        super().__init__(message, {"budget": budget, "used": used, "agent": agent})
        self.budget = budget
        self.used = used
        self.agent = agent


# Workflow Errors
class WorkflowError(GorgonError):
    """Base error for workflow execution failures."""

    code = "WORKFLOW_ERROR"


class StageFailedError(WorkflowError):
    """A workflow stage failed to complete."""

    code = "STAGE_FAILED"

    def __init__(self, message: str, stage: str = None, cause: Exception = None):
        details = {"stage": stage}
        if cause:
            details["cause"] = str(cause)
        super().__init__(message, details)
        self.stage = stage
        self.cause = cause


class MaxRetriesError(WorkflowError):
    """Maximum retry attempts exceeded."""

    code = "MAX_RETRIES"

    def __init__(self, message: str, stage: str = None, attempts: int = 0):
        super().__init__(message, {"stage": stage, "attempts": attempts})
        self.stage = stage
        self.attempts = attempts


class WorkflowNotFoundError(WorkflowError):
    """Workflow definition not found."""

    code = "WORKFLOW_NOT_FOUND"


class CheckpointError(WorkflowError):
    """Error during checkpoint/resume operations."""

    code = "CHECKPOINT_ERROR"


# State Errors
class StateError(GorgonError):
    """Error in state management."""

    code = "STATE_ERROR"


class ResumeError(StateError):
    """Cannot resume workflow from checkpoint."""

    code = "RESUME_ERROR"
