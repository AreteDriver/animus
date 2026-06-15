"""Workflow Executor with Contract Validation and State Persistence.

This module is a backward-compatibility shim. The WorkflowExecutor class
and supporting code have been refactored into focused submodules:

- executor_results.py: StepStatus, StepResult, ExecutionResult, StepHandler
- executor_clients.py: Client factories and circuit breaker setup
- executor_ai.py: AIHandlersMixin (Claude/OpenAI step handlers)
- executor_integrations.py: IntegrationHandlersMixin (GitHub, Notion, etc.)
- executor_patterns.py: DistributionPatternsMixin (fan-out, fan-in, map-reduce)
- executor_step.py: StepExecutionMixin (single step execution, fallbacks)
- executor_error.py: ErrorHandlerMixin (failure handling strategies)
- executor_parallel_exec.py: ParallelGroupMixin (auto-parallel, parallel groups)
- executor_core.py: WorkflowExecutor class (orchestration, sequential execution)
"""

from .executor_ai import AIHandlersMixin  # noqa: F401
from .executor_clients import (  # noqa: F401
    _get_claude_client,
    _get_openai_client,
    configure_circuit_breaker,
    get_circuit_breaker,
    reset_circuit_breakers,
)
from .executor_core import WorkflowExecutor  # noqa: F401
from .executor_error import ErrorHandlerMixin  # noqa: F401
from .executor_integrations import IntegrationHandlersMixin  # noqa: F401
from .executor_parallel_exec import ParallelGroupMixin  # noqa: F401
from .executor_patterns import DistributionPatternsMixin  # noqa: F401
from .executor_results import (  # noqa: F401
    ExecutionResult,
    StepHandler,
    StepResult,
    StepStatus,
)
from .executor_step import StepExecutionMixin  # noqa: F401
from .loader import StepConfig, WorkflowConfig  # noqa: F401

__all__ = [
    "WorkflowExecutor",
    "ExecutionResult",
    "StepHandler",
    "StepResult",
    "StepStatus",
    "StepConfig",
    "WorkflowConfig",
    "_get_claude_client",
    "_get_openai_client",
    "configure_circuit_breaker",
    "get_circuit_breaker",
    "reset_circuit_breakers",
    "IntegrationHandlersMixin",
    "AIHandlersMixin",
    "DistributionPatternsMixin",
    "StepExecutionMixin",
    "ErrorHandlerMixin",
    "ParallelGroupMixin",
]
