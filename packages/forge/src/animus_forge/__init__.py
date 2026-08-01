"""AI Workflow Orchestrator - A unified automation layer for AI-powered workflows."""

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("animus-forge")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+dev"

from .auth import TokenAuth, create_access_token, verify_token
from .config import Settings, get_settings
from .jobs import (
    Job,
    JobManager,
    JobStatus,
)
from .orchestrator import Workflow, WorkflowEngineAdapter, WorkflowResult, WorkflowStep
from .prompts import PromptTemplate, PromptTemplateManager
from .scheduler import (
    CronConfig,
    IntervalConfig,
    ScheduleManager,
    ScheduleStatus,
    ScheduleType,
    WorkflowSchedule,
)
from .webhooks import (
    PayloadMapping,
    Webhook,
    WebhookManager,
    WebhookStatus,
)

__all__ = [
    "Settings",
    "get_settings",
    "WorkflowEngineAdapter",
    "Workflow",
    "WorkflowStep",
    "WorkflowResult",
    "PromptTemplateManager",
    "PromptTemplate",
    "TokenAuth",
    "create_access_token",
    "verify_token",
    "ScheduleManager",
    "WorkflowSchedule",
    "ScheduleType",
    "ScheduleStatus",
    "CronConfig",
    "IntervalConfig",
    "WebhookManager",
    "Webhook",
    "WebhookStatus",
    "PayloadMapping",
    "JobManager",
    "Job",
    "JobStatus",
]
