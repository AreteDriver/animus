"""SkillBridge — Forge subsystem that executes ai-skills repo entries
as workflow steps via the in-process Provider stack.

Public surface:
    SkillResolver       — name + tier → ResolvedSkill
    SkillExecutor       — ResolvedSkill + inputs → SkillCallResult
    build_skill_step_handler — wires SkillBridge into the workflow
                              executor's `_handlers["skill"]` dispatch
"""

from __future__ import annotations

from .errors import (
    SkillApprovalDeniedError,
    SkillApprovalTimeoutError,
    SkillBridgeError,
    SkillBudgetError,
    SkillConfigError,
    SkillInputError,
    SkillNestingDepthError,
    SkillOutputDriftError,
    SkillResolutionError,
)
from .executor import (
    APPROVAL_DEFAULT_TIMEOUT_S,
    APPROVAL_POLL_INTERVAL_S,
    DEFAULT_MAX_NESTING_DEPTH,
    SkillCallResult,
    SkillExecutor,
    StepContext,
)
from .resolver import DEFAULT_AI_SKILLS_REPO, SkillResolver
from .step_handler import STEP_TYPE, build_skill_step_handler
from .types import (
    TIER_OUTPUT_CEILING,
    AgentSchema,
    RegistryEntry,
    ResolvedSkill,
    RiskLevel,
    Tier,
)

__all__ = [
    "APPROVAL_DEFAULT_TIMEOUT_S",
    "APPROVAL_POLL_INTERVAL_S",
    "DEFAULT_AI_SKILLS_REPO",
    "DEFAULT_MAX_NESTING_DEPTH",
    "STEP_TYPE",
    "AgentSchema",
    "RegistryEntry",
    "ResolvedSkill",
    "RiskLevel",
    "SkillApprovalDeniedError",
    "SkillApprovalTimeoutError",
    "SkillBridgeError",
    "SkillBudgetError",
    "SkillCallResult",
    "SkillConfigError",
    "SkillExecutor",
    "SkillInputError",
    "SkillNestingDepthError",
    "SkillOutputDriftError",
    "SkillResolutionError",
    "SkillResolver",
    "StepContext",
    "Tier",
    "TIER_OUTPUT_CEILING",
    "build_skill_step_handler",
]
