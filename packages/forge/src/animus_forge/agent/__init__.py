"""Animus Agent Platform v0 — pure-local autonomous agent CLI.

Entry point: `animus-agent run <task>`. Wraps smolagents.CodeAgent with
Claude-Code-pattern skill registry, pre/post-tool-use hook gates, and
Forge's BudgetManager + audit-log integration.

Spec: `packages/forge/src/animus_forge/agent/spec.md` (v0.1.2, 2026-05-24)
Roadmap phase: RA-0 of docs/ROADMAP_research_assistant.md
"""

from __future__ import annotations

from .identity_roots import IDENTITY_ROOTS
from .types import (
    AgentContext,
    Allow,
    Deny,
    HookDecision,
    Receipt,
    ReceiptStatus,
    Skill,
    ToolCallRecord,
    ToolError,
    ToolResult,
)

__all__ = [
    "AgentContext",
    "Allow",
    "Deny",
    "HookDecision",
    "IDENTITY_ROOTS",
    "Receipt",
    "ReceiptStatus",
    "Skill",
    "ToolCallRecord",
    "ToolError",
    "ToolResult",
]
