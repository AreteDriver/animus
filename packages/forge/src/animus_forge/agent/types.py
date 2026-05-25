"""Type definitions for the agent loop.

Concrete dataclass shapes for the names referenced in spec.md §4
(AgentContext, ToolResult, Receipt, Allow, Deny, Skill).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ToolError(Exception):
    """Raised by a tool when its input or environment is invalid.

    Distinct from `Exception`: tool errors are expected, captured by the
    agent loop, and surfaced back to the model as feedback (with a single
    retry per spec R2). Unexpected exceptions still propagate.
    """


class ReceiptStatus(str, Enum):
    """Terminal status for an agent run. Matches spec §4.9 receipt schema."""

    SUCCESS = "success"
    MAX_TURNS = "max_turns"
    BUDGET_EXCEEDED = "budget_exceeded"
    WALL_TIME_EXCEEDED = "wall_time_exceeded"
    TOOL_ERROR = "tool_error"
    AGENT_QUIT = "agent_quit"
    INTERNAL_ERROR = "internal_error"


# Exit-code mapping per spec R9.
EXIT_CODES: dict[ReceiptStatus, int] = {
    ReceiptStatus.SUCCESS: 0,
    ReceiptStatus.MAX_TURNS: 1,
    ReceiptStatus.BUDGET_EXCEEDED: 2,
    ReceiptStatus.WALL_TIME_EXCEEDED: 3,
    ReceiptStatus.TOOL_ERROR: 4,
    ReceiptStatus.AGENT_QUIT: 5,
    ReceiptStatus.INTERNAL_ERROR: 6,
}


@dataclass(frozen=True)
class Allow:
    """Hook decision: permit the tool call. Optional reason for logging."""

    reason: str = ""


@dataclass(frozen=True)
class Deny:
    """Hook decision: block the tool call. Reason is surfaced to the model."""

    reason: str


HookDecision = Allow | Deny


@dataclass
class ToolResult:
    """Outcome of a single tool invocation."""

    tool: str
    status: str  # "ok" | "error" | "denied"
    output: Any
    wall_ms: float
    error: str | None = None


@dataclass
class ToolCallRecord:
    """One entry in Receipt.tool_calls (spec §4.9)."""

    turn: int
    tool: str
    args_summary: str
    status: str  # "ok" | "error" | "denied"
    wall_ms: float


@dataclass
class Skill:
    """Loaded Claude Code skill (spec §4.7).

    Discovery: any directory under `~/.claude/skills/` containing
    `SKILL.md` or `skill.md`. YAML frontmatter parsed for `name`,
    `description`, `invokable` (default True).
    """

    name: str
    description: str
    body: str
    invokable: bool
    source_path: Path


@dataclass
class AgentContext:
    """Live state passed to hook functions across the agent loop.

    Hooks receive this read-only conceptually (the loop owns mutation);
    hook authors mutate via documented sinks only (`context.metadata`).
    """

    task: str
    project_root: Path
    provider: str  # e.g. "llamacpp"
    model: str  # e.g. "qwen3.6-research"
    turn: int = 0
    max_turns: int = 50
    budget_tokens: int = 200_000
    wall_seconds: int = 1800
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Receipt:
    """Final artifact emitted by an agent run (spec §4.9).

    JSON-serializable via `to_dict()`. Status drives exit code via EXIT_CODES.
    """

    task: str
    status: ReceiptStatus
    turns_taken: int
    tokens_in: int
    tokens_out: int
    wall_seconds: float
    model: str
    tool_calls: list[ToolCallRecord]
    skill_load_warnings: list[str]
    hook_errors: list[str]
    final_message: str
    error: str | None
    started_at: datetime
    ended_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status.value,
            "turns_taken": self.turns_taken,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "wall_seconds": self.wall_seconds,
            "model": self.model,
            "tool_calls": [
                {
                    "turn": tc.turn,
                    "tool": tc.tool,
                    "args_summary": tc.args_summary,
                    "status": tc.status,
                    "wall_ms": tc.wall_ms,
                }
                for tc in self.tool_calls
            ],
            "skill_load_warnings": list(self.skill_load_warnings),
            "hook_errors": list(self.hook_errors),
            "final_message": self.final_message,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
        }

    def exit_code(self) -> int:
        return EXIT_CODES[self.status]
