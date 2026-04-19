"""Shared dataclasses for the Lugh harvesting layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    """One conversational turn from a Claude Code JSONL transcript."""

    session_id: str
    turn_index: int
    timestamp: datetime | None
    role: str  # "user" | "assistant"
    is_sidechain: bool  # True if this turn is inside a subagent transcript
    agent_id: str | None  # subagent id when is_sidechain
    cwd: str | None  # working directory at turn time (real project key)
    model: str  # normalized model name
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    tool_calls: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    activity: str = "conversation"
    cost_usd: float = 0.0


@dataclass
class SessionSummary:
    """Aggregate view of a single Claude Code session file."""

    session_id: str
    session_file: str  # path relative to ~/.claude/projects
    is_sidechain: bool  # True for subagent transcripts
    agent_id: str | None
    project_dir: str  # mangled dir name under projects/
    primary_cwd: str | None  # most frequent cwd across turns
    date: str | None  # YYYY-MM-DD (min turn timestamp)
    duration_seconds: float | None  # wall clock (max - min timestamp)
    active_seconds: float | None  # sum of inter-turn gaps below IDLE_GAP_SECONDS
    turn_count: int
    tool_call_count: int
    mcp_call_count: int
    model_distribution: dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_cost_usd: float
    activity_breakdown: dict[str, float]  # activity -> % of turns
    cwd_breakdown: dict[str, int]  # cwd -> turn count
    unknown_line_types: list[str] = field(default_factory=list)  # schema drift signals
    unknown_models: list[str] = field(default_factory=list)  # unpriced models
    turns: list[Turn] = field(default_factory=list)
