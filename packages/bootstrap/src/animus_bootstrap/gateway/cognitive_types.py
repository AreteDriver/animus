"""Structured response types for cognitive backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CognitiveResponse:
    """Structured response from a cognitive backend."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens"
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Return True if any tool calls are present."""
        return len(self.tool_calls) > 0
