"""Tool registry and safety for agent execution."""

from __future__ import annotations

from .models import (
    DirectoryListing,
    EditProposal,
    FileContent,
    SearchResult,
    ToolCallRequest,
    ToolCallResult,
)
from .proposals import ProposalManager
from .registry import ForgeToolRegistry, ToolDefinition
from .safety import PathValidator, SecurityError
from .schema_validator import (
    ValidatingToolRegistry,
    parse_hermes_tool_call,
    parse_json_tool_call,
    validate_tool_call,
)

__all__ = [
    "DirectoryListing",
    "EditProposal",
    "FileContent",
    "ForgeToolRegistry",
    "PathValidator",
    "ProposalManager",
    "SearchResult",
    "SecurityError",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolDefinition",
    "ValidatingToolRegistry",
    "parse_hermes_tool_call",
    "parse_json_tool_call",
    "validate_tool_call",
]
