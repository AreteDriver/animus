"""Identity file tools — guardrailed read/write access to identity files."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# Module-level reference, set via set_identity_manager()
_identity_manager = None
_improvement_store = None


def set_identity_manager(manager) -> None:  # noqa: ANN001
    """Wire the live IdentityFileManager into identity tools."""
    global _identity_manager  # noqa: PLW0603
    _identity_manager = manager


def set_identity_improvement_store(store) -> None:  # noqa: ANN001
    """Wire the ImprovementStore for proposal creation on large changes."""
    global _improvement_store  # noqa: PLW0603
    _improvement_store = store


def _create_proposal(filename: str, current: str, content: str, reason: str) -> dict:
    """Create an identity change proposal in the improvement store."""
    proposal = {
        "area": f"identity:{filename}",
        "description": f"Proposed change to {filename}: {reason}",
        "status": "proposed",
        "timestamp": datetime.now(UTC).isoformat(),
        "analysis": f"Current length: {len(current)} chars, proposed: {len(content)} chars",
        "patch": content,
    }
    if _improvement_store is not None:
        proposal_id = _improvement_store.save(proposal)
        proposal["id"] = proposal_id
    else:
        proposal["id"] = 0
    return proposal


async def _identity_read(arguments: dict) -> str:
    """Read an identity file."""
    if _identity_manager is None:
        return "Identity manager not available."
    filename = arguments.get("filename", "")
    try:
        content = _identity_manager.read(filename)
        if not content:
            return f"{filename} is empty or does not exist."
        return content
    except ValueError as exc:
        return str(exc)


async def _identity_write(arguments: dict) -> str:
    """Write to an editable identity file with guardrails."""
    if _identity_manager is None:
        return "Identity manager not available."

    filename = arguments.get("filename", "")
    content = arguments.get("content", "")
    reason = arguments.get("reason", "No reason provided")

    # Hard block on locked files
    if filename == "CORE_VALUES.md":
        return (
            "CORE_VALUES.md is immutable and cannot be modified by Animus. "
            "Edit manually or via the dashboard."
        )

    try:
        _identity_manager._validate_filename(filename)
    except ValueError as exc:
        return str(exc)

    # 20% change threshold — large changes become proposals
    current = _identity_manager.read(filename)
    if current and len(content) > 0:
        change_ratio = abs(len(content) - len(current)) / max(len(current), 1)
        if change_ratio > 0.20:
            proposal = _create_proposal(filename, current, content, reason)
            return (
                f"Change exceeds 20% threshold ({change_ratio:.0%} size delta). "
                f"Proposal #{proposal['id']} created, pending approval in dashboard."
            )

    # Small changes: write directly
    try:
        _identity_manager.write(filename, content)
        return f"Successfully updated {filename}."
    except PermissionError as exc:
        return str(exc)


async def _identity_append_learned(arguments: dict) -> str:
    """Append a timestamped entry to LEARNED.md."""
    if _identity_manager is None:
        return "Identity manager not available."

    section = arguments.get("section", "General")
    entry = arguments.get("entry", "")
    if not entry:
        return "No entry provided."

    _identity_manager.append_to_learned(section, entry)
    return f"Added entry to LEARNED.md under '{section}'."


async def _identity_list(arguments: dict) -> str:  # noqa: ARG001
    """List all identity files with sizes and locked status."""
    if _identity_manager is None:
        return "Identity manager not available."

    files = []
    for filename in _identity_manager.ALL_FILES:
        exists = _identity_manager.exists(filename)
        size = len(_identity_manager.read(filename)) if exists else 0
        locked = filename in _identity_manager.LOCKED_FILES
        files.append(
            {
                "filename": filename,
                "exists": exists,
                "size_bytes": size,
                "locked": locked,
            }
        )
    return json.dumps(files, indent=2)


def get_identity_tools() -> list[ToolDefinition]:
    """Return identity file tool definitions."""
    return [
        ToolDefinition(
            name="identity_read",
            description="Read an identity file (CORE_VALUES.md, IDENTITY.md, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Identity file name (e.g., IDENTITY.md, CORE_VALUES.md)",
                    },
                },
                "required": ["filename"],
            },
            handler=_identity_read,
            category="identity",
            permission="auto",
        ),
        ToolDefinition(
            name="identity_write",
            description=(
                "Write to an editable identity file. Cannot modify CORE_VALUES.md. "
                "Changes >20% of file size become proposals for human review."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Identity file name to write (not CORE_VALUES.md)",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the file",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the change",
                    },
                },
                "required": ["filename", "content"],
            },
            handler=_identity_write,
            category="identity",
            permission="approve",
        ),
        ToolDefinition(
            name="identity_append_learned",
            description="Append a timestamped entry to LEARNED.md under a section header.",
            parameters={
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Section header in LEARNED.md",
                    },
                    "entry": {
                        "type": "string",
                        "description": "The entry to append",
                    },
                },
                "required": ["section", "entry"],
            },
            handler=_identity_append_learned,
            category="identity",
            permission="auto",
        ),
        ToolDefinition(
            name="identity_list",
            description="List all identity files with sizes and locked status.",
            parameters={"type": "object", "properties": {}},
            handler=_identity_list,
            category="identity",
            permission="auto",
        ),
    ]
