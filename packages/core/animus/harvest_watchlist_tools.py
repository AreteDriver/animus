"""Tool definitions and handlers for harvest watchlist integration."""

from __future__ import annotations

import asyncio
import json
import logging

from animus.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


def _tool_watchlist_add(params: dict) -> ToolResult:
    """Tool handler for animus_watchlist_add."""
    from animus.harvest_watchlist import add_to_watchlist

    target = params.get("target")
    if not target:
        return ToolResult(
            tool_name="animus_watchlist_add",
            success=False,
            output=None,
            error="Missing required parameter: target",
        )

    tags = params.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    notes = params.get("notes")

    try:
        entry = add_to_watchlist(target=target, tags=tags, notes=notes)
        return ToolResult(
            tool_name="animus_watchlist_add",
            success=True,
            output=json.dumps(entry, indent=2),
        )
    except ValueError as e:
        return ToolResult(
            tool_name="animus_watchlist_add",
            success=False,
            output=None,
            error=str(e),
        )
    except Exception as e:
        logger.exception("Watchlist add failed for %s", target)
        return ToolResult(
            tool_name="animus_watchlist_add",
            success=False,
            output=None,
            error=f"Failed to add: {e}",
        )


def _tool_watchlist_remove(params: dict) -> ToolResult:
    """Tool handler for animus_watchlist_remove."""
    from animus.harvest_watchlist import remove_from_watchlist

    target = params.get("target")
    if not target:
        return ToolResult(
            tool_name="animus_watchlist_remove",
            success=False,
            output=None,
            error="Missing required parameter: target",
        )

    try:
        removed = remove_from_watchlist(target=target)
        if removed:
            return ToolResult(
                tool_name="animus_watchlist_remove",
                success=True,
                output=f"Removed '{target}' from watchlist",
            )
        return ToolResult(
            tool_name="animus_watchlist_remove",
            success=False,
            output=None,
            error=f"'{target}' not found on watchlist",
        )
    except Exception as e:
        logger.exception("Watchlist remove failed for %s", target)
        return ToolResult(
            tool_name="animus_watchlist_remove",
            success=False,
            output=None,
            error=f"Failed to remove: {e}",
        )


def _tool_watchlist_list(params: dict) -> ToolResult:
    """Tool handler for animus_watchlist_list."""
    from animus.harvest_watchlist import get_watchlist

    try:
        repos = get_watchlist()
        return ToolResult(
            tool_name="animus_watchlist_list",
            success=True,
            output=json.dumps(repos, indent=2),
        )
    except Exception as e:
        logger.exception("Watchlist list failed")
        return ToolResult(
            tool_name="animus_watchlist_list",
            success=False,
            output=None,
            error=f"Failed to list: {e}",
        )


def _tool_watchlist_scan(params: dict) -> ToolResult:
    """Tool handler for animus_watchlist_scan."""
    from animus.harvest_watchlist import run_watchlist_scan

    interval = params.get("interval_hours")

    try:
        report = asyncio.run(run_watchlist_scan(interval_hours=interval))
        return ToolResult(
            tool_name="animus_watchlist_scan",
            success=True,
            output=json.dumps(report, indent=2),
        )
    except Exception as e:
        logger.exception("Watchlist scan failed")
        return ToolResult(
            tool_name="animus_watchlist_scan",
            success=False,
            output=None,
            error=f"Scan failed: {e}",
        )


WATCHLIST_ADD_TOOL = Tool(
    name="animus_watchlist_add",
    description="Add a GitHub repo to the competition watchlist for periodic scanning",
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "GitHub repo URL or username/repo",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags (e.g., 'competitor,eve-frontier')",
            },
            "notes": {
                "type": "string",
                "description": "Notes about why this repo matters",
            },
        },
        "required": ["target"],
    },
    handler=_tool_watchlist_add,
    category="analysis",
)

WATCHLIST_REMOVE_TOOL = Tool(
    name="animus_watchlist_remove",
    description="Remove a GitHub repo from the competition watchlist",
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "GitHub repo URL or username/repo to remove",
            },
        },
        "required": ["target"],
    },
    handler=_tool_watchlist_remove,
    category="analysis",
)

WATCHLIST_LIST_TOOL = Tool(
    name="animus_watchlist_list",
    description="List all repos on the competition watchlist with their last scan data",
    parameters={
        "type": "object",
        "properties": {},
    },
    handler=_tool_watchlist_list,
    category="analysis",
)

WATCHLIST_SCAN_TOOL = Tool(
    name="animus_watchlist_scan",
    description="Run harvest scans on all due repos and return a changes report",
    parameters={
        "type": "object",
        "properties": {
            "interval_hours": {
                "type": "integer",
                "description": "Override scan interval in hours (default: 168 = 7 days)",
            },
        },
    },
    handler=_tool_watchlist_scan,
    category="analysis",
)
