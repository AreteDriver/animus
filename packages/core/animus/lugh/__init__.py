"""
Lugh — Animus harvesting layer.

Named for the Celtic god of many skills (agriculture, craft, commerce).
Submodules:
    repos          — repo pattern harvester (GitHub -> learnable patterns)
    watchlist      — periodic scan scheduling + diff reports
    watchlist_tools — Tool wrappers for MCP/CLI registration
    transcripts    — Claude Code JSONL session harvester
    models         — shared dataclasses
"""

from __future__ import annotations

from animus.lugh.cache import TranscriptCache
from animus.lugh.drift_client import DriftMonitorClient
from animus.lugh.models import SessionSummary, Turn
from animus.lugh.repos import (
    HARVEST_TOOL,
    HarvestResult,
    _extract_repo_name,
    _normalize_target,
    harvest_repo,
)
from animus.lugh.transcripts import (
    KNOWN_LINE_TYPES,
    MODEL_PRICING,
    MODEL_PRICING_LAST_UPDATED,
    drift_signals,
    harvest_transcripts,
    parse_session,
    rollup_by_cwd,
)
from animus.lugh.watchlist import (
    DEFAULT_SCAN_INTERVAL_HOURS,
    WATCHLIST_FILE,
    add_to_watchlist,
    get_changes_report,
    get_due_repos,
    get_watchlist,
    remove_from_watchlist,
    run_watchlist_scan,
    update_scan_result,
)
from animus.lugh.watchlist_tools import (
    WATCHLIST_ADD_TOOL,
    WATCHLIST_LIST_TOOL,
    WATCHLIST_REMOVE_TOOL,
    WATCHLIST_SCAN_TOOL,
)

__all__ = [
    # repos
    "HARVEST_TOOL",
    "HarvestResult",
    "harvest_repo",
    "_extract_repo_name",
    "_normalize_target",
    # watchlist
    "WATCHLIST_FILE",
    "DEFAULT_SCAN_INTERVAL_HOURS",
    "add_to_watchlist",
    "get_changes_report",
    "get_due_repos",
    "get_watchlist",
    "remove_from_watchlist",
    "run_watchlist_scan",
    "update_scan_result",
    # watchlist tools
    "WATCHLIST_ADD_TOOL",
    "WATCHLIST_LIST_TOOL",
    "WATCHLIST_REMOVE_TOOL",
    "WATCHLIST_SCAN_TOOL",
    # transcripts
    "SessionSummary",
    "Turn",
    "drift_signals",
    "harvest_transcripts",
    "parse_session",
    "rollup_by_cwd",
    "KNOWN_LINE_TYPES",
    "MODEL_PRICING",
    "MODEL_PRICING_LAST_UPDATED",
    # infrastructure
    "TranscriptCache",
    "DriftMonitorClient",
]
