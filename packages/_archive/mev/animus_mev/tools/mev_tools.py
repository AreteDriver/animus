"""Animus-compatible tool definitions for MEV operations."""

from __future__ import annotations

import logging
from typing import Any

from animus_mev.config import MEVConfig
from animus_mev.store import MEVStore
from animus_mev.tracker import PnLTracker

logger = logging.getLogger("animus_mev.tools")

_config: MEVConfig | None = None
_store: MEVStore | None = None
_tracker: PnLTracker | None = None


def _ensure_initialized() -> tuple[MEVConfig, MEVStore, PnLTracker]:
    global _config, _store, _tracker
    if _config is None:
        _config = MEVConfig()
        _config.ensure_dirs()
        if _config.config_file.exists():
            _config = MEVConfig.from_yaml(_config.config_file)
            _config.ensure_dirs()
        _store = MEVStore(_config.data_dir)
        _tracker = PnLTracker()
    return _config, _store, _tracker


def mev_monitor(params: dict[str, Any]) -> dict[str, Any]:
    """Get MEV monitoring status."""
    config, store, _ = _ensure_initialized()

    action = params.get("action", "status")

    if action == "status":
        enabled = [c.chain.value for c in config.enabled_chains()]
        return {
            "success": True,
            "mode": "live" if config.execution.live else "dry_run",
            "enabled_chains": enabled,
            "total_opportunities": store.total_opportunities(),
            "sandwich_execution": "BLOCKED",
            "execution_config": {
                "live": config.execution.live,
                "private_tx": config.execution.private_tx,
                "max_slippage_bps": config.execution.max_slippage_bps,
            },
        }
    return {
        "success": False,
        "error": f"Unknown action: {action}. Use: status",
    }


def mev_opportunities(params: dict[str, Any]) -> dict[str, Any]:
    """Get recently detected MEV opportunities."""
    _, store, _ = _ensure_initialized()

    limit = params.get("limit", 20)
    recent = store.recent_opportunities(limit=limit)

    return {
        "success": True,
        "count": len(recent),
        "opportunities": [opp.to_dict() for opp in recent],
    }


def mev_status(params: dict[str, Any]) -> dict[str, Any]:
    """Get MEV P&L and extraction summary."""
    config, store, tracker = _ensure_initialized()

    summary = tracker.get_summary()
    results = store.recent_results(limit=10)

    return {
        "success": True,
        "pnl": summary.to_dict(),
        "recent_results": [r.to_dict() for r in results],
        "mode": "live" if config.execution.live else "dry_run",
    }


MEV_TOOLS = [
    {
        "name": "mev_monitor",
        "description": "Get MEV monitoring status — chains, mode, config.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: status",
                },
            },
        },
        "handler": mev_monitor,
        "category": "mev",
    },
    {
        "name": "mev_opportunities",
        "description": "Get recently detected MEV opportunities.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max opportunities to return (default 20)",
                },
            },
        },
        "handler": mev_opportunities,
        "category": "mev",
    },
    {
        "name": "mev_status",
        "description": "Get MEV P&L summary, extraction stats, and recent results.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "handler": mev_status,
        "category": "mev",
    },
]
