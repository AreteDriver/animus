"""Animus-compatible tool definitions for prospector operations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from animus_prospector.config import ProspectorConfig
from animus_prospector.feed import OpportunityFeed
from animus_prospector.orchestrator import Prospector

logger = logging.getLogger("animus_prospector.tools")

_prospector: Prospector | None = None
_config: ProspectorConfig | None = None


def _ensure_initialized() -> tuple[Prospector, ProspectorConfig]:
    global _prospector, _config
    if _prospector is None:
        _config = ProspectorConfig()
        _config.ensure_dirs()
        if _config.config_file.exists():
            _config = ProspectorConfig.from_yaml(_config.config_file)
            _config.ensure_dirs()
        _prospector = Prospector(_config)
    return _prospector, _config


def prospector_scan(params: dict[str, Any]) -> dict[str, Any]:
    """Run a full opportunity scan."""
    prospector, _ = _ensure_initialized()
    report = asyncio.get_event_loop().run_until_complete(prospector.run_scan())
    return {"success": True, "report": report.to_dict()}


def prospector_feed(params: dict[str, Any]) -> dict[str, Any]:
    """Get the current opportunity feed."""
    _, config = _ensure_initialized()
    feed = OpportunityFeed(config.data_dir)

    view = params.get("view", "actionable")
    if view == "actionable":
        items = feed.get_actionable()
        return {
            "success": True,
            "opportunities": [
                {**opp.to_dict(), "evaluation": ev.to_dict() if ev else None} for opp, ev in items
            ],
        }
    elif view == "flagged":
        items = feed.get_flagged()
        return {
            "success": True,
            "opportunities": [
                {**opp.to_dict(), "evaluation": ev.to_dict() if ev else None} for opp, ev in items
            ],
        }
    elif view == "faucets":
        faucets = feed.get_new_faucets()
        return {
            "success": True,
            "opportunities": [f.to_dict() for f in faucets],
        }
    elif view == "summary":
        return {"success": True, "summary": feed.get_summary()}
    else:
        return {
            "success": False,
            "error": f"Unknown view: {view}. Use: actionable|flagged|faucets|summary",
        }


def prospector_health(params: dict[str, Any]) -> dict[str, Any]:
    """Check hunter health."""
    prospector, _ = _ensure_initialized()
    health = asyncio.get_event_loop().run_until_complete(prospector.health_check())
    return {"success": True, "hunters": health}


PROSPECTOR_TOOLS = [
    {
        "name": "prospector_scan",
        "description": "Scan for new crypto opportunities (airdrops, faucets, quests, giveaways).",
        "parameters": {"type": "object", "properties": {}},
        "handler": prospector_scan,
        "category": "prospector",
    },
    {
        "name": "prospector_feed",
        "description": "Get discovered opportunities. Views: actionable, flagged, faucets, summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "description": "Feed view: actionable|flagged|faucets|summary",
                },
            },
        },
        "handler": prospector_feed,
        "category": "prospector",
    },
    {
        "name": "prospector_health",
        "description": "Check health of all opportunity hunters.",
        "parameters": {"type": "object", "properties": {}},
        "handler": prospector_health,
        "category": "prospector",
    },
]
