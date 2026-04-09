"""Animus-compatible tool definitions for bounty operations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from animus_bounty.config import BountyConfig
from animus_bounty.matcher import SkillMatcher
from animus_bounty.models import Bounty, Platform
from animus_bounty.platforms.base import ScanResult
from animus_bounty.platforms.gitcoin import GitcoinPlatform
from animus_bounty.platforms.github import GitHubPlatform
from animus_bounty.tracker import BountyTracker

logger = logging.getLogger("animus_bounty.tools")

_tracker: BountyTracker | None = None
_config: BountyConfig | None = None
_matcher: SkillMatcher | None = None


def _ensure_initialized() -> tuple[BountyTracker, BountyConfig, SkillMatcher]:
    global _tracker, _config, _matcher
    if _tracker is None:
        _config = BountyConfig()
        _config.ensure_dirs()
        if _config.config_file.exists():
            _config = BountyConfig.from_yaml(_config.config_file)
            _config.ensure_dirs()
        _tracker = BountyTracker(_config.data_dir)
        _matcher = SkillMatcher(_config)
    return _tracker, _config, _matcher  # type: ignore[return-value]


def _get_scanners(config: BountyConfig) -> list:
    """Build platform scanners based on config."""
    scanners = []
    allowed = config.filters.platforms
    if not allowed or "github" in allowed:
        scanners.append(GitHubPlatform(config))
    if not allowed or "gitcoin" in allowed:
        scanners.append(GitcoinPlatform(config))
    return scanners


async def _run_scan(config: BountyConfig, tracker: BountyTracker) -> dict[str, Any]:
    """Run a scan across all configured platforms."""
    scanners = _get_scanners(config)
    results: list[ScanResult] = []

    for scanner in scanners:
        result = await scanner.scan()
        results.append(result)

    total_new = 0
    platform_results = []
    for result in results:
        new_count = tracker.record_bounties(result.bounties)
        total_new += new_count
        platform_results.append(
            {
                "platform": result.platform_name,
                "found": result.count,
                "new": new_count,
                "errors": result.errors,
                "scan_seconds": round(result.scan_duration_seconds, 2),
            }
        )

    return {
        "total_scanned": sum(r.count for r in results),
        "total_new": total_new,
        "platforms": platform_results,
    }


def bounty_scan(params: dict[str, Any]) -> dict[str, Any]:
    """Scan platforms for new bounties."""
    tracker, config, _ = _ensure_initialized()
    report = asyncio.run(_run_scan(config, tracker))
    return {"success": True, "report": report}


def bounty_status(params: dict[str, Any]) -> dict[str, Any]:
    """Get bounty tracking summary."""
    tracker, _, _ = _ensure_initialized()
    summary = tracker.get_summary()
    return {"success": True, "summary": summary}


def bounty_match(params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a specific bounty against skill profile."""
    tracker, _, matcher = _ensure_initialized()
    bounty_id = params.get("bounty_id", "")
    url = params.get("url", "")

    if bounty_id:
        bounty = tracker.bounty_store.get_by_id(bounty_id)
    elif url:
        # Create a minimal bounty from URL for matching
        bounty = Bounty(
            title=params.get("title", "Unknown"),
            url=url,
            platform=Platform(params.get("platform", "generic")),
            description_snippet=params.get("description", ""),
            skills_required=params.get("skills_required", []),
        )
    else:
        return {"success": False, "error": "Provide bounty_id or url"}

    if bounty is None:
        return {"success": False, "error": f"Bounty not found: {bounty_id}"}

    match = matcher.quick_match(bounty)
    tracker.record_match(match)

    return {
        "success": True,
        "match": match.to_dict(),
        "auto_claim_recommended": matcher.should_auto_claim(bounty, match),
    }


def bounty_claim(params: dict[str, Any]) -> dict[str, Any]:
    """Claim a bounty."""
    tracker, _, _ = _ensure_initialized()
    bounty_id = params.get("bounty_id", "")
    notes = params.get("notes", "")

    if not bounty_id:
        return {"success": False, "error": "bounty_id required"}

    bounty = tracker.bounty_store.get_by_id(bounty_id)
    if bounty is None:
        return {"success": False, "error": f"Bounty not found: {bounty_id}"}

    existing = tracker.claim_store.get_for_bounty(bounty_id)
    if existing is not None:
        return {
            "success": False,
            "error": f"Bounty already claimed (status: {existing.status.value})",
        }

    claim = tracker.claim_bounty(bounty_id, notes=notes)
    return {"success": True, "claim": claim.to_dict()}


BOUNTY_TOOLS = [
    {
        "name": "bounty_scan",
        "description": "Scan platforms for new developer bounties (Gitcoin, GitHub, etc).",
        "parameters": {"type": "object", "properties": {}},
        "handler": bounty_scan,
        "category": "bounty",
    },
    {
        "name": "bounty_status",
        "description": "Get tracking summary: discovered, claimed, paid, earnings.",
        "parameters": {"type": "object", "properties": {}},
        "handler": bounty_status,
        "category": "bounty",
    },
    {
        "name": "bounty_match",
        "description": "Evaluate a bounty against your skill profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "bounty_id": {
                    "type": "string",
                    "description": "ID of a discovered bounty to evaluate",
                },
                "url": {
                    "type": "string",
                    "description": "URL of a bounty to evaluate (if not in store)",
                },
                "title": {"type": "string", "description": "Title of the bounty"},
                "description": {"type": "string", "description": "Bounty description"},
                "platform": {"type": "string", "description": "Platform name"},
                "skills_required": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skills the bounty requires",
                },
            },
        },
        "handler": bounty_match,
        "category": "bounty",
    },
    {
        "name": "bounty_claim",
        "description": "Claim a discovered bounty.",
        "parameters": {
            "type": "object",
            "properties": {
                "bounty_id": {
                    "type": "string",
                    "description": "ID of the bounty to claim",
                },
                "notes": {
                    "type": "string",
                    "description": "Notes about the claim",
                },
            },
            "required": ["bounty_id"],
        },
        "handler": bounty_claim,
        "category": "bounty",
    },
]
