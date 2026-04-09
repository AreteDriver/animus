"""Animus-compatible tool definitions for referral operations.

These tools follow the Core ToolRegistry pattern so they can be registered
as Animus tools and exposed via MCP.
"""

from __future__ import annotations

import logging
from typing import Any

from animus_referral.config import ReferralConfig
from animus_referral.manager import ReferralManager
from animus_referral.models import BonusType, ProgramType, ReferralProgram
from animus_referral.tracker import ReferralTracker

logger = logging.getLogger("animus_referral.tools")

_manager: ReferralManager | None = None
_config: ReferralConfig | None = None
_tracker: ReferralTracker | None = None


def _ensure_initialized() -> tuple[ReferralManager, ReferralConfig, ReferralTracker]:
    """Lazy init from default config."""
    global _manager, _config, _tracker
    if _manager is None:
        _config = ReferralConfig()
        _config.ensure_dirs()
        if _config.config_file.exists():
            _config = ReferralConfig.from_yaml(_config.config_file)
            _config.ensure_dirs()
        _manager = ReferralManager(_config)
        _tracker = ReferralTracker(_config.data_dir)
    return _manager, _config, _tracker


def reset_tools_state() -> None:
    """Reset module-level singletons. Used in tests."""
    global _manager, _config, _tracker
    _manager = None
    _config = None
    _tracker = None


def referral_programs(params: dict[str, Any]) -> dict[str, Any]:
    """List or add referral programs."""
    manager, config, _ = _ensure_initialized()

    action = params.get("action", "list")

    if action == "add":
        name = params.get("name")
        platform_url = params.get("platform_url")
        if not name or not platform_url:
            return {"success": False, "error": "Both 'name' and 'platform_url' are required"}

        program_type_str = params.get("program_type", "other")
        try:
            program_type = ProgramType(program_type_str)
        except ValueError:
            valid = [t.value for t in ProgramType]
            return {"success": False, "error": f"Invalid program_type. Valid: {valid}"}

        bonus_type_str = params.get("bonus_type", "flat")
        try:
            bonus_type = BonusType(bonus_type_str)
        except ValueError:
            valid = [t.value for t in BonusType]
            return {"success": False, "error": f"Invalid bonus_type. Valid: {valid}"}

        program = ReferralProgram(
            name=name,
            platform_url=platform_url,
            program_type=program_type,
            bonus_referrer_usd=params.get("bonus_referrer_usd", 0.0),
            bonus_referee_usd=params.get("bonus_referee_usd", 0.0),
            bonus_type=bonus_type,
            requirements=params.get("requirements", ""),
            referral_url_template=params.get("referral_url_template", ""),
            chain=params.get("chain", ""),
        )
        added = manager.add_program(program)
        if not added:
            return {"success": False, "error": f"Program '{name}' already exists"}
        return {"success": True, "program": program.to_dict()}

    # Default: list
    program_type_filter = params.get("program_type")
    active_only = params.get("active_only", True)
    ptype = ProgramType(program_type_filter) if program_type_filter else None
    programs = manager.get_programs(program_type=ptype, active_only=active_only)
    return {
        "success": True,
        "programs": [p.to_dict() for p in programs],
        "total": len(programs),
    }


def referral_status(params: dict[str, Any]) -> dict[str, Any]:
    """Get referral earnings summary."""
    manager, _, tracker = _ensure_initialized()
    days = params.get("days", 30)
    report = tracker.get_report(days=days)
    summary = manager.get_summary()
    return {
        "success": True,
        "summary": summary,
        "report": report,
    }


def referral_link(params: dict[str, Any]) -> dict[str, Any]:
    """Generate or get a referral link for a program."""
    manager, _, _ = _ensure_initialized()

    program_name = params.get("program_name")
    referrer_alias = params.get("referrer_alias")
    referral_code = params.get("referral_code")

    if not program_name or not referrer_alias:
        return {
            "success": False,
            "error": "Both 'program_name' and 'referrer_alias' are required",
        }

    # Try to get existing link first
    existing = manager.get_link(program_name, referrer_alias)
    if existing:
        return {"success": True, "link": existing.to_dict(), "existing": True}

    # Generate new link
    if not referral_code:
        return {"success": False, "error": "'referral_code' required to generate new link"}

    link = manager.generate_link(program_name, referrer_alias, referral_code)
    if link is None:
        return {"success": False, "error": f"Program '{program_name}' not found"}

    return {"success": True, "link": link.to_dict(), "existing": False}


REFERRAL_TOOLS = [
    {
        "name": "referral_programs",
        "description": (
            "List or add referral programs. "
            "Use action='list' to view programs, action='add' to register a new one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'list' or 'add' (default: list)",
                    "enum": ["list", "add"],
                },
                "name": {
                    "type": "string",
                    "description": "Program name (required for add)",
                },
                "platform_url": {
                    "type": "string",
                    "description": "Platform URL (required for add)",
                },
                "program_type": {
                    "type": "string",
                    "description": "Type: exchange, defi, faucet, wallet, quest_platform, other",
                },
                "bonus_referrer_usd": {
                    "type": "number",
                    "description": "Bonus for referrer in USD",
                },
                "bonus_referee_usd": {
                    "type": "number",
                    "description": "Bonus for referee in USD",
                },
                "bonus_type": {
                    "type": "string",
                    "description": "Bonus type: flat, percentage, token",
                },
                "requirements": {
                    "type": "string",
                    "description": "Requirements for the referral to count",
                },
                "referral_url_template": {
                    "type": "string",
                    "description": "URL template with {code} placeholder",
                },
                "chain": {
                    "type": "string",
                    "description": "Blockchain network",
                },
                "active_only": {
                    "type": "boolean",
                    "description": "Only show active programs (default: true)",
                },
            },
        },
        "handler": referral_programs,
        "category": "referral",
    },
    {
        "name": "referral_status",
        "description": "Get referral earnings summary including totals, top programs, and conversion rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to report on (default 30)",
                },
            },
        },
        "handler": referral_status,
        "category": "referral",
    },
    {
        "name": "referral_link",
        "description": "Generate or retrieve a referral link for a program and account.",
        "parameters": {
            "type": "object",
            "properties": {
                "program_name": {
                    "type": "string",
                    "description": "Name of the referral program",
                },
                "referrer_alias": {
                    "type": "string",
                    "description": "Alias of the referring account",
                },
                "referral_code": {
                    "type": "string",
                    "description": "Referral code (required for new link generation)",
                },
            },
            "required": ["program_name", "referrer_alias"],
        },
        "handler": referral_link,
        "category": "referral",
    },
]
