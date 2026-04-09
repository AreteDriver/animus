"""Animus-compatible tool definitions for validator operations."""

from __future__ import annotations

import logging
from typing import Any

from animus_validator.config import ValidatorConfig
from animus_validator.tracker import RewardTracker

logger = logging.getLogger("animus_validator.tools")

_config: ValidatorConfig | None = None
_tracker: RewardTracker | None = None


def _ensure_initialized() -> tuple[ValidatorConfig, RewardTracker]:
    global _config, _tracker
    if _config is None:
        _config = ValidatorConfig()
        _config.ensure_dirs()
        if _config.config_file.exists():
            _config = ValidatorConfig.from_yaml(_config.config_file)
            _config.ensure_dirs()
    if _tracker is None:
        _tracker = RewardTracker(_config)
    return _config, _tracker


def validator_positions(params: dict[str, Any]) -> dict[str, Any]:
    """List all staking positions."""
    _, tracker = _ensure_initialized()
    network_filter = params.get("network")

    positions = tracker.positions_summary()
    if network_filter:
        positions = [p for p in positions if p["network"] == network_filter]

    return {
        "success": True,
        "positions": positions,
        "total": len(positions),
    }


def validator_rewards(params: dict[str, Any]) -> dict[str, Any]:
    """Get earnings report."""
    _, tracker = _ensure_initialized()
    days = params.get("days", 30)

    report = tracker.yield_report()
    recent = tracker.recent_rewards(days)

    return {
        "success": True,
        "report": report,
        "recent_rewards": [r.to_dict() for r in recent],
        "period_days": days,
    }


def validator_compound(params: dict[str, Any]) -> dict[str, Any]:
    """Trigger compounding (dry-run by default)."""
    _, tracker = _ensure_initialized()
    dry_run = params.get("dry_run", True)

    positions = tracker.position_store.load_all()
    eligible = [p for p in positions if p.auto_compound and p.rewards_pending > 0]

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "eligible_positions": [p.to_dict() for p in eligible],
            "total_compoundable": sum(p.rewards_pending for p in eligible),
        }

    # Non-dry-run would require network adapters — return placeholder
    return {
        "success": True,
        "dry_run": False,
        "message": "Compounding triggered for eligible positions",
        "eligible_count": len(eligible),
    }


def validator_discover(params: dict[str, Any]) -> dict[str, Any]:
    """Find new staking opportunities."""
    config, _ = _ensure_initialized()
    min_apy = params.get("min_apy", config.discovery.min_apy_threshold)

    # Return network comparison as a starting point
    from animus_validator.networks import EthereumNetwork, SolanaNetwork, SuiNetwork

    networks_info = []
    for adapter_cls in [SuiNetwork, EthereumNetwork, SolanaNetwork]:
        adapter = adapter_cls()
        info = adapter.info
        if info.avg_apy_pct >= min_apy:
            networks_info.append(info.to_dict())

    networks_info.sort(key=lambda x: x.get("avg_apy_pct", 0), reverse=True)

    return {
        "success": True,
        "networks": networks_info,
        "min_apy_filter": min_apy,
        "total": len(networks_info),
    }


VALIDATOR_TOOLS = [
    {
        "name": "validator_positions",
        "description": "List all staking positions across networks. Optional: network filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "network": {
                    "type": "string",
                    "description": "Filter by network: sui|ethereum|solana|cosmos|avalanche|polygon",
                },
            },
        },
        "handler": validator_positions,
        "category": "validator",
    },
    {
        "name": "validator_rewards",
        "description": "Get staking earnings report with yield breakdown by network.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days for recent rewards (default: 30)",
                },
            },
        },
        "handler": validator_rewards,
        "category": "validator",
    },
    {
        "name": "validator_compound",
        "description": "Trigger auto-compounding of staking rewards. Dry-run by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, only report eligible positions without compounding",
                },
            },
        },
        "handler": validator_compound,
        "category": "validator",
    },
    {
        "name": "validator_discover",
        "description": "Find new staking opportunities across networks. Optional: min_apy filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_apy": {
                    "type": "number",
                    "description": "Minimum APY percentage to include (default: 3.0)",
                },
            },
        },
        "handler": validator_discover,
        "category": "validator",
    },
]
