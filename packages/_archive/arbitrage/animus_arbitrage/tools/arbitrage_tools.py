"""Animus-compatible tool definitions for arbitrage operations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.detector import ArbitrageDetector
from animus_arbitrage.executor import TradeExecutor
from animus_arbitrage.risk import RiskManager
from animus_arbitrage.store import ArbitrageStore
from animus_arbitrage.tracker import PnLTracker

logger = logging.getLogger("animus_arbitrage.tools")

_config: ArbitrageConfig | None = None
_detector: ArbitrageDetector | None = None
_executor: TradeExecutor | None = None
_tracker: PnLTracker | None = None
_store: ArbitrageStore | None = None
_risk: RiskManager | None = None


def _ensure_initialized() -> tuple[
    ArbitrageConfig, ArbitrageDetector, TradeExecutor, PnLTracker, RiskManager
]:
    global _config, _detector, _executor, _tracker, _store, _risk
    if _config is None:
        _config = ArbitrageConfig()
        _config.ensure_dirs()
        if _config.config_file.exists():
            _config = ArbitrageConfig.from_yaml(_config.config_file)
            _config.ensure_dirs()
        _store = ArbitrageStore(_config.data_dir)
        _risk = RiskManager(_config)
        _detector = ArbitrageDetector(_config)
        _executor = TradeExecutor(_config, _risk)
        _tracker = PnLTracker(_store)
    return _config, _detector, _executor, _tracker, _risk


def arbitrage_scan(params: dict[str, Any]) -> dict[str, Any]:
    """Scan for current arbitrage opportunities."""
    _, detector, _, tracker, _ = _ensure_initialized()

    trade_size = params.get("trade_size_usd", 0.0)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    opportunities = loop.run_until_complete(detector.scan(trade_size_usd=trade_size))

    # Record discovered opportunities
    global _store
    if _store is not None:
        _store.record_opportunities(opportunities)

    return {
        "success": True,
        "opportunities": [o.to_dict() for o in opportunities],
        "count": len(opportunities),
    }


def arbitrage_status(params: dict[str, Any]) -> dict[str, Any]:
    """Get P&L tracking summary and risk status."""
    _, _, _, tracker, risk = _ensure_initialized()

    return {
        "success": True,
        "pnl": tracker.get_summary(),
        "risk": risk.status,
        "recent_trades": tracker.get_recent(limit=params.get("limit", 10)),
    }


def arbitrage_config(params: dict[str, Any]) -> dict[str, Any]:
    """View or update arbitrage configuration."""
    config, _, _, _, _ = _ensure_initialized()

    # If update params provided, update risk config
    updates = params.get("updates", {})
    if updates:
        risk = config.risk
        for key, value in updates.items():
            if hasattr(risk, key):
                setattr(risk, key, value)
        return {
            "success": True,
            "message": f"Updated {len(updates)} parameter(s)",
            "config": config.to_dict(),
        }

    return {
        "success": True,
        "config": config.to_dict(),
    }


def arbitrage_execute(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a specific arbitrage opportunity. Requires live mode."""
    config, _, executor, tracker, _ = _ensure_initialized()

    if not config.execution.is_live:
        return {
            "success": False,
            "error": (
                "Execution blocked: mode is DRY_RUN. "
                "Set execution.mode = 'live' in config to enable."
            ),
        }

    opportunity_id = params.get("opportunity_id", "")
    if not opportunity_id:
        return {"success": False, "error": "opportunity_id is required"}

    global _store
    if _store is None:
        return {"success": False, "error": "Store not initialized"}

    # Find the opportunity
    opportunities = _store.load_opportunities()
    target = None
    for opp in opportunities:
        if opp.opportunity_id == opportunity_id:
            target = opp
            break

    if target is None:
        return {"success": False, "error": f"Opportunity {opportunity_id} not found"}

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(executor.execute(target))
        tracker.record_result(result)
        return {"success": True, "result": result.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


ARBITRAGE_TOOLS = [
    {
        "name": "arbitrage_scan",
        "description": (
            "Scan for current DEX arbitrage opportunities across chains. "
            "Detects price spreads that exceed gas costs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trade_size_usd": {
                    "type": "number",
                    "description": "Trade size in USD to calculate profit (default: config max)",
                },
            },
        },
        "handler": arbitrage_scan,
        "category": "arbitrage",
    },
    {
        "name": "arbitrage_status",
        "description": "Get P&L summary, risk status, and recent trade history.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent trades to show (default: 10)",
                },
            },
        },
        "handler": arbitrage_status,
        "category": "arbitrage",
    },
    {
        "name": "arbitrage_config",
        "description": "View or update arbitrage risk parameters and configuration.",
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "object",
                    "description": "Key-value pairs to update in risk config",
                },
            },
        },
        "handler": arbitrage_config,
        "category": "arbitrage",
    },
    {
        "name": "arbitrage_execute",
        "description": (
            "Execute a specific arbitrage opportunity. "
            "REQUIRES live mode enabled in config. Dry-run by default."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "opportunity_id": {
                    "type": "string",
                    "description": "ID of the opportunity to execute",
                },
            },
            "required": ["opportunity_id"],
        },
        "handler": arbitrage_execute,
        "category": "arbitrage",
    },
]
