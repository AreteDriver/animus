"""Animus-compatible tool definitions for faucet operations.

These tools follow the Core ToolRegistry pattern so they can be registered
as Animus tools and exposed via MCP.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from animus_faucet.config import FaucetConfig
from animus_faucet.models import Chain
from animus_faucet.scheduler import FaucetScheduler
from animus_faucet.store import ClaimStore
from animus_faucet.wallet.manager import WalletManager

logger = logging.getLogger("animus_faucet.tools")

# Module-level singleton — initialized by register_faucet_tools()
_scheduler: FaucetScheduler | None = None
_config: FaucetConfig | None = None
_wallet_manager: WalletManager | None = None


def _ensure_initialized() -> tuple[FaucetScheduler, FaucetConfig, WalletManager]:
    """Lazy init from default config."""
    global _scheduler, _config, _wallet_manager
    if _scheduler is None:
        _config = FaucetConfig()
        _config.ensure_dirs()
        if _config.faucets_file.exists():
            _config = FaucetConfig.from_yaml(_config.faucets_file)
            _config.ensure_dirs()
        _wallet_manager = WalletManager(_config.data_dir)
        _scheduler = FaucetScheduler(_config, _wallet_manager)
    return _scheduler, _config, _wallet_manager


def faucet_claim(params: dict[str, Any]) -> dict[str, Any]:
    """Claim from a specific faucet or all enabled faucets."""
    scheduler, config, _ = _ensure_initialized()
    faucet_name = params.get("faucet_name")

    if faucet_name:
        faucet = next((f for f in config.faucets if f.name == faucet_name), None)
        if faucet is None:
            return {"success": False, "error": f"Faucet '{faucet_name}' not found"}
        results = asyncio.get_event_loop().run_until_complete(scheduler.claim_faucet(faucet))
        return {"success": True, "result": results.to_dict()}
    else:
        results = asyncio.get_event_loop().run_until_complete(scheduler.run_once())
        return {
            "success": True,
            "results": [r.to_dict() for r in results],
            "succeeded": sum(1 for r in results if r.succeeded),
            "total": len(results),
        }


def faucet_status(params: dict[str, Any]) -> dict[str, Any]:
    """Get faucet health and earnings summary."""
    scheduler, _, _ = _ensure_initialized()
    return {"success": True, "summary": scheduler.get_summary()}


def faucet_add_wallet(params: dict[str, Any]) -> dict[str, Any]:
    """Register a wallet address for a chain."""
    _, _, wallet_manager = _ensure_initialized()
    chain_str = params.get("chain")
    address = params.get("address")

    if not chain_str or not address:
        return {"success": False, "error": "Both 'chain' and 'address' are required"}

    try:
        chain = Chain(chain_str)
    except ValueError:
        valid = [c.value for c in Chain]
        return {"success": False, "error": f"Invalid chain. Valid: {valid}"}

    wallet_manager.set_address(chain, address)
    return {"success": True, "chain": chain.value, "address": address}


def faucet_list_wallets(params: dict[str, Any]) -> dict[str, Any]:
    """List all registered wallet addresses."""
    _, _, wallet_manager = _ensure_initialized()
    return {"success": True, "wallets": wallet_manager.list_wallets()}


def faucet_earnings(params: dict[str, Any]) -> dict[str, Any]:
    """Get earnings report."""
    _, config, _ = _ensure_initialized()
    store = ClaimStore(config.data_dir)
    days = params.get("days", 30)

    claims = store.load_all()
    if days:
        from datetime import UTC, datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)
        claims = [c for c in claims if c.timestamp >= cutoff]

    succeeded = [c for c in claims if c.succeeded]
    total_usd = sum(c.amount_usd for c in succeeded if c.amount_usd)

    by_faucet: dict[str, float] = {}
    for c in succeeded:
        if c.amount_usd:
            by_faucet[c.faucet_name] = by_faucet.get(c.faucet_name, 0) + c.amount_usd

    return {
        "success": True,
        "period_days": days,
        "total_claims": len(claims),
        "successful_claims": len(succeeded),
        "total_earned_usd": total_usd,
        "success_rate": len(succeeded) / len(claims) if claims else 0,
        "by_faucet": by_faucet,
    }


# Tool definitions for Animus ToolRegistry registration
FAUCET_TOOLS = [
    {
        "name": "faucet_claim",
        "description": "Claim from crypto faucets. Omit faucet_name to claim from all enabled faucets.",
        "parameters": {
            "type": "object",
            "properties": {
                "faucet_name": {
                    "type": "string",
                    "description": "Specific faucet to claim from (optional — claims all if omitted)",
                },
            },
        },
        "handler": faucet_claim,
        "category": "faucet",
    },
    {
        "name": "faucet_status",
        "description": "Get health and earnings summary for all faucets.",
        "parameters": {"type": "object", "properties": {}},
        "handler": faucet_status,
        "category": "faucet",
    },
    {
        "name": "faucet_add_wallet",
        "description": "Register a wallet address for a blockchain chain.",
        "parameters": {
            "type": "object",
            "properties": {
                "chain": {
                    "type": "string",
                    "description": "Blockchain (btc, eth, base_sepolia, sui_testnet, sol, doge, ltc, matic)",
                },
                "address": {
                    "type": "string",
                    "description": "Wallet address for this chain",
                },
            },
            "required": ["chain", "address"],
        },
        "handler": faucet_add_wallet,
        "category": "faucet",
    },
    {
        "name": "faucet_list_wallets",
        "description": "List all registered wallet addresses.",
        "parameters": {"type": "object", "properties": {}},
        "handler": faucet_list_wallets,
        "category": "faucet",
    },
    {
        "name": "faucet_earnings",
        "description": "Get earnings report for a time period.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to report on (default 30)",
                },
            },
        },
        "handler": faucet_earnings,
        "category": "faucet",
    },
]
