"""Optional ChainLog integration for tracing agent actions on-chain.

Provides a lazy-initialized ChainLog instance gated by the
CHAINLOG_CONTRACT_ADDRESS environment variable. If chainlog is not
installed or env vars are missing, all operations gracefully no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from chainlog import ChainLog
    from chainlog.types import ChainLogConfig, TraceResult

    HAS_CHAINLOG = True
except ImportError:
    HAS_CHAINLOG = False
    ChainLog = None  # type: ignore[assignment,misc]
    ChainLogConfig = None  # type: ignore[assignment,misc]
    TraceResult = None  # type: ignore[assignment,misc]

# Module-level singleton
_chainlog_instance: ChainLog | None = None


def get_chainlog() -> ChainLog | None:
    """Get or create the ChainLog singleton.

    Returns a ChainLog instance if:
    - chainlog package is installed
    - CHAINLOG_CONTRACT_ADDRESS env var is set

    Otherwise returns None (tracing disabled).
    """
    global _chainlog_instance

    if not HAS_CHAINLOG:
        return None

    if _chainlog_instance is not None:
        return _chainlog_instance

    contract = os.environ.get("CHAINLOG_CONTRACT_ADDRESS")
    if not contract:
        return None

    config = ChainLogConfig(
        contract_address=contract,
        private_key=os.environ.get("CHAINLOG_PRIVATE_KEY", ""),
        rpc_url=os.environ.get("CHAINLOG_RPC_URL"),
        db_path=os.environ.get("CHAINLOG_DB_PATH", "./forge-chainlog.db"),
    )

    try:
        _chainlog_instance = ChainLog(config)
        logger.info("ChainLog tracing enabled (contract=%s)", contract[:10] + "...")
    except Exception as exc:
        logger.warning("ChainLog init failed: %s", exc)
        return None

    return _chainlog_instance


async def trace_agent_action(
    agent_id: str,
    action_type: str,
    input_data: dict[str, Any],
    output: str,
    duration_ms: int,
    model_id: str = "unknown",
) -> TraceResult | None:
    """Trace an already-completed agent action to ChainLog.

    This is a post-execution trace — the action has already run,
    we just record its fingerprint. Returns None if tracing is
    disabled or fails.

    Args:
        agent_id: Agent role (e.g. "forge-builder").
        action_type: Action category (e.g. "delegation", "workflow_step").
        input_data: Input context (will be hashed, not stored on-chain).
        output: Agent output text.
        duration_ms: How long the action took.
        model_id: Model identifier.

    Returns:
        TraceResult if traced, None otherwise.
    """
    cl = get_chainlog()
    if cl is None:
        return None

    try:
        result = await cl.trace_action(
            agent_id=f"forge-{agent_id}",
            action_type=action_type,
            input_data=input_data,
            execute=_make_noop_coroutine(output),
            model_id=model_id,
        )
        logger.debug(
            "ChainLog traced %s: hash=%s stored=%s",
            agent_id,
            result.action_hash[:16] + "...",
            result.stored,
        )
        return result
    except Exception as exc:
        logger.warning("ChainLog trace failed for %s: %s", agent_id, exc)
        return None


def _make_noop_coroutine(value: Any):
    """Create a coroutine factory that returns a pre-computed value."""

    async def _noop():
        return value

    return _noop


def reset_chainlog() -> None:
    """Reset the singleton (for testing)."""
    global _chainlog_instance
    if _chainlog_instance is not None:
        try:
            _chainlog_instance.close()
        except Exception:
            pass
    _chainlog_instance = None
