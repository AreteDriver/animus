"""Verdict sync proactive check — periodic sync of Verdict decisions to Core memory."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck

logger = logging.getLogger(__name__)

# Module-level references, wired at runtime via set_verdict_sync_deps()
_memory_layer = None
_sync_state: dict[str, datetime | None] = {"last_sync": None}


def set_verdict_sync_deps(
    memory_layer=None,  # noqa: ANN001
) -> None:
    """Wire runtime dependencies into the verdict sync check."""
    global _memory_layer  # noqa: PLW0603
    _memory_layer = memory_layer


async def _run_verdict_sync() -> str | None:
    """Sync recent Verdict decisions to episodic memory.

    Only syncs decisions created since the last successful sync.
    Returns a nudge message if decisions were synced, None otherwise.
    """
    if _memory_layer is None:
        logger.debug("Verdict sync skipped — no memory layer")
        return None

    try:
        from animus.integrations.arete_bridge import auto_sync_verdicts
    except ImportError:
        logger.debug("Verdict sync skipped — arete_bridge not available")
        return None

    since = _sync_state["last_sync"] or (datetime.now(UTC) - timedelta(hours=24))

    try:
        count = auto_sync_verdicts(_memory_layer, since=since)
        _sync_state["last_sync"] = datetime.now(UTC)
        if count > 0:
            logger.info("Verdict sync: %d decisions synced to memory", count)
            return f"Synced {count} verdict decisions to episodic memory."
        return None
    except Exception:
        logger.exception("Verdict sync failed")
        return None


def get_verdict_sync_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for periodic verdict sync."""
    return ProactiveCheck(
        name="verdict_sync",
        schedule="every 6h",
        checker=_run_verdict_sync,
        channels=[],
        priority="low",
        enabled=True,
    )
