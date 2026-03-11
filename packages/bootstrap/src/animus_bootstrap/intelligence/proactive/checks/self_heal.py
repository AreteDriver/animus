"""Self-healing proactive check — monitors tool failures and auto-proposes improvements."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck

logger = logging.getLogger(__name__)

# Module-level references, wired at runtime via set_self_heal_deps()
_tool_executor = None
_improvement_store = None
_cognitive_backend = None
_tool_history_store = None

# Track what we've already proposed to avoid duplicates
_proposed_areas: set[str] = set()

# Thresholds
_FAILURE_RATE_THRESHOLD = 0.3  # 30% failure rate triggers proposal
_SLOW_THRESHOLD_MS = 10_000  # 10s is slow
_MIN_EXECUTIONS = 5  # Need at least 5 executions to judge


def set_self_heal_deps(
    tool_executor=None,  # noqa: ANN001
    improvement_store=None,  # noqa: ANN001
    cognitive_backend=None,  # noqa: ANN001
    tool_history_store=None,  # noqa: ANN001
) -> None:
    """Wire runtime dependencies into the self-heal check."""
    global _tool_executor, _improvement_store  # noqa: PLW0603
    global _cognitive_backend, _tool_history_store  # noqa: PLW0603
    _tool_executor = tool_executor
    _improvement_store = improvement_store
    _cognitive_backend = cognitive_backend
    _tool_history_store = tool_history_store


def clear_proposed_areas() -> None:
    """Reset proposed areas tracking (for testing)."""
    _proposed_areas.clear()


async def _run_self_heal() -> str | None:
    """Analyze recent tool execution patterns and auto-propose improvements.

    Triggers when:
    1. A tool's failure rate exceeds 30% (with >= 5 executions)
    2. A tool's average response time exceeds 10s
    3. The same error pattern appears 3+ times

    Returns nudge text if proposals were created, None otherwise.
    """
    if _tool_executor is None:
        logger.debug("Self-heal skipped — no tool executor")
        return None

    # Get recent history (last 24h worth)
    history = _tool_executor.get_history(limit=200)
    if not history:
        logger.debug("Self-heal skipped — no tool history")
        return None

    # Filter to last 24h
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    recent = [r for r in history if r.timestamp >= cutoff]
    if len(recent) < _MIN_EXECUTIONS:
        logger.debug("Self-heal skipped — only %d recent executions", len(recent))
        return None

    proposals_created = 0

    # --- Check 1: High failure rate per tool ---
    tool_stats: dict[str, dict] = {}
    for r in recent:
        if r.tool_name not in tool_stats:
            tool_stats[r.tool_name] = {"total": 0, "failures": 0, "total_ms": 0.0, "errors": []}
        stats = tool_stats[r.tool_name]
        stats["total"] += 1
        stats["total_ms"] += r.duration_ms
        if not r.success:
            stats["failures"] += 1
            stats["errors"].append(r.output[:200])

    for tool_name, stats in tool_stats.items():
        if stats["total"] < _MIN_EXECUTIONS:
            continue

        failure_rate = stats["failures"] / stats["total"]
        area = f"tool:{tool_name}"

        # High failure rate
        if failure_rate >= _FAILURE_RATE_THRESHOLD and area not in _proposed_areas:
            error_sample = "; ".join(stats["errors"][:3])
            description = (
                f"Tool '{tool_name}' has {failure_rate:.0%} failure rate "
                f"({stats['failures']}/{stats['total']} in last 24h). "
                f"Sample errors: {error_sample}"
            )
            proposal_id = await _create_proposal(area, description)
            if proposal_id:
                _proposed_areas.add(area)
                proposals_created += 1

        # Slow execution
        avg_ms = stats["total_ms"] / stats["total"]
        slow_area = f"perf:{tool_name}"
        if avg_ms > _SLOW_THRESHOLD_MS and slow_area not in _proposed_areas:
            description = (
                f"Tool '{tool_name}' averaging {avg_ms:.0f}ms per call "
                f"({stats['total']} calls in last 24h). "
                f"Threshold: {_SLOW_THRESHOLD_MS}ms."
            )
            proposal_id = await _create_proposal(slow_area, description)
            if proposal_id:
                _proposed_areas.add(slow_area)
                proposals_created += 1

    # --- Check 2: Repeated error patterns ---
    error_counts: dict[str, int] = {}
    for r in recent:
        if not r.success:
            # Normalize error to first 100 chars for grouping
            error_key = f"{r.tool_name}:{r.output[:100]}"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1

    for error_key, count in error_counts.items():
        if count >= 3:
            area = f"error_pattern:{error_key[:50]}"
            if area not in _proposed_areas:
                tool_name = error_key.split(":", 1)[0]
                error_msg = error_key.split(":", 1)[1] if ":" in error_key else error_key
                description = f"Repeated error ({count}x in 24h) in tool '{tool_name}': {error_msg}"
                proposal_id = await _create_proposal(area, description)
                if proposal_id:
                    _proposed_areas.add(area)
                    proposals_created += 1

    if proposals_created:
        msg = (
            f"Self-heal: {proposals_created} improvement proposal(s) "
            f"auto-generated from tool failures."
        )
        logger.info(msg)
        return msg

    return None


async def _create_proposal(area: str, description: str) -> int | None:
    """Create an improvement proposal, optionally with AI analysis."""
    if _improvement_store is None:
        logger.warning("Self-heal: no improvement store — cannot persist proposal")
        return None

    proposal = {
        "area": area,
        "description": description,
        "status": "proposed",
        "timestamp": datetime.now(UTC).isoformat(),
        "analysis": None,
        "patch": None,
    }

    # Use cognitive backend for AI analysis if available
    if _cognitive_backend is not None:
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"You are analyzing the Animus AI system for self-healing.\n"
                        f"Area: {area}\n"
                        f"Issue: {description}\n\n"
                        f"Diagnose the root cause and propose a fix. Consider:\n"
                        f"1. Is this a configuration issue? (suggest YAML config change)\n"
                        f"2. Is this a prompt issue? (suggest system prompt adjustment)\n"
                        f"3. Is this a tool parameter issue? (suggest default changes)\n"
                        f"4. Is this an external dependency issue? (suggest fallback)\n\n"
                        f"Be specific and actionable. If you can express the fix as a "
                        f"config change, provide the exact YAML path and value."
                    ),
                }
            ]
            response = await _cognitive_backend.generate_response(
                messages=messages,
                system_prompt="You are an AI system self-healing analyst. Be concise and specific.",
                max_tokens=512,
            )
            proposal["analysis"] = response
        except Exception:
            logger.exception("Self-heal AI analysis failed")
            proposal["analysis"] = f"Auto-detected issue (AI analysis unavailable): {description}"
    else:
        proposal["analysis"] = f"Auto-detected issue: {description}"

    proposal_id = _improvement_store.save(proposal)
    logger.info("Self-heal proposal #%d created: %s", proposal_id, area)
    return proposal_id


def get_self_heal_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for periodic self-healing."""
    return ProactiveCheck(
        name="self_heal",
        schedule="0 */6 * * *",  # Every 6 hours
        checker=_run_self_heal,
        channels=[],  # Internal — proposals visible in dashboard
        priority="normal",
        enabled=True,
    )
