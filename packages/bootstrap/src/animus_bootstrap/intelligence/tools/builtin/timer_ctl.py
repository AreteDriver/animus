"""Timer management tools — create, list, cancel scheduled tasks at runtime."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# In-memory registry for dynamic timers (supplementing ProactiveEngine).
# Each entry: {"name": str, "schedule": str, "action": str, "created": str}
_dynamic_timers: list[dict] = []

# ProactiveEngine reference — set at registration time by runtime
_proactive_engine = None

# Persistent store reference — set at runtime
_timer_store = None


def set_proactive_engine(engine: object) -> None:
    """Wire the live ProactiveEngine for timer registration."""
    global _proactive_engine  # noqa: PLW0603
    _proactive_engine = engine


def set_timer_store(store: object) -> None:
    """Wire the persistent timer store."""
    global _timer_store  # noqa: PLW0603
    _timer_store = store


def get_dynamic_timers() -> list[dict]:
    """Return all dynamic timers (for testing/inspection)."""
    return list(_dynamic_timers)


def clear_dynamic_timers() -> None:
    """Clear all dynamic timers."""
    _dynamic_timers.clear()


def restore_timers() -> int:
    """Restore timers from persistent store into in-memory registry.

    Called during runtime startup to re-register saved timers with
    the ProactiveEngine. Returns the number of timers restored.
    """
    if _timer_store is None:
        return 0

    saved = _timer_store.list_all()
    count = 0
    for entry in saved:
        # Add to in-memory list (avoid duplicates)
        if any(t["name"] == entry["name"] for t in _dynamic_timers):
            continue
        _dynamic_timers.append(entry)

        # Register with ProactiveEngine if available
        if _proactive_engine is not None:
            _register_with_engine(
                entry["name"], entry["schedule"], entry["action"], entry["channels"]
            )

        count += 1

    if count:
        logger.info("Restored %d timers from persistent store", count)
    return count


def _register_with_engine(name: str, schedule: str, action: str, channels: list[str]) -> None:
    """Register a timer as a ProactiveCheck."""
    from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck

    async def _timer_checker() -> str | None:
        return action

    check = ProactiveCheck(
        name=f"timer:{name}",
        schedule=schedule,
        checker=_timer_checker,
        channels=channels,
        priority="normal",
    )
    _proactive_engine.register_check(check)


async def _timer_create(name: str, schedule: str, action: str, channels: str = "") -> str:
    """Create a new timer that fires on a schedule.

    The ``action`` is a text message that will be sent as a nudge when the
    timer fires.  Schedule supports cron expressions (``0 9 * * 1-5``) and
    intervals (``every 30m``, ``every 2h``).
    """
    # Validate schedule format
    from animus_bootstrap.intelligence.proactive.schedule import ScheduleParser

    if ScheduleParser.is_interval(schedule):
        try:
            ScheduleParser.parse_interval(schedule)
        except ValueError as exc:
            return f"Invalid interval schedule: {exc}"
    else:
        try:
            ScheduleParser.next_cron_fire(schedule)
        except (ValueError, IndexError) as exc:
            return f"Invalid cron schedule: {exc}"

    channel_list = [c.strip() for c in channels.split(",") if c.strip()] if channels else []
    created = datetime.now(UTC).isoformat()

    timer_entry = {
        "name": name,
        "schedule": schedule,
        "action": action,
        "channels": channel_list,
        "created": created,
    }
    _dynamic_timers.append(timer_entry)

    # Persist to store
    if _timer_store is not None:
        _timer_store.save(name, schedule, action, channel_list, created)

    # Register with ProactiveEngine if available
    if _proactive_engine is not None:
        _register_with_engine(name, schedule, action, channel_list)
        logger.info("Timer '%s' registered with ProactiveEngine", name)
        return f"Timer '{name}' created and registered (schedule: {schedule})"

    logger.info("Timer '%s' created (no ProactiveEngine — stored only)", name)
    return f"Timer '{name}' created (schedule: {schedule}, not live — no ProactiveEngine)"


async def _timer_list() -> str:
    """List all dynamic timers and their status."""
    if not _dynamic_timers:
        engine_checks = ""
        if _proactive_engine is not None:
            checks = _proactive_engine.list_checks()
            timer_checks = [c for c in checks if c.name.startswith("timer:")]
            if timer_checks:
                engine_checks = "\n\nLive ProactiveEngine timers:\n" + "\n".join(
                    f"  - {c.name} ({c.schedule}, enabled={c.enabled})" for c in timer_checks
                )
        return f"No dynamic timers registered.{engine_checks}"

    lines = []
    for t in _dynamic_timers:
        live = ""
        if _proactive_engine is not None:
            checks = _proactive_engine.list_checks()
            is_live = any(c.name == f"timer:{t['name']}" for c in checks)
            live = " [LIVE]" if is_live else " [NOT LIVE]"
        lines.append(
            f'  - {t["name"]}: {t["schedule"]} → "{t["action"]}"{live} (created: {t["created"]})'
        )

    return f"Dynamic timers ({len(_dynamic_timers)}):\n" + "\n".join(lines)


async def _timer_cancel(name: str) -> str:
    """Cancel a dynamic timer by name."""
    # Remove from local registry
    found = False
    for i, t in enumerate(_dynamic_timers):
        if t["name"] == name:
            _dynamic_timers.pop(i)
            found = True
            break

    # Remove from persistent store
    if _timer_store is not None:
        _timer_store.remove(name)

    # Unregister from ProactiveEngine
    if _proactive_engine is not None:
        _proactive_engine.unregister_check(f"timer:{name}")

    if found:
        logger.info("Timer '%s' cancelled", name)
        return f"Timer '{name}' cancelled"
    return f"Timer '{name}' not found"


async def _timer_update(
    name: str,
    schedule: str | None = None,
    action: str | None = None,
    channels: str | None = None,
) -> str:
    """Update an existing timer's schedule, action, or channels.

    Only provided fields are changed; others remain as-is.
    """
    # Find the timer
    timer = None
    for t in _dynamic_timers:
        if t["name"] == name:
            timer = t
            break

    if timer is None:
        return f"Timer '{name}' not found"

    # Validate new schedule if provided
    if schedule is not None:
        from animus_bootstrap.intelligence.proactive.schedule import ScheduleParser

        if ScheduleParser.is_interval(schedule):
            try:
                ScheduleParser.parse_interval(schedule)
            except ValueError as exc:
                return f"Invalid interval schedule: {exc}"
        else:
            try:
                ScheduleParser.next_cron_fire(schedule)
            except (ValueError, IndexError) as exc:
                return f"Invalid cron schedule: {exc}"
        timer["schedule"] = schedule

    if action is not None:
        timer["action"] = action

    if channels is not None:
        timer["channels"] = (
            [c.strip() for c in channels.split(",") if c.strip()]
            if channels
            else []
        )

    # Update persistent store
    if _timer_store is not None:
        _timer_store.save(
            timer["name"],
            timer["schedule"],
            timer["action"],
            timer["channels"],
            timer["created"],
        )

    # Re-register with ProactiveEngine if available
    if _proactive_engine is not None:
        _proactive_engine.unregister_check(f"timer:{name}")
        _register_with_engine(
            name, timer["schedule"], timer["action"], timer["channels"]
        )

    changes = []
    if schedule is not None:
        changes.append(f"schedule={schedule}")
    if action is not None:
        changes.append(f"action={action}")
    if channels is not None:
        changes.append(f"channels={timer['channels']}")

    logger.info("Timer '%s' updated: %s", name, ", ".join(changes))
    return f"Timer '{name}' updated: {', '.join(changes)}"


async def _timer_fire(name: str) -> str:
    """Manually fire a timer immediately (for testing)."""
    if _proactive_engine is None:
        return "No ProactiveEngine available — cannot fire timer"

    check_name = f"timer:{name}"
    try:
        nudge = await _proactive_engine.run_check(check_name)
        if nudge:
            return f"Timer '{name}' fired: {nudge.text}"
        return f"Timer '{name}' fired but produced no output"
    except KeyError:
        return f"Timer '{name}' not found in ProactiveEngine"


def get_timer_tools() -> list[ToolDefinition]:
    """Return timer management tool definitions."""
    return [
        ToolDefinition(
            name="timer_create",
            description=(
                "Create a scheduled timer. Supports cron expressions "
                "('0 9 * * 1-5') and intervals ('every 30m'). "
                "The action text is sent as a nudge when the timer fires."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for this timer.",
                    },
                    "schedule": {
                        "type": "string",
                        "description": "Cron expression or interval (e.g. 'every 30m').",
                    },
                    "action": {
                        "type": "string",
                        "description": "Message to send when the timer fires.",
                    },
                    "channels": {
                        "type": "string",
                        "description": "Comma-separated channel names for delivery.",
                        "default": "",
                    },
                },
                "required": ["name", "schedule", "action"],
            },
            handler=_timer_create,
            category="timer",
        ),
        ToolDefinition(
            name="timer_list",
            description="List all dynamic timers and their status.",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=_timer_list,
            category="timer",
        ),
        ToolDefinition(
            name="timer_cancel",
            description="Cancel a dynamic timer by name.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the timer to cancel.",
                    },
                },
                "required": ["name"],
            },
            handler=_timer_cancel,
            category="timer",
        ),
        ToolDefinition(
            name="timer_update",
            description=(
                "Update an existing timer's schedule, action, or channels. "
                "Only provided fields are changed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the timer to update.",
                    },
                    "schedule": {
                        "type": "string",
                        "description": "New cron expression or interval.",
                    },
                    "action": {
                        "type": "string",
                        "description": "New action message.",
                    },
                    "channels": {
                        "type": "string",
                        "description": "New comma-separated channel list.",
                    },
                },
                "required": ["name"],
            },
            handler=_timer_update,
            category="timer",
        ),
        ToolDefinition(
            name="timer_fire",
            description="Manually fire a timer immediately (for testing/debugging).",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the timer to fire.",
                    },
                },
                "required": ["name"],
            },
            handler=_timer_fire,
            category="timer",
        ),
    ]
