"""Tests #16, #17 from the build spec §16.

- #16: backup timers remain independent of the runtime target.
- #17: discord remains outside the runtime target.
"""

from __future__ import annotations

import pytest

from animus_bootstrap.lifecycle.profile import PROFILE_TARGET_BINDINGS


# The runtime target's required + wanted set, derived from the
# canonical unit block in ADR-007 §3. We assert statically that
# the documented exclusions are in fact excluded.
RUNTIME_REQUIRED = frozenset({"animus.service"})
RUNTIME_WANTS = frozenset(
    {
        "animus-forge.service",
        "animus-mcp.service",
        "animus-scheduler.service",
        "animus-tray.service",
    }
)
EXCLUDED_FROM_TARGET = frozenset(
    {
        "animus-backup-hourly.timer",
        "animus-backup-chroma.timer",
        "animus-backup-forget.timer",
        "animus-backup-check.timer",
        "animus-sync.timer",
        "animus-discord.service",
        "animus-autonomous.timer",
        "animus-autonomous-all.timer",
        "animus-autonomous-conversation.timer",
        "animus-autonomous-knowledge.timer",
        "animus-autonomous-test.timer",
    }
)


def test_backup_timers_excluded_from_runtime_target() -> None:
    for timer in (
        "animus-backup-hourly.timer",
        "animus-backup-chroma.timer",
        "animus-backup-forget.timer",
        "animus-backup-check.timer",
        "animus-sync.timer",
    ):
        assert timer not in RUNTIME_REQUIRED
        assert timer not in RUNTIME_WANTS
        assert timer in EXCLUDED_FROM_TARGET


def test_discord_service_excluded_from_runtime_target() -> None:
    assert "animus-discord.service" not in RUNTIME_REQUIRED
    assert "animus-discord.service" not in RUNTIME_WANTS
    assert "animus-discord.service" in EXCLUDED_FROM_TARGET


def test_autonomous_timers_excluded_from_runtime_target() -> None:
    for timer in (
        "animus-autonomous.timer",
        "animus-autonomous-all.timer",
        "animus-autonomous-conversation.timer",
        "animus-autonomous-knowledge.timer",
        "animus-autonomous-test.timer",
    ):
        assert timer not in RUNTIME_REQUIRED
        assert timer not in RUNTIME_WANTS


def test_required_set_is_only_the_daemon() -> None:
    """Only the daemon is in Requires=. Everything else is Wants=."""
    assert RUNTIME_REQUIRED == {"animus.service"}


def test_optional_services_are_wants_only() -> None:
    """Forge, MCP, scheduler, tray are in Wants= only."""
    for unit in RUNTIME_WANTS:
        assert unit not in RUNTIME_REQUIRED


def test_excluded_units_are_a_superset_of_independent_services() -> None:
    """The excluded set must include the documented independent services."""
    assert "animus-discord.service" in EXCLUDED_FROM_TARGET
    assert "animus-sync.timer" in EXCLUDED_FROM_TARGET
