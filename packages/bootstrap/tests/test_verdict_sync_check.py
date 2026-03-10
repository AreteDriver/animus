"""Tests for verdict_sync proactive check."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from animus_bootstrap.intelligence.proactive.checks.verdict_sync import (
    _run_verdict_sync,
    get_verdict_sync_check,
    set_verdict_sync_deps,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


class TestVerdictSyncCheck:
    """Tests for the verdict sync proactive check."""

    def test_check_config(self):
        check = get_verdict_sync_check()
        assert check.name == "verdict_sync"
        assert check.schedule == "every 6h"
        assert check.priority == "low"
        assert check.enabled is True

    def test_skips_without_memory_layer(self):
        set_verdict_sync_deps(memory_layer=None)
        result = _run(_run_verdict_sync())
        assert result is None

    def test_returns_none_when_zero_synced(self):
        memory = MagicMock()
        set_verdict_sync_deps(memory_layer=memory)

        mock_module = MagicMock()
        mock_module.auto_sync_verdicts.return_value = 0
        with patch.dict(
            "sys.modules",
            {"animus.integrations.arete_bridge": mock_module},
        ):
            result = _run(_run_verdict_sync())

        assert result is None
        set_verdict_sync_deps(memory_layer=None)

    def test_exception_returns_none(self):
        memory = MagicMock()
        set_verdict_sync_deps(memory_layer=memory)

        mock_module = MagicMock()
        mock_module.auto_sync_verdicts.side_effect = RuntimeError("DB locked")
        with patch.dict(
            "sys.modules",
            {"animus.integrations.arete_bridge": mock_module},
        ):
            result = _run(_run_verdict_sync())

        assert result is None
        set_verdict_sync_deps(memory_layer=None)

    def test_syncs_with_message(self):
        memory = MagicMock()
        set_verdict_sync_deps(memory_layer=memory)

        mock_module = MagicMock()
        mock_module.auto_sync_verdicts.return_value = 3
        with patch.dict(
            "sys.modules",
            {"animus.integrations.arete_bridge": mock_module},
        ):
            result = _run(_run_verdict_sync())

        assert result is not None
        assert "3" in result
        set_verdict_sync_deps(memory_layer=None)
