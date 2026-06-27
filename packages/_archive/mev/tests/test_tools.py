"""Tests for MEV MCP tools."""

from __future__ import annotations

import pytest

import animus_mev.tools.mev_tools as tools_mod
from animus_mev.tools.mev_tools import (
    MEV_TOOLS,
    mev_monitor,
    mev_opportunities,
    mev_status,
)


@pytest.fixture(autouse=True)
def _reset_tools_state(tmp_data_dir, monkeypatch):
    """Reset tool module globals before each test."""
    monkeypatch.setenv("MEV_DATA_DIR", str(tmp_data_dir))
    tools_mod._config = None
    tools_mod._store = None
    tools_mod._tracker = None
    yield
    tools_mod._config = None
    tools_mod._store = None
    tools_mod._tracker = None


class TestMEVTools:
    def test_tool_count(self):
        assert len(MEV_TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in MEV_TOOLS}
        assert names == {"mev_monitor", "mev_opportunities", "mev_status"}

    def test_tool_categories(self):
        for tool in MEV_TOOLS:
            assert tool["category"] == "mev"

    def test_all_have_handlers(self):
        for tool in MEV_TOOLS:
            assert callable(tool["handler"])


class TestMevMonitor:
    def test_status(self):
        result = mev_monitor({"action": "status"})
        assert result["success"] is True
        assert result["mode"] == "dry_run"
        assert result["sandwich_execution"] == "BLOCKED"
        assert len(result["enabled_chains"]) >= 1

    def test_default_action(self):
        result = mev_monitor({})
        assert result["success"] is True

    def test_unknown_action(self):
        result = mev_monitor({"action": "invalid"})
        assert result["success"] is False
        assert "Unknown action" in result["error"]


class TestMevOpportunities:
    def test_empty(self):
        result = mev_opportunities({})
        assert result["success"] is True
        assert result["count"] == 0
        assert result["opportunities"] == []

    def test_with_limit(self):
        result = mev_opportunities({"limit": 5})
        assert result["success"] is True


class TestMevStatus:
    def test_status(self):
        result = mev_status({})
        assert result["success"] is True
        assert "pnl" in result
        assert result["mode"] == "dry_run"
        assert "recent_results" in result

    def test_pnl_fields(self):
        result = mev_status({})
        pnl = result["pnl"]
        assert "total_opportunities" in pnl
        assert "net_profit_usd" in pnl
        assert "win_rate" in pnl
