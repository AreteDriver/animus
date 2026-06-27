"""Tests for Animus MCP tool definitions."""

from __future__ import annotations

from pathlib import Path

import pytest

import animus_arbitrage.tools.arbitrage_tools as tools_module
from animus_arbitrage.tools.arbitrage_tools import (
    ARBITRAGE_TOOLS,
    arbitrage_config,
    arbitrage_execute,
    arbitrage_scan,
    arbitrage_status,
)


@pytest.fixture(autouse=True)
def reset_tool_globals():
    """Reset global state between tests."""
    tools_module._config = None
    tools_module._detector = None
    tools_module._executor = None
    tools_module._tracker = None
    tools_module._store = None
    tools_module._risk = None
    yield
    tools_module._config = None
    tools_module._detector = None
    tools_module._executor = None
    tools_module._tracker = None
    tools_module._store = None
    tools_module._risk = None


class TestArbitrageTools:
    def test_tools_list(self):
        assert len(ARBITRAGE_TOOLS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in ARBITRAGE_TOOLS}
        assert names == {
            "arbitrage_scan",
            "arbitrage_status",
            "arbitrage_config",
            "arbitrage_execute",
        }

    def test_all_have_category(self):
        for tool in ARBITRAGE_TOOLS:
            assert tool["category"] == "arbitrage"

    def test_all_have_handler(self):
        for tool in ARBITRAGE_TOOLS:
            assert callable(tool["handler"])

    def test_all_have_description(self):
        for tool in ARBITRAGE_TOOLS:
            assert len(tool["description"]) > 0

    def test_all_have_parameters(self):
        for tool in ARBITRAGE_TOOLS:
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"


class TestArbitrageScan:
    def test_scan_returns_success(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_scan({})
        assert result["success"] is True
        assert "opportunities" in result
        assert "count" in result

    def test_scan_with_trade_size(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_scan({"trade_size_usd": 500.0})
        assert result["success"] is True


class TestArbitrageStatus:
    def test_status_returns_success(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_status({})
        assert result["success"] is True
        assert "pnl" in result
        assert "risk" in result
        assert "recent_trades" in result

    def test_status_with_limit(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_status({"limit": 5})
        assert result["success"] is True


class TestArbitrageConfig:
    def test_config_view(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_config({})
        assert result["success"] is True
        assert "config" in result

    def test_config_update(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_config({"updates": {"min_spread_pct": 0.5}})
        assert result["success"] is True
        assert "Updated" in result["message"]

    def test_config_update_nonexistent_key(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_config({"updates": {"nonexistent_key": 123}})
        assert result["success"] is True


class TestArbitrageExecute:
    def test_execute_blocked_dry_run(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        result = arbitrage_execute({"opportunity_id": "test123"})
        assert result["success"] is False
        assert "DRY_RUN" in result["error"]

    def test_execute_missing_id(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        monkeypatch.setenv("ARBITRAGE_EXECUTION_MODE", "live")
        result = arbitrage_execute({})
        assert result["success"] is False
        assert "required" in result["error"]

    def test_execute_opportunity_not_found(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        monkeypatch.setenv("ARBITRAGE_EXECUTION_MODE", "live")
        result = arbitrage_execute({"opportunity_id": "nonexistent"})
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_execute_found_opportunity(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Execute with a stored opportunity in live mode."""
        from animus_arbitrage.store import ArbitrageStore

        from .conftest import make_opportunity

        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        monkeypatch.setenv("ARBITRAGE_EXECUTION_MODE", "live")

        # Store an opportunity first
        store = ArbitrageStore(tmp_data_dir)
        opp = make_opportunity(trade_size=50.0)
        store.record_opportunity(opp)

        result = arbitrage_execute({"opportunity_id": opp.opportunity_id})
        assert result["success"] is True
        assert "result" in result

    def test_execute_with_config_file(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Test initialization path when config file exists."""
        import yaml

        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        config_file = tmp_data_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"risk": {"min_spread_pct": 0.1}}, f)

        result = arbitrage_scan({})
        assert result["success"] is True

    def test_ensure_initialized_caches(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Second call should reuse cached instances."""
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_data_dir))
        from animus_arbitrage.tools.arbitrage_tools import _ensure_initialized

        c1, d1, e1, t1, r1 = _ensure_initialized()
        c2, d2, e2, t2, r2 = _ensure_initialized()
        assert c1 is c2
        assert d1 is d2
