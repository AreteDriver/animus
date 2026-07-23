"""Tests for the AlertManager threshold alerting and health scoring."""

from __future__ import annotations

import time

import pytest

from animus_bootstrap.intelligence.alert_manager import AlertManager
from animus_bootstrap.intelligence.event_ledger import EventLedger


class TestThresholdChecks:
    """Alert threshold evaluation."""

    def test_no_alerts_when_quiet(self) -> None:
        ledger = EventLedger()
        mgr = AlertManager(ledger, error_rate_max=5.0, tool_failure_rate_max=3.0)
        alerts = mgr.check()
        assert alerts == []

    def test_error_rate_alert(self) -> None:
        ledger = EventLedger()
        # Inject 6 errors in the last 5 minutes = 1.2/min
        for _ in range(6):
            ledger.record("error", "test", {"msg": "boom"})
        mgr = AlertManager(ledger, error_rate_max=1.0, tool_failure_rate_max=3.0)
        alerts = mgr.check()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "error_rate"
        assert alerts[0]["severity"] in ("warning", "critical")

    def test_tool_failure_alert(self) -> None:
        ledger = EventLedger()
        for _ in range(15):
            ledger.record("tool_execution", "test", {"tool_name": "x", "success": False})
        mgr = AlertManager(ledger, error_rate_max=5.0, tool_failure_rate_max=2.0)
        alerts = mgr.check()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "tool_failure_rate"

    def test_no_alert_below_threshold(self) -> None:
        ledger = EventLedger()
        ledger.record("error", "test")
        mgr = AlertManager(ledger, error_rate_max=5.0)
        assert mgr.check() == []

    def test_cooldown_prevents_spam(self) -> None:
        ledger = EventLedger()
        for _ in range(10):
            ledger.record("error", "test")
        mgr = AlertManager(ledger, error_rate_max=1.0)
        first = mgr.check()
        assert len(first) == 1
        second = mgr.check()
        assert second == []  # Cooldown active

    def test_alert_records_to_ledger(self) -> None:
        ledger = EventLedger()
        for _ in range(10):
            ledger.record("error", "test")
        mgr = AlertManager(ledger, error_rate_max=1.0)
        mgr.check_and_record()
        alerts = ledger.query(event_type="alert")
        assert len(alerts) == 1
        assert alerts[0]["payload"]["type"] == "error_rate"


class TestHealthScore:
    """Composite health score calculation."""

    def test_perfect_health(self) -> None:
        ledger = EventLedger()
        mgr = AlertManager(ledger)
        health = mgr.get_health_score()
        assert health["score"] == 100
        assert health["status"] == "healthy"

    def test_degraded_health(self) -> None:
        ledger = EventLedger()
        # Enough errors + tool failures to pull score below 80 but above 50
        for _ in range(15):
            ledger.record("error", "test")
        for _ in range(8):
            ledger.record("tool_execution", "test", {"success": False})
        mgr = AlertManager(ledger, error_rate_max=5.0, tool_failure_rate_max=3.0)
        health = mgr.get_health_score()
        assert 50 < health["score"] < 80
        assert health["status"] == "degraded"

    def test_critical_health(self) -> None:
        ledger = EventLedger()
        for _ in range(20):
            ledger.record("error", "test")
        for _ in range(10):
            ledger.record("tool_execution", "test", {"success": False})
        mgr = AlertManager(ledger, error_rate_max=5.0, tool_failure_rate_max=3.0)
        health = mgr.get_health_score()
        assert health["score"] < 50
        assert health["status"] == "critical"

    def test_health_factors_present(self) -> None:
        ledger = EventLedger()
        ledger.record("error", "test")
        mgr = AlertManager(ledger)
        health = mgr.get_health_score()
        assert "error_rate" in health["factors"]
        assert "tool_failure_rate" in health["factors"]
        assert "recent_alerts" in health["factors"]

    def test_no_ledger_returns_unknown(self) -> None:
        mgr = AlertManager(None)
        health = mgr.get_health_score()
        assert health["score"] == 0
        assert health["status"] == "unknown"


class TestActiveAlerts:
    """Retrieving recent alert events."""

    def test_get_active_alerts(self) -> None:
        ledger = EventLedger()
        mgr = AlertManager(ledger, error_rate_max=1.0)
        for _ in range(5):
            ledger.record("error", "test")
        mgr.check_and_record()
        active = mgr.get_active_alerts()
        assert len(active) == 1
        assert active[0]["type"] == "alert"
