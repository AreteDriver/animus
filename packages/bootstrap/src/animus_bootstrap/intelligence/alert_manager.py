"""Alert manager — threshold-based alerting for the Operations Center.

Checks event ledger rates and emits ``alert`` events when thresholds breach.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class AlertManager:
    """Monitor event rates and emit alert events when thresholds breach.

    Thresholds
    ----------
    ``error_rate_max``          – errors/min before alerting (default 5.0).
    ``tool_failure_rate_max``   – failed tool execs/min before alerting (default 3.0).
    ``window_seconds``          – Time window for rate calculation (default 300s = 5min).
    """

    def __init__(
        self,
        event_ledger: Any,
        error_rate_max: float = 5.0,
        tool_failure_rate_max: float = 3.0,
        window_seconds: float = 300,
    ) -> None:
        self._ledger = event_ledger
        self._error_rate_max = error_rate_max
        self._tool_failure_rate_max = tool_failure_rate_max
        self._window_seconds = window_seconds
        self._last_alert_time: dict[str, float] = {}
        self._cooldown_seconds = 60.0  # Don't re-alert for the same condition within 60s

    def check(self) -> list[dict[str, Any]]:
        """Evaluate all thresholds and return any triggered alerts.

        Each alert is a dict with ``type``, ``message``, ``severity``,
        and ``rate``.  If the ledger is not available, returns an empty list.
        """
        if self._ledger is None:
            return []

        alerts: list[dict[str, Any]] = []
        now = time.time()

        # Error rate check
        error_rate = self._ledger.get_error_rate(self._window_seconds)
        if error_rate >= self._error_rate_max:
            if self._can_alert("error_rate"):
                alerts.append(
                    {
                        "type": "error_rate",
                        "message": f"Error rate {error_rate}/min exceeds threshold {self._error_rate_max}/min",
                        "severity": "critical" if error_rate >= self._error_rate_max * 2 else "warning",
                        "rate": error_rate,
                        "threshold": self._error_rate_max,
                    }
                )
                self._last_alert_time["error_rate"] = now

        # Tool failure rate check
        fail_rate = self._ledger.get_tool_failure_rate(self._window_seconds)
        if fail_rate >= self._tool_failure_rate_max:
            if self._can_alert("tool_failure_rate"):
                alerts.append(
                    {
                        "type": "tool_failure_rate",
                        "message": f"Tool failure rate {fail_rate}/min exceeds threshold {self._tool_failure_rate_max}/min",
                        "severity": "critical" if fail_rate >= self._tool_failure_rate_max * 2 else "warning",
                        "rate": fail_rate,
                        "threshold": self._tool_failure_rate_max,
                    }
                )
                self._last_alert_time["tool_failure_rate"] = now

        return alerts

    def check_and_record(self) -> list[dict[str, Any]]:
        """Run :meth:`check` and record any alerts to the event ledger."""
        alerts = self.check()
        for alert in alerts:
            if self._ledger is not None:
                self._ledger.record(
                    "alert",
                    "alert_manager",
                    alert,
                )
            logger.warning("Alert triggered: %s", alert["message"])
        return alerts

    def _can_alert(self, key: str) -> bool:
        """Return True if enough time has passed since the last alert of this type."""
        last = self._last_alert_time.get(key)
        if last is None:
            return True
        return time.time() - last >= self._cooldown_seconds

    def get_active_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent alert events from the ledger."""
        if self._ledger is None:
            return []
        return self._ledger.query(event_type="alert", limit=limit)

    def get_health_score(self) -> dict[str, Any]:
        """Calculate a composite system health score (0–100).

        Factors
        -------
        - Error rate (weight 40%)
        - Tool failure rate (weight 30%)
        - Recent alert count (weight 30%)
        """
        if self._ledger is None:
            return {"score": 0, "status": "unknown", "factors": {}}

        error_rate = self._ledger.get_error_rate(self._window_seconds)
        fail_rate = self._ledger.get_tool_failure_rate(self._window_seconds)
        alerts = self._ledger.query(event_type="alert", limit=100)
        recent_alerts = [a for a in alerts if time.time() - a["timestamp"] <= self._window_seconds]

        # Normalize each factor to 0–100 (lower is worse)
        error_score = max(0, 100 - (error_rate / self._error_rate_max) * 100)
        fail_score = max(0, 100 - (fail_rate / self._tool_failure_rate_max) * 100)
        alert_score = max(0, 100 - len(recent_alerts) * 10)

        # Weighted composite
        score = int(error_score * 0.4 + fail_score * 0.3 + alert_score * 0.3)
        score = max(0, min(100, score))

        if score >= 80:
            status = "healthy"
        elif score >= 50:
            status = "degraded"
        else:
            status = "critical"

        return {
            "score": score,
            "status": status,
            "factors": {
                "error_rate": round(error_rate, 2),
                "tool_failure_rate": round(fail_rate, 2),
                "recent_alerts": len(recent_alerts),
            },
        }
