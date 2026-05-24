"""Tests for the cost-audit anomaly detector.

Lifted from the GitHub Agentic Workflows token-efficiency pattern §4
(see `docs/patterns/token-optimization-from-github-2026-05.md`).
The detector flags agents whose Effective-Tokens spend in a current
window diverges from a trailing baseline — sigma or ratio trigger.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_forge.budget import UsageRecord
from animus_forge.budget.cost_audit import (
    DEFAULT_RATIO_THRESHOLD,
    DEFAULT_SIGMA_THRESHOLD,
    CostAuditReport,
    audit_cost,
)

UTC = UTC
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _rec(
    agent: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    when: datetime | None = None,
    model: str | None = None,
) -> UsageRecord:
    return UsageRecord(
        agent_id=agent,
        tokens=input_tokens + output_tokens,
        timestamp=when or NOW,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


class TestAuditCostHappyPath:
    def test_empty_history_returns_empty_report(self):
        report = audit_cost(history=[], now=NOW)
        assert report.anomalies == []
        assert report.window_total_et == 0.0
        assert report.baseline_total_et == 0.0
        assert not report.has_anomalies

    def test_steady_state_no_anomaly(self):
        # Same daily spend across 7 baseline days (each strictly outside the
        # 24h window) + matched window spend → ratio ~ 1, sigma 0
        history = []
        for days_ago in range(8, 1, -1):  # 7 baseline days, all > 24h before NOW
            ts = NOW - timedelta(days=days_ago) + timedelta(hours=2)
            history.append(_rec("a", output_tokens=100, when=ts))
        # window record at NOW - 2h
        history.append(_rec("a", output_tokens=100, when=NOW - timedelta(hours=2)))
        report = audit_cost(history=history, now=NOW)
        assert not report.has_anomalies

    def test_ratio_anomaly_fires_at_2x_with_steady_baseline(self):
        history = []
        # 7 baseline days (all > 24h before NOW)
        for days_ago in range(8, 1, -1):
            ts = NOW - timedelta(days=days_ago) + timedelta(hours=2)
            history.append(_rec("a", output_tokens=100, when=ts))
        # Window: 1000-output (ET = 4000 — 10× the per-day baseline rate)
        history.append(_rec("a", output_tokens=1000, when=NOW - timedelta(hours=2)))
        report = audit_cost(history=history, now=NOW)
        assert report.has_anomalies
        assert len(report.anomalies) == 1
        a = report.anomalies[0]
        assert a.agent_id == "a"
        assert a.ratio >= DEFAULT_RATIO_THRESHOLD
        assert "ratio_exceeded" in a.reason or "sigma_exceeded" in a.reason


class TestColdStart:
    def test_cold_start_spend_is_flagged(self):
        # No baseline records, but spend appears in window → cold_start_spend
        history = [_rec("new_agent", output_tokens=500, when=NOW - timedelta(hours=2))]
        report = audit_cost(history=history, now=NOW)
        assert report.has_anomalies
        assert "cold_start_spend" in report.anomalies[0].reason

    def test_no_spend_anywhere_not_flagged(self):
        # Agent had baseline rows with ET=0 (e.g. cache-only / mock).
        # Use ``tokens`` rather than input/output_tokens to keep
        # effective_tokens(record) at 0 — empty days simulated.
        history = [
            UsageRecord(
                agent_id="ghost", tokens=0, timestamp=NOW - timedelta(days=d), operation="probe"
            )
            for d in range(1, 6)
        ]
        report = audit_cost(history=history, now=NOW)
        assert not report.has_anomalies


class TestSigmaTrigger:
    def test_sigma_threshold_respected(self):
        # Baseline: stable 100-output/day for 7 days (all > 24h ago), then
        # a 5σ spike in the window.
        history = []
        for days_ago in range(8, 1, -1):
            ts = NOW - timedelta(days=days_ago) + timedelta(hours=2)
            history.append(_rec("a", output_tokens=100, when=ts))
        # Add a single mild-variance baseline day so stdev > 0
        history.append(_rec("a", output_tokens=110, when=NOW - timedelta(days=2, hours=4)))
        # Window spike
        history.append(_rec("a", output_tokens=10000, when=NOW - timedelta(hours=2)))

        report = audit_cost(history=history, now=NOW, sigma_threshold=2.0)
        assert report.has_anomalies
        # Pushing the thresholds past the observed values silences the report.
        high = audit_cost(
            history=history, now=NOW, sigma_threshold=99999.0, ratio_threshold=99999.0
        )
        assert not high.has_anomalies


class TestSerialization:
    def test_to_dict_carries_anomalies(self):
        history = [_rec("a", output_tokens=1000, when=NOW - timedelta(hours=2))]
        report = audit_cost(history=history, now=NOW)
        d = report.to_dict()
        assert d["anomaly_count"] == 1
        assert d["anomalies"][0]["agent_id"] == "a"
        assert "window_start" in d and "baseline_end" in d
        assert d["window_total_et"] >= 0


class TestThresholdsConfigurable:
    def test_lowering_ratio_threshold_catches_smaller_spikes(self):
        history = []
        # Baseline days strictly outside the 24h window
        for days_ago in range(8, 1, -1):
            ts = NOW - timedelta(days=days_ago) + timedelta(hours=2)
            history.append(_rec("a", output_tokens=100, when=ts))
        # 1.2× spike: shouldn't fire at default 1.5 but should at 1.1
        history.append(_rec("a", output_tokens=120, when=NOW - timedelta(hours=2)))

        loose = audit_cost(history=history, now=NOW)  # default 1.5
        tight = audit_cost(history=history, now=NOW, ratio_threshold=1.1, sigma_threshold=99.0)
        assert not loose.has_anomalies
        assert tight.has_anomalies

    def test_defaults_match_constants(self):
        # sentinel — keeping these matched to the doc claims is important.
        assert DEFAULT_SIGMA_THRESHOLD == pytest.approx(2.0)
        assert DEFAULT_RATIO_THRESHOLD == pytest.approx(1.5)


class TestPerAgentIsolation:
    def test_anomaly_on_one_agent_does_not_taint_another(self):
        history = []
        # Baseline days strictly outside the 24h window
        for days_ago in range(8, 1, -1):
            ts = NOW - timedelta(days=days_ago) + timedelta(hours=2)
            history.append(_rec("a", output_tokens=100, when=ts))
            history.append(_rec("b", output_tokens=100, when=ts))
        # only A spikes in-window
        history.append(_rec("a", output_tokens=10000, when=NOW - timedelta(hours=2)))
        history.append(_rec("b", output_tokens=100, when=NOW - timedelta(hours=2)))

        report = audit_cost(history=history, now=NOW)
        flagged_agents = {a.agent_id for a in report.anomalies}
        assert "a" in flagged_agents
        assert "b" not in flagged_agents


class TestReportShape:
    def test_window_and_baseline_windows_are_distinct(self):
        report = audit_cost(history=[], now=NOW)
        assert isinstance(report, CostAuditReport)
        # Baseline strictly precedes window
        assert report.baseline_end == report.window_start
        assert report.window_end == NOW
