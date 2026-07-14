"""Tests for Citizen 005 — The Session Steward.

Covers all 8 heuristics, policy diff generation, proposal creation,
and daemon integration.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from animus.citizens.proposal import ProposalStatus
from animus.citizens.session_steward import (
    EfficiencyPattern,
    PolicyDiff,
    SessionAuditReport,
    SessionStewardCitizen,
)


# ── Mock Telemetry ────────────────────────────────────────────────


@dataclass
class MockTelemetryEvent:
    session_id: str
    event_name: str
    utilization_percent: float = 0.0
    elapsed_seconds: float = 0.0
    turns: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    # New fields for H4–H8
    model_name: str = ""
    summary_quality_score: float = 0.0
    correction_count: int = 0
    user_turns: int = 0
    vague_prompt_count: int = 0
    tool_calls: int = 0


class MockTelemetryProvider:
    """Fake session controller for testing heuristics."""

    def __init__(self, events: list[MockTelemetryEvent]):
        self._events = events

    def get_telemetry(self, session_id: str | None = None) -> list[MockTelemetryEvent]:
        if session_id is None:
            return self._events
        return [e for e in self._events if e.session_id == session_id]

    def get_summary_stats(self) -> dict:
        return {"total_sessions": len({e.session_id for e in self._events})}


# ── Basic Citizen Tests ───────────────────────────────────────────


class TestSessionStewardInit:
    def test_init_defaults(self):
        steward = SessionStewardCitizen()
        assert steward.min_sessions == 5
        assert steward.analysis_window_hours == 24.0
        assert steward.persistence_dir.exists()

    def test_init_custom(self):
        with tempfile.TemporaryDirectory() as td:
            steward = SessionStewardCitizen(
                min_sessions=10,
                analysis_window_hours=48.0,
                persistence_dir=td,
            )
            assert steward.min_sessions == 10
            assert steward.analysis_window_hours == 48.0


class TestH1TimerWaste:
    def test_detects_low_utilization(self):
        steward = SessionStewardCitizen(min_sessions=2)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                elapsed_seconds=1200,
                timestamp=now,
                message="timer expired",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=55.0,
                elapsed_seconds=1300,
                timestamp=now,
                message="timer expired",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h1 = [p for p in patterns if p.heuristic == "H1"]
        assert len(h1) == 1
        assert h1[0].severity == "medium"
        assert "52.5%" in h1[0].description
        assert "suggested_timer" in h1[0].data

    def test_no_detection_when_utilization_high(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                message="timer expired",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H1" for p in patterns)

    def test_no_detection_without_timer_message(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=now,
                message="token utilization reached",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H1" for p in patterns)


class TestH2ThresholdTight:
    def test_detects_high_utilization(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=99.0,
                timestamp=now,
                message="token utilization reached",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=98.5,
                timestamp=now,
                message="token utilization reached",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h2 = [p for p in patterns if p.heuristic == "H2"]
        assert len(h2) == 1
        assert h2[0].severity == "high"
        assert "98.7" in h2[0].description or "98.8" in h2[0].description

    def test_no_detection_when_utilization_normal(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=95.0,
                timestamp=now,
                message="token utilization reached",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H2" for p in patterns)


class TestH3RestartFatigue:
    def test_detects_restart_clusters(self):
        steward = SessionStewardCitizen(min_sessions=2)
        base = datetime.now(timezone.utc)
        events = [
            # Need wrapup events to pass min_sessions threshold
            MockTelemetryEvent(
                session_id="w1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=base,
                message="timer expired",
            ),
            MockTelemetryEvent(
                session_id="w2",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=base,
                message="timer expired",
            ),
            # Restart cluster
            MockTelemetryEvent(
                session_id="s1",
                event_name="RESTARTING",
                timestamp=base,
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="RESTARTING",
                timestamp=base + timedelta(minutes=10),
            ),
            MockTelemetryEvent(
                session_id="s3",
                event_name="RESTARTING",
                timestamp=base + timedelta(minutes=20),
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h3 = [p for p in patterns if p.heuristic == "H3"]
        assert len(h3) == 1
        assert h3[0].severity == "medium"
        assert "3 restarts" in h3[0].description

    def test_no_detection_with_few_restarts(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="RESTARTING",
                timestamp=now,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H3" for p in patterns)


class TestH4ModelDrift:
    def test_detects_utilization_decline(self):
        steward = SessionStewardCitizen(min_sessions=2)
        now = datetime.now(timezone.utc)
        events = [
            # Early wrapups: high utilization
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=85.0,
                timestamp=now - timedelta(hours=6),
                model_name="qwen2.5-coder:14b",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=88.0,
                timestamp=now - timedelta(hours=5),
                model_name="qwen2.5-coder:14b",
            ),
            # Recent wrapups: low utilization (drift)
            MockTelemetryEvent(
                session_id="s3",
                event_name="WRAPPING_UP",
                utilization_percent=55.0,
                timestamp=now - timedelta(hours=1),
                model_name="qwen2.5-coder:14b",
            ),
            MockTelemetryEvent(
                session_id="s4",
                event_name="WRAPPING_UP",
                utilization_percent=52.0,
                timestamp=now,
                model_name="qwen2.5-coder:14b",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h4 = [p for p in patterns if p.heuristic == "H4"]
        assert len(h4) == 1
        assert h4[0].severity == "high"
        assert "qwen2.5-coder:14b" in h4[0].description
        assert "dropped" in h4[0].description

    def test_no_detection_with_stable_utilization(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now - timedelta(hours=2),
                model_name="qwen2.5-coder:14b",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=78.0,
                timestamp=now,
                model_name="qwen2.5-coder:14b",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H4" for p in patterns)

    def test_no_detection_without_model_name(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=now,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H4" for p in patterns)


class TestH5SummaryQuality:
    def test_detects_low_quality_wrapups(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                summary_quality_score=0.4,
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=82.0,
                timestamp=now,
                summary_quality_score=0.3,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h5 = [p for p in patterns if p.heuristic == "H5"]
        # Should detect both low-score and regression cluster
        assert len(h5) == 2
        assert any(p.severity == "medium" and "below 0.5" in p.description for p in h5)
        assert any(p.severity == "high" and "regression" in p.description for p in h5)

    def test_no_detection_when_quality_normal(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                summary_quality_score=0.85,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H5" for p in patterns)

    def test_no_detection_without_quality_scores(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H5" for p in patterns)


class TestH6CorrectionLoops:
    def test_detects_high_correction_ratio(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                correction_count=5,
                user_turns=8,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h6 = [p for p in patterns if p.heuristic == "H6"]
        assert len(h6) == 1
        assert h6[0].severity == "high"
        assert "62%" in h6[0].description or "63%" in h6[0].description

    def test_no_detection_when_ratio_low(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                correction_count=1,
                user_turns=10,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H6" for p in patterns)

    def test_medium_severity_for_moderate_ratio(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                correction_count=2,
                user_turns=6,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h6 = [p for p in patterns if p.heuristic == "H6"]
        assert len(h6) == 1
        assert h6[0].severity == "medium"


class TestH7VaguePrompts:
    def test_detects_vague_prompt_cluster(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                vague_prompt_count=1,
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=82.0,
                timestamp=now,
                vague_prompt_count=2,
            ),
            MockTelemetryEvent(
                session_id="s3",
                event_name="WRAPPING_UP",
                utilization_percent=79.0,
                timestamp=now,
                vague_prompt_count=1,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h7 = [p for p in patterns if p.heuristic == "H7"]
        assert len(h7) == 1
        assert h7[0].severity == "medium"
        assert "3 sessions" in h7[0].description

    def test_low_severity_for_few_sessions(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                vague_prompt_count=1,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h7 = [p for p in patterns if p.heuristic == "H7"]
        assert len(h7) == 1
        assert h7[0].severity == "low"

    def test_no_detection_without_vague_prompts(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H7" for p in patterns)


class TestH8ToolOveruse:
    def test_detects_high_tool_ratio(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                turns=3,
                tool_calls=12,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h8 = [p for p in patterns if p.heuristic == "H8"]
        assert len(h8) == 1
        assert h8[0].severity == "low"
        assert "4.0x" in h8[0].description

    def test_detects_cluster_as_medium(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                turns=2,
                tool_calls=8,
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=82.0,
                timestamp=now,
                turns=2,
                tool_calls=7,
            ),
            MockTelemetryEvent(
                session_id="s3",
                event_name="WRAPPING_UP",
                utilization_percent=79.0,
                timestamp=now,
                turns=2,
                tool_calls=9,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        h8 = [p for p in patterns if p.heuristic == "H8"]
        assert len(h8) == 1
        assert h8[0].severity == "medium"
        assert "3 session(s)" in h8[0].description

    def test_no_detection_when_ratio_normal(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                turns=10,
                tool_calls=5,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H8" for p in patterns)

    def test_no_detection_without_turns(self):
        steward = SessionStewardCitizen(min_sessions=1)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                tool_calls=10,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert not any(p.heuristic == "H8" for p in patterns)


class TestPolicyDiffH4H8:
    def test_h4_generates_model_override_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H4",
                description="Model drift",
                severity="high",
                recommendation="Rotate model",
                data={"model_name": "qwen2.5-coder:14b", "drop_percent": 30.0},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert any(d.parameter == "model_override" for d in diffs)

    def test_h5_generates_wrapup_model_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H5",
                description="Low quality",
                severity="medium",
                recommendation="Switch model",
                data={},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert any(d.parameter == "wrapup_model" for d in diffs)

    def test_h6_generates_confirmation_step_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H6",
                description="Correction loops",
                severity="high",
                recommendation="Add confirmation",
                data={},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert any(d.parameter == "confirmation_step" for d in diffs)

    def test_h7_generates_intent_templates_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H7",
                description="Vague prompts",
                severity="medium",
                recommendation="Add templates",
                data={},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert any(d.parameter == "intent_templates" for d in diffs)

    def test_h8_generates_reasoning_budget_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H8",
                description="Tool thrashing",
                severity="medium",
                recommendation="Increase budget",
                data={},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert any(d.parameter == "reasoning_budget" for d in diffs)


class TestMinSessionsThreshold:
    def test_skips_analysis_below_threshold(self):
        steward = SessionStewardCitizen(min_sessions=10)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id=f"s{i}",
                event_name="WRAPPING_UP",
                timestamp=now,
            )
            for i in range(3)
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert len(patterns) == 0

    def test_runs_analysis_at_threshold(self):
        steward = SessionStewardCitizen(min_sessions=5)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id=f"s{i}",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=now,
                message="timer expired",
            )
            for i in range(5)
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert len(patterns) >= 1


class TestPolicyDiffGeneration:
    def test_h1_generates_timer_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H1",
                description="Low utilization",
                severity="medium",
                recommendation="Increase timer",
                data={"suggested_timer": "45m"},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert len(diffs) == 1
        assert diffs[0].parameter == "session_timer"
        assert diffs[0].proposed_value == "45m"

    def test_h2_generates_threshold_diff(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H2",
                description="Too tight",
                severity="high",
                recommendation="Lower threshold",
                data={"avg_utilization": 99.0},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert len(diffs) == 1
        assert diffs[0].parameter == "wrapup_threshold"
        assert diffs[0].proposed_value == "0.93"

    def test_h3_generates_timer_increase(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H3",
                description="Restart cluster",
                severity="medium",
                recommendation="Increase timer",
                data={},
            ),
        ]
        diffs = steward.generate_policy_diffs(patterns)
        assert len(diffs) == 1
        assert diffs[0].parameter == "session_timer"
        assert "+50%" in diffs[0].proposed_value


class TestProposalGeneration:
    def test_no_proposal_without_patterns(self):
        steward = SessionStewardCitizen(min_sessions=1)
        proposal = steward.generate_proposal([])
        assert proposal is None

    def test_generates_proposal_with_patterns(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H2",
                description="Threshold too tight",
                severity="high",
                recommendation="Lower threshold",
                data={},
            ),
        ]
        proposal = steward.generate_proposal(patterns)
        assert proposal is not None
        assert proposal.status == ProposalStatus.DRAFT
        assert "H2" in proposal.recommendation
        assert proposal.confidence_score > 0

    def test_proposal_includes_policy_diffs(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H1",
                description="Timer waste",
                severity="medium",
                recommendation="Increase timer",
                data={"suggested_timer": "45m"},
            ),
        ]
        diffs = [
            PolicyDiff(
                parameter="session_timer",
                current_value="30m",
                proposed_value="45m",
                rationale="Low utilization",
                expected_impact="Better usage",
            ),
        ]
        proposal = steward.generate_proposal(patterns, diffs)
        assert "session_timer" in proposal.recommendation
        assert "30m" in proposal.recommendation
        assert "45m" in proposal.recommendation

    def test_no_proposal_for_low_severity_only(self):
        steward = SessionStewardCitizen(min_sessions=1)
        patterns = [
            EfficiencyPattern(
                heuristic="H8",
                description="Minor issue",
                severity="low",
                recommendation="Monitor",
                data={},
            ),
        ]
        proposal = steward.generate_proposal(patterns)
        assert proposal is None


class TestAuditReport:
    def test_report_properties(self):
        report = SessionAuditReport(
            patterns=[
                EfficiencyPattern(
                    heuristic="H1",
                    description="test",
                    severity="medium",
                    recommendation="fix",
                ),
            ],
            sessions_analyzed=10,
        )
        assert report.has_actionable_findings is True
        assert report.sessions_analyzed == 10

    def test_report_no_actionable(self):
        report = SessionAuditReport(
            patterns=[
                EfficiencyPattern(
                    heuristic="H8",
                    description="test",
                    severity="low",
                    recommendation="monitor",
                ),
            ],
            sessions_analyzed=5,
        )
        assert report.has_actionable_findings is False


class TestAuditEntryPoint:
    def test_full_audit_workflow(self):
        with tempfile.TemporaryDirectory() as td:
            steward = SessionStewardCitizen(
                min_sessions=2,
                persistence_dir=td,
            )
            now = datetime.now(timezone.utc)
            events = [
                MockTelemetryEvent(
                    session_id="s1",
                    event_name="WRAPPING_UP",
                    utilization_percent=50.0,
                    elapsed_seconds=1200,
                    timestamp=now,
                    message="timer expired",
                ),
                MockTelemetryEvent(
                    session_id="s2",
                    event_name="WRAPPING_UP",
                    utilization_percent=50.0,
                    elapsed_seconds=1200,
                    timestamp=now,
                    message="timer expired",
                ),
            ]
            provider = MockTelemetryProvider(events)
            report = steward.audit(provider)

            assert report.sessions_analyzed == 2
            assert len(report.patterns) >= 1
            assert report.has_actionable_findings is True
            assert len(steward._audit_history) == 1

    def test_audit_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            steward = SessionStewardCitizen(
                min_sessions=2,
                persistence_dir=td,
            )
            now = datetime.now(timezone.utc)
            events = [
                MockTelemetryEvent(
                    session_id="s1",
                    event_name="WRAPPING_UP",
                    utilization_percent=50.0,
                    timestamp=now,
                    message="timer expired",
                ),
                MockTelemetryEvent(
                    session_id="s2",
                    event_name="WRAPPING_UP",
                    utilization_percent=50.0,
                    timestamp=now,
                    message="timer expired",
                ),
            ]
            provider = MockTelemetryProvider(events)
            steward.audit(provider)

            # Verify history was written
            history_file = Path(td) / "audit_history.jsonl"
            assert history_file.exists()
            lines = history_file.read_text().strip().splitlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["sessions_analyzed"] == 2
            assert data["has_actionable"] is True


class TestAnalysisWindow:
    def test_ignores_stale_events(self):
        steward = SessionStewardCitizen(
            min_sessions=2,
            analysis_window_hours=1.0,
        )
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=3)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=old,
                message="timer expired",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=old,
                message="timer expired",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        # Events are outside the 1-hour window
        assert len(patterns) == 0

    def test_includes_recent_events(self):
        steward = SessionStewardCitizen(
            min_sessions=2,
            analysis_window_hours=24.0,
        )
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=now,
                message="timer expired",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                timestamp=now,
                message="timer expired",
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert len(patterns) >= 1


class TestStoreProposal:
    def test_store_without_memory_layer(self):
        steward = SessionStewardCitizen(min_sessions=1)
        from animus.citizens.proposal import ImprovementProposal

        proposal = ImprovementProposal(
            id="test-1",
            title="Test",
            problem="Test problem",
        )
        result = steward.store_proposal(proposal)
        assert result is False


class TestHelpers:
    def test_suggest_timer(self):
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                elapsed_seconds=1200,
            ),
        ]
        result = SessionStewardCitizen._suggest_timer(events)
        assert result.endswith("m")
        minutes = int(result[:-1])
        assert minutes >= 30

    def test_score_to_confidence(self):
        assert SessionStewardCitizen._score_to_confidence(0.95).value == "very_high"
        assert SessionStewardCitizen._score_to_confidence(0.8).value == "high"
        assert SessionStewardCitizen._score_to_confidence(0.6).value == "medium"
        assert SessionStewardCitizen._score_to_confidence(0.3).value == "low"
        assert SessionStewardCitizen._score_to_confidence(0.1).value == "very_low"

    def test_short_id(self):
        sid = SessionStewardCitizen._short_id()
        assert len(sid) == 6
        assert sid.isalnum()


class TestEmptyTelemetry:
    def test_no_events_returns_empty(self):
        steward = SessionStewardCitizen(min_sessions=1)
        provider = MockTelemetryProvider([])
        patterns = steward.observe_telemetry(provider)
        assert patterns == []

    def test_only_non_wrapup_events(self):
        steward = SessionStewardCitizen(min_sessions=2)
        now = datetime.now(timezone.utc)
        events = [
            MockTelemetryEvent(
                session_id="s1",
                event_name="RUNNING",
                timestamp=now,
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="RUNNING",
                timestamp=now,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)
        assert patterns == []


class TestMixedHeuristics:
    def test_multiple_patterns_detected(self):
        steward = SessionStewardCitizen(min_sessions=2)
        now = datetime.now(timezone.utc)
        base = now - timedelta(minutes=30)
        events = [
            # H1: timer waste
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=50.0,
                elapsed_seconds=1200,
                timestamp=now,
                message="timer expired",
            ),
            # H2: threshold tight
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=99.0,
                timestamp=now,
                message="token utilization reached",
            ),
            # H3: restart cluster
            MockTelemetryEvent(
                session_id="s3",
                event_name="RESTARTING",
                timestamp=base,
            ),
            MockTelemetryEvent(
                session_id="s4",
                event_name="RESTARTING",
                timestamp=base + timedelta(minutes=10),
            ),
            MockTelemetryEvent(
                session_id="s5",
                event_name="RESTARTING",
                timestamp=base + timedelta(minutes=20),
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        heuristics = {p.heuristic for p in patterns}
        assert "H1" in heuristics
        assert "H2" in heuristics
        assert "H3" in heuristics

    def test_h4_h8_mixed_patterns(self):
        steward = SessionStewardCitizen(min_sessions=2)
        now = datetime.now(timezone.utc)
        base = now - timedelta(minutes=30)
        events = [
            # H4: model drift (high early, low recent)
            MockTelemetryEvent(
                session_id="s1",
                event_name="WRAPPING_UP",
                utilization_percent=90.0,
                timestamp=now - timedelta(hours=6),
                model_name="qwen2.5-coder:14b",
            ),
            MockTelemetryEvent(
                session_id="s2",
                event_name="WRAPPING_UP",
                utilization_percent=85.0,
                timestamp=now - timedelta(hours=5),
                model_name="qwen2.5-coder:14b",
            ),
            MockTelemetryEvent(
                session_id="s3",
                event_name="WRAPPING_UP",
                utilization_percent=55.0,
                timestamp=now - timedelta(hours=1),
                model_name="qwen2.5-coder:14b",
            ),
            MockTelemetryEvent(
                session_id="s4",
                event_name="WRAPPING_UP",
                utilization_percent=52.0,
                timestamp=now,
                model_name="qwen2.5-coder:14b",
            ),
            # H5: low summary quality
            MockTelemetryEvent(
                session_id="s5",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                summary_quality_score=0.4,
            ),
            # H6: correction loops
            MockTelemetryEvent(
                session_id="s6",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                correction_count=4,
                user_turns=6,
            ),
            # H7: vague prompts cluster
            MockTelemetryEvent(
                session_id="s7",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                vague_prompt_count=1,
            ),
            MockTelemetryEvent(
                session_id="s8",
                event_name="WRAPPING_UP",
                utilization_percent=82.0,
                timestamp=now,
                vague_prompt_count=1,
            ),
            MockTelemetryEvent(
                session_id="s9",
                event_name="WRAPPING_UP",
                utilization_percent=79.0,
                timestamp=now,
                vague_prompt_count=2,
            ),
            # H8: tool overuse cluster
            MockTelemetryEvent(
                session_id="s10",
                event_name="WRAPPING_UP",
                utilization_percent=80.0,
                timestamp=now,
                turns=2,
                tool_calls=8,
            ),
            MockTelemetryEvent(
                session_id="s11",
                event_name="WRAPPING_UP",
                utilization_percent=82.0,
                timestamp=now,
                turns=2,
                tool_calls=7,
            ),
            MockTelemetryEvent(
                session_id="s12",
                event_name="WRAPPING_UP",
                utilization_percent=79.0,
                timestamp=now,
                turns=2,
                tool_calls=9,
            ),
        ]
        provider = MockTelemetryProvider(events)
        patterns = steward.observe_telemetry(provider)

        heuristics = {p.heuristic for p in patterns}
        assert "H4" in heuristics
        assert "H5" in heuristics
        assert "H6" in heuristics
        assert "H7" in heuristics
        assert "H8" in heuristics