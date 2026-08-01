"""Citizen 005 — The Session Steward.

Retrospective auditor of session lifecycle telemetry. Reads SessionController
event logs, identifies policy inefficiencies, and produces evidence-backed
improvement proposals.

Governance rules:
- Never auto-applies policy changes.
- Never modifies running sessions.
- Minimum 5 sessions before generating a proposal.
- All proposals go through standard queue → human approval → Forge commission.

Pipeline: Observe telemetry → Analyze patterns → Produce Proposal → Human Approval
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalConfidence,
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

logger = get_logger("citizens.session_steward")


# ── Protocols for loose coupling ───────────────────────────────────


class TelemetryEvent(Protocol):
    """Protocol for telemetry events from any session controller."""

    session_id: str
    event_name: str
    utilization_percent: float
    elapsed_seconds: float
    turns: int
    timestamp: datetime
    message: str


class TelemetryProvider(Protocol):
    """Protocol for objects that provide session telemetry."""

    def get_telemetry(self, session_id: str | None = None) -> list[TelemetryEvent]: ...

    def get_summary_stats(self) -> dict[str, float | int | str]: ...


# ── Data structures ────────────────────────────────────────────────


@dataclass
class EfficiencyPattern:
    """A detected inefficiency pattern in session telemetry."""

    heuristic: str  # "H1" through "H8"
    description: str
    severity: str  # "critical", "high", "medium", "low"
    recommendation: str
    data: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass
class PolicyDiff:
    """A proposed configuration change with before/after values."""

    parameter: str
    current_value: str | int | float | None
    proposed_value: str | int | float | None
    rationale: str
    expected_impact: str


@dataclass
class SessionAuditReport:
    """Complete audit output from the Session Steward."""

    patterns: list[EfficiencyPattern] = field(default_factory=list)
    policy_diffs: list[PolicyDiff] = field(default_factory=list)
    sessions_analyzed: int = 0
    analysis_window_hours: float = 24.0
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def has_actionable_findings(self) -> bool:
        return any(p.severity in ("critical", "high", "medium") for p in self.patterns)


class SessionStewardCitizen:
    """Citizen 005 — The Session Steward.

    Retrospective auditor of session lifecycle telemetry.
    Observes sessions, detects inefficiency patterns, and produces
    evidence-backed proposals for policy optimization.

    Integration points:
    - Daemon: scheduled via TaskScheduler as background audit task
    - Meta-Thinker: reads meta events from WarmSession for anomaly context
    - Eval: uses rubric scores from session outputs for quality heuristics
    """

    def __init__(
        self,
        min_sessions: int = 5,
        analysis_window_hours: float = 24.0,
        persistence_dir: str | Path | None = None,
        memory_layer=None,
    ) -> None:
        self.min_sessions = min_sessions
        self.analysis_window_hours = analysis_window_hours
        self.memory_layer = memory_layer
        self.persistence_dir = (
            Path(persistence_dir).expanduser()
            if persistence_dir
            else Path("~/.animus/session_steward").expanduser()
        )
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self._patterns: list[EfficiencyPattern] = []
        self._policy_diffs: list[PolicyDiff] = []
        self._audit_history: list[SessionAuditReport] = []
        self._load_history()

    # ── Persistence ───────────────────────────────────────────────────

    def _history_path(self) -> Path:
        return self.persistence_dir / "audit_history.jsonl"

    def _load_history(self) -> None:
        path = self._history_path()
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            try:
                data = json.loads(line)
                report = SessionAuditReport(
                    sessions_analyzed=data.get("sessions_analyzed", 0),
                    analysis_window_hours=data.get("analysis_window_hours", 24.0),
                    generated_at=datetime.fromisoformat(data["generated_at"]),
                )
                self._audit_history.append(report)
            except Exception:
                continue

    def _persist_report(self, report: SessionAuditReport) -> None:
        path = self._history_path()
        entry = {
            "sessions_analyzed": report.sessions_analyzed,
            "analysis_window_hours": report.analysis_window_hours,
            "generated_at": report.generated_at.isoformat(),
            "pattern_count": len(report.patterns),
            "has_actionable": report.has_actionable_findings,
        }
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Observation ───────────────────────────────────────────────────

    def observe_telemetry(
        self,
        controller: TelemetryProvider,
    ) -> list[EfficiencyPattern]:
        """Read session telemetry and detect inefficiency patterns.

        Args:
            controller: Any object implementing TelemetryProvider.

        Returns:
            List of detected patterns.
        """
        patterns: list[EfficiencyPattern] = []

        telemetry = controller.get_telemetry()
        if not telemetry:
            logger.info("No telemetry available.")
            return patterns

        # Filter to analysis window (use UTC-aware cutoff to match kernel timestamps)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.analysis_window_hours)
        recent = [
            t for t in telemetry if getattr(t, "timestamp", datetime.now(timezone.utc)) > cutoff
        ]

        wrapup_events = [t for t in recent if getattr(t, "event_name", "") == "WRAPPING_UP"]
        if len(wrapup_events) < self.min_sessions:
            logger.info(
                "Only %d wrapup events in window (min %d). Skipping analysis.",
                len(wrapup_events),
                self.min_sessions,
            )
            return patterns

        # Build per-session stats
        session_ids = {getattr(t, "session_id", "unknown") for t in recent}
        per_session: dict[str, list[TelemetryEvent]] = {}
        for sid in session_ids:
            per_session[sid] = controller.get_telemetry(sid)

        stats = controller.get_summary_stats()

        # Run heuristics
        patterns.extend(self._h1_timer_waste(per_session, wrapup_events))
        patterns.extend(self._h2_threshold_tight(per_session, wrapup_events))
        patterns.extend(self._h3_restart_fatigue(per_session))
        patterns.extend(self._h4_model_drift(per_session))
        patterns.extend(self._h5_summary_quality(per_session))
        patterns.extend(self._h6_correction_loops(per_session))
        patterns.extend(self._h7_vague_prompts(per_session))
        patterns.extend(self._h8_tool_overuse(per_session))

        self._patterns = patterns
        logger.info(
            "Detected %d patterns from %d sessions (window: %.1fh)",
            len(patterns),
            len(session_ids),
            self.analysis_window_hours,
        )
        return patterns

    # ── Heuristics ────────────────────────────────────────────────────

    def _h1_timer_waste(
        self,
        per_session: dict[str, list[TelemetryEvent]],
        wrapup_events: list[TelemetryEvent],
    ) -> list[EfficiencyPattern]:
        """Detect timer-triggered wrapups that under-utilize context window."""
        patterns: list[EfficiencyPattern] = []

        timer_wrapups = [
            e for e in wrapup_events if "timer expired" in (getattr(e, "message", "") or "")
        ]
        if not timer_wrapups:
            return patterns

        avg_util = sum(getattr(e, "utilization_percent", 0.0) for e in timer_wrapups) / len(
            timer_wrapups
        )
        if avg_util < 70.0:
            suggested_timer = self._suggest_timer(timer_wrapups)
            patterns.append(
                EfficiencyPattern(
                    heuristic="H1",
                    description=(
                        f"Timer-triggered wrapups average {avg_util:.1f}% utilization "
                        f"(below 70% threshold). {len(timer_wrapups)} sessions affected."
                    ),
                    severity="medium",
                    recommendation=(
                        f"Consider increasing session timer to {suggested_timer} "
                        f"or lowering wrapup threshold to capture more value per session."
                    ),
                    data={
                        "avg_utilization": avg_util,
                        "session_count": len(timer_wrapups),
                        "suggested_timer": suggested_timer,
                    },
                )
            )
        return patterns

    def _h2_threshold_tight(
        self,
        per_session: dict[str, list[TelemetryEvent]],
        wrapup_events: list[TelemetryEvent],
    ) -> list[EfficiencyPattern]:
        """Detect token-threshold wrapups that push too close to the limit."""
        patterns: list[EfficiencyPattern] = []

        threshold_wrapups = [
            e for e in wrapup_events if "token utilization" in (getattr(e, "message", "") or "")
        ]
        if not threshold_wrapups:
            return patterns

        avg_util = sum(getattr(e, "utilization_percent", 0.0) for e in threshold_wrapups) / len(
            threshold_wrapups
        )
        if avg_util > 98.0:
            patterns.append(
                EfficiencyPattern(
                    heuristic="H2",
                    description=(
                        f"Token-threshold wrapups average {avg_util:.1f}% utilization "
                        f"(above 98% threshold). Model-generated summaries at this level "
                        f"are lower quality. {len(threshold_wrapups)} sessions affected."
                    ),
                    severity="high",
                    recommendation=(
                        "Lower wrapup threshold to 0.92–0.94 to give the model "
                        "more headroom for quality summary generation."
                    ),
                    data={
                        "avg_utilization": avg_util,
                        "session_count": len(threshold_wrapups),
                    },
                )
            )
        return patterns

    def _h3_restart_fatigue(
        self,
        per_session: dict[str, list[TelemetryEvent]],
    ) -> list[EfficiencyPattern]:
        """Detect clusters of frequent restarts within short time windows."""
        patterns: list[EfficiencyPattern] = []

        restart_events = []
        for sid, events in per_session.items():
            for e in events:
                if getattr(e, "event_name", "") == "RESTARTING":
                    restart_events.append(e)

        if len(restart_events) < 3:
            return patterns

        restart_events.sort(key=lambda x: getattr(x, "timestamp", datetime.min))
        window = timedelta(hours=2)
        clusters: list[list[TelemetryEvent]] = []
        current: list[TelemetryEvent] = []

        for e in restart_events:
            if not current:
                current = [e]
            elif (
                getattr(e, "timestamp", datetime.min)
                - getattr(current[0], "timestamp", datetime.min)
                <= window
            ):
                current.append(e)
            else:
                if len(current) >= 3:
                    clusters.append(current)
                current = [e]
        if len(current) >= 3:
            clusters.append(current)

        for cluster in clusters:
            elapsed = (
                getattr(cluster[-1], "timestamp", datetime.min)
                - getattr(cluster[0], "timestamp", datetime.min)
            ).total_seconds()
            patterns.append(
                EfficiencyPattern(
                    heuristic="H3",
                    description=(
                        f"{len(cluster)} restarts within {elapsed / 60:.0f} minutes "
                        f"({getattr(cluster[0], 'session_id', 'unknown')[:12]}…)."
                    ),
                    severity="medium",
                    recommendation=(
                        "Consider a longer timer or higher threshold for sustained "
                        "deep-work sessions to reduce context fragmentation."
                    ),
                    data={
                        "restart_count": len(cluster),
                        "elapsed_seconds": elapsed,
                        "window_hours": 2,
                    },
                )
            )
        return patterns

    def _h4_model_drift(
        self,
        per_session: dict[str, list[TelemetryEvent]],
    ) -> list[EfficiencyPattern]:
        """Detect model-specific utilization decline (drift).

        Compares the most recent half of wrapup events to the earlier half
        for each model. A significant drop in average utilization indicates
        the model is degrading for the current workload.
        """
        patterns: list[EfficiencyPattern] = []

        # Gather all wrapup events with model names
        wrapups: list[TelemetryEvent] = []
        for events in per_session.values():
            for e in events:
                if getattr(e, "event_name", "") == "WRAPPING_UP":
                    model = getattr(e, "model_name", "")
                    if model:
                        wrapups.append(e)

        if len(wrapups) < 4:
            return patterns

        # Group by model
        by_model: dict[str, list[TelemetryEvent]] = {}
        for e in wrapups:
            model = getattr(e, "model_name", "unknown")
            by_model.setdefault(model, []).append(e)

        for model, events in by_model.items():
            if len(events) < 4:
                continue

            events.sort(key=lambda x: getattr(x, "timestamp", datetime.min))
            mid = len(events) // 2
            first_half = events[:mid]
            second_half = events[mid:]

            avg_first = sum(getattr(e, "utilization_percent", 0.0) for e in first_half) / len(
                first_half
            )
            avg_second = sum(getattr(e, "utilization_percent", 0.0) for e in second_half) / len(
                second_half
            )

            if avg_first == 0:
                continue

            drop_percent = (avg_first - avg_second) / avg_first * 100

            if drop_percent > 15.0:
                severity = "high" if drop_percent > 25.0 else "medium"
                patterns.append(
                    EfficiencyPattern(
                        heuristic="H4",
                        description=(
                            f"Model '{model}' utilization dropped {drop_percent:.1f}% "
                            f"(from {avg_first:.1f}% to {avg_second:.1f}%) across "
                            f"{len(events)} sessions."
                        ),
                        severity=severity,
                        recommendation=(
                            f"Consider rotating away from '{model}' or lowering its "
                            f"wrapup threshold to compensate for quality degradation."
                        ),
                        data={
                            "model_name": model,
                            "avg_first": avg_first,
                            "avg_second": avg_second,
                            "drop_percent": drop_percent,
                            "session_count": len(events),
                        },
                    )
                )

        return patterns

    def _h5_summary_quality(
        self,
        per_session: dict[str, list[TelemetryEvent]],
    ) -> list[EfficiencyPattern]:
        """Detect wrapup summary quality regressions.

        Wrapup events may carry a ``summary_quality_score`` (0.0–1.0)
        from offline rubric evaluation. Flags single low scores and
        regression clusters.
        """
        patterns: list[EfficiencyPattern] = []

        scored_wrapups: list[TelemetryEvent] = []
        for events in per_session.values():
            for e in events:
                if getattr(e, "event_name", "") == "WRAPPING_UP":
                    score = getattr(e, "summary_quality_score", 0.0)
                    if score > 0.0:
                        scored_wrapups.append(e)

        if not scored_wrapups:
            return patterns

        # Sort chronologically
        scored_wrapups.sort(key=lambda x: getattr(x, "timestamp", datetime.min))

        # Single low scores
        low_scores = [e for e in scored_wrapups if getattr(e, "summary_quality_score", 1.0) < 0.5]
        if low_scores:
            patterns.append(
                EfficiencyPattern(
                    heuristic="H5",
                    description=(
                        f"{len(low_scores)} wrapup summary(s) scored below 0.5 "
                        f"(out of {len(scored_wrapups)} evaluated)."
                    ),
                    severity="medium",
                    recommendation=(
                        "Review summary generation prompt. Consider switching to a "
                        "higher-capability model for wrapup turns."
                    ),
                    data={
                        "low_score_count": len(low_scores),
                        "total_evaluated": len(scored_wrapups),
                    },
                )
            )

        # Regression cluster: 2+ consecutive low scores
        regression_count = 0
        consecutive = 0
        for e in scored_wrapups:
            score = getattr(e, "summary_quality_score", 1.0)
            if score < 0.6:
                consecutive += 1
                if consecutive >= 2:
                    regression_count += 1
            else:
                consecutive = 0

        if regression_count > 0:
            patterns.append(
                EfficiencyPattern(
                    heuristic="H5",
                    description=(
                        f"{regression_count} regression cluster(s) of 2+ consecutive "
                        f"low-quality wrapup summaries detected."
                    ),
                    severity="high",
                    recommendation=(
                        "Raise wrapup threshold to give the model more headroom, "
                        "or switch to a dedicated summary model."
                    ),
                    data={
                        "regression_count": regression_count,
                        "threshold": 0.6,
                    },
                )
            )

        return patterns

    def _h6_correction_loops(
        self,
        per_session: dict[str, list[TelemetryEvent]],
    ) -> list[EfficiencyPattern]:
        """Detect sessions with high correction-turn ratios.

        Telemetry events carry ``correction_count`` and ``user_turns``.
        A ratio > 0.3 indicates the user is spending significant effort
        correcting the agent.
        """
        patterns: list[EfficiencyPattern] = []

        flagged_sessions: list[dict[str, Any]] = []
        for sid, events in per_session.items():
            # Aggregate correction counts and user turns across all events for session
            correction_count = 0
            user_turns = 0
            for e in events:
                correction_count += getattr(e, "correction_count", 0)
                user_turns += getattr(e, "user_turns", 0)

            if user_turns == 0:
                continue

            ratio = correction_count / user_turns
            if ratio > 0.3:
                flagged_sessions.append(
                    {
                        "session_id": sid,
                        "correction_count": correction_count,
                        "user_turns": user_turns,
                        "ratio": ratio,
                    }
                )

        if not flagged_sessions:
            return patterns

        # Severity based on worst ratio
        max_ratio = max(s["ratio"] for s in flagged_sessions)
        severity = "high" if max_ratio > 0.5 else "medium"
        total_corrections = sum(s["correction_count"] for s in flagged_sessions)

        patterns.append(
            EfficiencyPattern(
                heuristic="H6",
                description=(
                    f"{len(flagged_sessions)} session(s) show high correction ratios "
                    f"(max {max_ratio:.0%}). {total_corrections} total corrections."
                ),
                severity=severity,
                recommendation=(
                    "Add a confirmation step before execution. Improve intent parsing "
                    "to catch the user's actual goal on the first turn."
                ),
                data={
                    "affected_sessions": len(flagged_sessions),
                    "max_ratio": max_ratio,
                    "total_corrections": total_corrections,
                },
            )
        )

        return patterns

    def _h7_vague_prompts(
        self,
        per_session: dict[str, list[TelemetryEvent]],
    ) -> list[EfficiencyPattern]:
        """Detect sessions triggered by vague or underspecified prompts.

        Telemetry events carry ``vague_prompt_count`` per session.
        Clusters of vague-start sessions indicate missing shortcuts
        or poor discoverability.
        """
        patterns: list[EfficiencyPattern] = []

        vague_sessions: list[dict[str, Any]] = []
        for sid, events in per_session.items():
            vague_count = sum(getattr(e, "vague_prompt_count", 0) for e in events)
            if vague_count > 0:
                vague_sessions.append(
                    {
                        "session_id": sid,
                        "vague_count": vague_count,
                    }
                )

        if not vague_sessions:
            return patterns

        total_vague = sum(s["vague_count"] for s in vague_sessions)

        # Cluster detection: ≥3 sessions with vague prompts
        if len(vague_sessions) >= 3:
            patterns.append(
                EfficiencyPattern(
                    heuristic="H7",
                    description=(
                        f"{len(vague_sessions)} sessions started with vague prompts "
                        f"({total_vague} total)."
                    ),
                    severity="medium",
                    recommendation=(
                        "Add structured command shortcuts or template options for "
                        "the most common ambiguous intents."
                    ),
                    data={
                        "affected_sessions": len(vague_sessions),
                        "total_vague_prompts": total_vague,
                    },
                )
            )
        else:
            patterns.append(
                EfficiencyPattern(
                    heuristic="H7",
                    description=(
                        f"{len(vague_sessions)} session(s) with vague prompts "
                        f"({total_vague} total)."
                    ),
                    severity="low",
                    recommendation=(
                        "Monitor for pattern growth. Consider clarifying questions "
                        "when intent confidence is low."
                    ),
                    data={
                        "affected_sessions": len(vague_sessions),
                        "total_vague_prompts": total_vague,
                    },
                )
            )

        return patterns

    def _h8_tool_overuse(
        self,
        per_session: dict[str, list[TelemetryEvent]],
    ) -> list[EfficiencyPattern]:
        """Detect sessions with excessive tool call ratios.

        High tool-to-turn ratios suggest the agent is thrashing rather than
        reasoning. Often correlates with poor prompt engineering.
        """
        patterns: list[EfficiencyPattern] = []
        flagged_sessions: list[dict[str, Any]] = []

        for sid, events in per_session.items():
            tool_calls = sum(getattr(e, "tool_calls", 0) for e in events)
            turns = sum(getattr(e, "turns", 0) for e in events)
            if turns == 0:
                continue

            ratio = tool_calls / turns
            if ratio > 3.0:
                flagged_sessions.append(
                    {
                        "session_id": sid,
                        "tool_calls": tool_calls,
                        "turns": turns,
                        "ratio": ratio,
                    }
                )

        if not flagged_sessions:
            return patterns

        avg_ratio = sum(s["ratio"] for s in flagged_sessions) / len(flagged_sessions)
        max_ratio = max(s["ratio"] for s in flagged_sessions)
        severity = "medium" if len(flagged_sessions) >= 3 else "low"

        patterns.append(
            EfficiencyPattern(
                heuristic="H8",
                description=(
                    f"{len(flagged_sessions)} session(s) show elevated tool call ratios "
                    f"(avg {avg_ratio:.1f}x, max {max_ratio:.1f}x per turn). "
                    f"Agent may be thrashing rather than reasoning."
                ),
                severity=severity,
                recommendation=(
                    "Review tool descriptions for clarity. Consider enabling "
                    "Meta-Thinker tool-use cooldown or increasing reasoning budget."
                ),
                data={
                    "affected_sessions": len(flagged_sessions),
                    "avg_ratio": avg_ratio,
                    "max_ratio": max_ratio,
                },
            )
        )

        return patterns

    # ── Policy Diff Generation ────────────────────────────────────────

    def generate_policy_diffs(
        self,
        patterns: list[EfficiencyPattern],
    ) -> list[PolicyDiff]:
        """Convert detected patterns into concrete configuration changes.

        Returns a list of PolicyDiff objects with before/after values.
        These are recommendations, not applied changes.
        """
        diffs: list[PolicyDiff] = []

        for pattern in patterns:
            if pattern.heuristic == "H1" and pattern.data.get("suggested_timer"):
                diffs.append(
                    PolicyDiff(
                        parameter="session_timer",
                        current_value="current",
                        proposed_value=pattern.data["suggested_timer"],
                        rationale=pattern.description,
                        expected_impact="Higher utilization per session, fewer timer-triggered wrapups",
                    )
                )
            elif pattern.heuristic == "H2":
                diffs.append(
                    PolicyDiff(
                        parameter="wrapup_threshold",
                        current_value="0.96",
                        proposed_value="0.93",
                        rationale=pattern.description,
                        expected_impact="Better summary quality with 3% more headroom",
                    )
                )
            elif pattern.heuristic == "H3":
                diffs.append(
                    PolicyDiff(
                        parameter="session_timer",
                        current_value="current",
                        proposed_value="+50%",
                        rationale=pattern.description,
                        expected_impact="Fewer restarts, deeper context windows",
                    )
                )
            elif pattern.heuristic == "H4" and pattern.data.get("model_name"):
                diffs.append(
                    PolicyDiff(
                        parameter="model_override",
                        current_value=pattern.data["model_name"],
                        proposed_value="rotate_to_best_provider",
                        rationale=pattern.description,
                        expected_impact="Restore utilization levels by routing away from degraded model",
                    )
                )
            elif pattern.heuristic == "H5":
                diffs.append(
                    PolicyDiff(
                        parameter="wrapup_model",
                        current_value="default",
                        proposed_value="higher_capability",
                        rationale=pattern.description,
                        expected_impact="Improve summary quality at session wrapup",
                    )
                )
            elif pattern.heuristic == "H6":
                diffs.append(
                    PolicyDiff(
                        parameter="confirmation_step",
                        current_value="disabled",
                        proposed_value="enabled",
                        rationale=pattern.description,
                        expected_impact="Reduce correction loops by confirming intent before execution",
                    )
                )
            elif pattern.heuristic == "H7":
                diffs.append(
                    PolicyDiff(
                        parameter="intent_templates",
                        current_value="existing",
                        proposed_value="expand_top_3",
                        rationale=pattern.description,
                        expected_impact="Reduce vague prompts by providing structured starting points",
                    )
                )
            elif pattern.heuristic == "H8":
                diffs.append(
                    PolicyDiff(
                        parameter="reasoning_budget",
                        current_value="default",
                        proposed_value="increase_20pct",
                        rationale=pattern.description,
                        expected_impact="Reduce tool thrashing by giving the model more reasoning tokens per turn",
                    )
                )

        self._policy_diffs = diffs
        return diffs

    # ── Audit Report ────────────────────────────────────────────────

    def audit(
        self,
        controller: TelemetryProvider,
    ) -> SessionAuditReport:
        """Run a full audit and return a report.

        This is the main entry point for daemon integration.
        """
        patterns = self.observe_telemetry(controller)
        diffs = self.generate_policy_diffs(patterns)

        report = SessionAuditReport(
            patterns=patterns,
            policy_diffs=diffs,
            sessions_analyzed=len(
                {getattr(t, "session_id", "") for t in controller.get_telemetry()}
            ),
            analysis_window_hours=self.analysis_window_hours,
        )

        self._persist_report(report)
        self._audit_history.append(report)
        return report

    # ── Proposal generation ─────────────────────────────────────────

    def generate_proposal(
        self,
        patterns: list[EfficiencyPattern] | None = None,
        diffs: list[PolicyDiff] | None = None,
    ) -> ImprovementProposal | None:
        """Generate an ImprovementProposal from detected patterns.

        Returns None if no actionable patterns found.
        """
        if patterns is None:
            patterns = self._patterns
        if diffs is None:
            diffs = self._policy_diffs

        if not patterns:
            return None

        # Aggregate findings
        critical = [p for p in patterns if p.severity == "critical"]
        high = [p for p in patterns if p.severity == "high"]
        medium = [p for p in patterns if p.severity == "medium"]

        if not (critical or high or medium):
            return None

        # Build problem statement
        problem_parts = []
        if critical:
            problem_parts.append(f"{len(critical)} critical inefficiency detected")
        if high:
            problem_parts.append(f"{len(high)} high-severity inefficiency detected")
        if medium:
            problem_parts.append(f"{len(medium)} medium-severity inefficiency detected")
        problem = "; ".join(problem_parts) + "."

        # Build recommendation
        recs = []
        for p in patterns[:3]:  # Top 3 patterns
            recs.append(f"[{p.heuristic}] {p.recommendation}")
        if diffs:
            recs.append("\nProposed configuration changes:")
            for d in diffs[:3]:
                recs.append(f"  - {d.parameter}: {d.current_value} → {d.proposed_value}")
        recommendation = "\n".join(recs)

        # Evidence
        evidence = [
            EvidenceItem(
                source="session_telemetry",
                description=f"Detected {len(patterns)} efficiency patterns across session lifecycle events.",
                data={"patterns": [{"h": p.heuristic, "sev": p.severity} for p in patterns]},
            )
        ]

        # Risks
        risks: list[RiskAssessment] = []
        if any(p.heuristic == "H2" for p in patterns):
            risks.append(
                RiskAssessment(
                    description="Lowering wrapup threshold may increase session frequency.",
                    severity="low",
                    mitigation="Monitor checkpoint size and total session count after change.",
                )
            )

        # Confidence based on sample size
        total_sessions = max(len(patterns), 5)
        confidence_score = min(0.95, 0.5 + (total_sessions * 0.05))
        confidence = self._score_to_confidence(confidence_score)

        return ImprovementProposal(
            id=f"ADL-{datetime.now().strftime('%Y%m%d')}-{self._short_id()}",
            title=f"Session Policy Optimization — {len(patterns)} pattern(s) detected",
            problem=problem,
            recommendation=recommendation,
            affected_components=["animus_kernel.head.session_controller"],
            confidence_score=confidence_score,
            confidence_label=confidence,
            estimated_effort_hours=0,  # Pure config change
            evidence=evidence,
            potential_risks=risks,
            status=ProposalStatus.DRAFT,
        )

    def store_proposal(self, proposal: ImprovementProposal) -> bool:
        """Store proposal in memory layer if available."""
        if self.memory_layer is None:
            return False
        try:
            self.memory_layer.store(
                content=proposal.recommendation,
                memory_type="procedural",
                tags=f"session-steward,proposal,{proposal.id}",
                metadata={
                    "proposal_id": proposal.id,
                    "confidence_score": proposal.confidence_score,
                    "affected_components": proposal.affected_components,
                },
            )
            return True
        except Exception:
            logger.warning("Failed to store proposal in memory", exc_info=True)
            return False

    # ── Daemon Integration ──────────────────────────────────────────

    def create_daemon_task(
        self,
        daemon_scheduler: object,
        controller: TelemetryProvider,
        interval_seconds: int = 3600,
    ) -> object:
        """Register a recurring audit task with the daemon scheduler.

        Args:
            daemon_scheduler: A TaskScheduler instance.
            controller: Telemetry provider to audit.
            interval_seconds: How often to run audits.

        Returns:
            The scheduled task object.
        """

        def audit_callback() -> None:
            logger.info("Running scheduled Session Steward audit")
            report = self.audit(controller)
            if report.has_actionable_findings:
                proposal = self.generate_proposal(report.patterns, report.policy_diffs)
                if proposal:
                    self.store_proposal(proposal)
                    logger.info(f"Session Steward generated proposal: {proposal.id}")
            else:
                logger.info("No actionable findings this cycle")

        # Store callback in metadata for execution
        task = daemon_scheduler.schedule_interval(
            description="Session Steward telemetry audit",
            seconds=interval_seconds,
            priority="normal",
            metadata={"citizen": "session_steward", "callback": audit_callback},
        )
        return task

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _suggest_timer(timer_wrapups: list[TelemetryEvent]) -> str:
        """Suggest a longer timer based on observed patterns."""
        avg_elapsed = sum(getattr(e, "elapsed_seconds", 0.0) for e in timer_wrapups) / max(
            len(timer_wrapups), 1
        )
        suggestion = max(avg_elapsed * 1.5, 1800)  # At least 30m
        minutes = int(suggestion / 60)
        return f"{minutes}m"

    @staticmethod
    def _score_to_confidence(score: float) -> ProposalConfidence:
        if score >= 0.9:
            return ProposalConfidence.VERY_HIGH
        if score >= 0.75:
            return ProposalConfidence.HIGH
        if score >= 0.5:
            return ProposalConfidence.MEDIUM
        if score >= 0.25:
            return ProposalConfidence.LOW
        return ProposalConfidence.VERY_LOW

    @staticmethod
    def _short_id() -> str:
        return uuid.uuid4().hex[:6]
