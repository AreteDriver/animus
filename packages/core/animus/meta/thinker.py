"""Meta-Thinker: strategic oversight and anomaly detection for Animus Head.

Design: event-driven observer. Head emits events; Meta-Thinker consumes them,
runs detectors on a sliding window, and emits signals back to the Head.

Integration:
    meta = MetaThinker(config)
    meta.set_original_prompt(user_prompt)

    for iteration in loop:
        meta.observe(IterationStarted(...))
        # ... model generates response ...
        meta.observe(ResponseReceived(...))
        # ... execute tools ...
        for tool_result in results:
            meta.observe(ToolExecution(...))

        # Between iterations, check for signals
        signals = meta.check()
        for signal in signals:
            if signal.signal_type == SignalType.REPLAN:
                # Inject replan guidance into messages
                ...
            elif signal.signal_type == SignalType.HALT:
                # Break loop early
                ...
"""

from dataclasses import dataclass, field
from typing import Any

from animus.logging import get_logger
from animus.meta.anomalies import (
    AnomalyDetector,
    AnomalyReport,
    CircularToolUse,
    RepeatedFailures,
    GoalDrift,
    Stagnation,
)
from animus.meta.events import Event
from animus.meta.signals import (
    EscalateSignal,
    HaltSignal,
    InjectBriefSignal,
    ReplanSignal,
    ReplanStrategy,
    Signal,
    SignalType,
)

logger = get_logger("meta-thinker")


@dataclass
class MetaThinkerConfig:
    """Configuration for Meta-Thinker behavior."""

    enabled: bool = True
    max_signals_per_check: int = 3
    # Detector thresholds
    circular_tool_threshold: float = 0.7
    repeated_failure_threshold: float = 0.6
    goal_drift_threshold: float = 0.7
    stagnation_threshold: float = 0.7
    # Signal behavior
    replan_strategy: ReplanStrategy = ReplanStrategy.ADJUST_APPROACH
    halt_on_max_iterations: bool = True
    escalate_on_repeated_failures: bool = True


class MetaThinker:
    """Strategic oversight layer for agentic loops.

    Maintains a sliding window of execution events, runs anomaly detectors,
    and produces signals to guide or interrupt the Head loop.
    """

    _MAX_EVENTS = 500

    def __init__(self, config: MetaThinkerConfig | None = None):
        self.config = config or MetaThinkerConfig()
        self._events: list[Event] = []
        self._original_prompt: str = ""
        self._signals: list[Signal] = []
        self._detectors: list[AnomalyDetector] = []
        self._init_detectors()

    def _init_detectors(self) -> None:
        """Initialize anomaly detectors with config thresholds."""
        self._detectors = [
            CircularToolUse(threshold=self.config.circular_tool_threshold),
            RepeatedFailures(threshold=self.config.repeated_failure_threshold),
            GoalDrift(threshold=self.config.goal_drift_threshold),
            Stagnation(threshold=self.config.stagnation_threshold),
        ]

    def set_original_prompt(self, prompt: str) -> None:
        """Set the original user prompt for drift detection."""
        self._original_prompt = prompt
        for detector in self._detectors:
            if hasattr(detector, "set_original_prompt"):
                detector.set_original_prompt(prompt)

    def observe(self, event: Event) -> None:
        """Record an execution event."""
        if not self.config.enabled:
            return
        self._events.append(event)
        if len(self._events) > self._MAX_EVENTS:
            self._events = self._events[-self._MAX_EVENTS:]
        # Forward to detectors
        for detector in self._detectors:
            detector.observe(event)

    def reset(self) -> None:
        """Clear all state for a new session."""
        self._events = []
        self._signals = []
        self._original_prompt = ""
        for detector in self._detectors:
            detector.reset()

    def check(self) -> list[Signal]:
        """Run anomaly detectors and return any signals.

        Should be called between iterations of the agentic loop.
        """
        if not self.config.enabled:
            return []

        signals: list[Signal] = []

        # Run all detectors
        reports: list[AnomalyReport] = []
        for detector in self._detectors:
            report = detector.check()
            if report:
                reports.append(report)
                logger.warning(
                    f"Anomaly detected: {report.detector} "
                    f"(confidence={report.confidence:.2f}) — {report.description}"
                )

        # Sort by confidence descending
        reports.sort(key=lambda r: r.confidence, reverse=True)

        # Convert reports to signals
        for report in reports[: self.config.max_signals_per_check]:
            signal = self._report_to_signal(report)
            if signal:
                signals.append(signal)

        # Also check for max_iterations reached (explicit event)
        if self._events and hasattr(self._events[-1], "final_response"):
            # This is a MaxIterationsReached event
            if self.config.halt_on_max_iterations:
                signals.append(
                    HaltSignal(
                        confidence=1.0,
                        reason="Max iterations reached without completion",
                        partial_result=self._events[-1].final_response,
                        explanation="The agent loop exhausted its iteration budget. Consider simplifying the task or providing more specific instructions.",
                    )
                )

        return signals

    def _report_to_signal(self, report: AnomalyReport) -> Signal | None:
        """Convert an anomaly report to an appropriate signal."""
        if report.detector == "CircularToolUse":
            return ReplanSignal(
                confidence=report.confidence,
                reason=report.description,
                strategy=ReplanStrategy.ADJUST_APPROACH,
                suggested_approach=(
                    "You appear to be calling the same tool repeatedly. "
                    "Consider: (1) checking if the tool result already contains what you need, "
                    "(2) using a different tool, or (3) asking the user for clarification."
                ),
            )

        elif report.detector == "RepeatedFailures":
            if self.config.escalate_on_repeated_failures:
                return EscalateSignal(
                    confidence=report.confidence,
                    reason=report.description,
                    target_mode="deep",
                )
            return ReplanSignal(
                confidence=report.confidence,
                reason=report.description,
                strategy=ReplanStrategy.ASK_CLARIFICATION,
                suggested_approach="This tool is failing repeatedly. Please ask the user for guidance or try a different approach.",
            )

        elif report.detector == "GoalDrift":
            return InjectBriefSignal(
                confidence=report.confidence,
                reason=report.description,
                brief_text=(
                    f"[STRATEGIC BRIEF] Your responses appear to have drifted from the original task. "
                    f"Original goal: {self._original_prompt[:200]}... "
                    f"Please refocus on the original request."
                ),
                priority="high",
            )

        elif report.detector == "Stagnation":
            return ReplanSignal(
                confidence=report.confidence,
                reason=report.description,
                strategy=ReplanStrategy.SIMPLIFY_TASK,
                suggested_approach=(
                    "You appear to be stuck in a repeating pattern. "
                    "Break the task into smaller steps and tackle one at a time."
                ),
            )

        return None

    def get_brief(self, iteration: int) -> str | None:
        """Generate a strategic brief summarizing recent execution.

        Can be injected into context to help the model stay oriented.
        """
        if not self._events:
            return None

        # Count recent tool uses
        recent = [e for e in self._events if e.iteration >= iteration - 3]
        tool_execs = [e for e in recent if hasattr(e, "tool_name")]
        successes = sum(1 for e in tool_execs if getattr(e, "success", False))
        failures = len(tool_execs) - successes

        lines = [
            "[Execution Summary]",
            f"- Iterations so far: {iteration}",
            f"- Tools used recently: {len(tool_execs)}",
            f"- Successes: {successes}, Failures: {failures}",
        ]

        # Mention repeated tools
        from collections import Counter
        tool_names = [e.tool_name for e in tool_execs]
        repeats = Counter(tool_names).most_common(1)
        if repeats and repeats[0][1] > 1:
            lines.append(f"- Most used tool: {repeats[0][0]} ({repeats[0][1]}x)")

        lines.append("Stay focused on the original task.")
        return "\n".join(lines)
