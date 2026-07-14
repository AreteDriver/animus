"""Tests for Meta-Thinker strategic oversight layer.

Validates P-20260706-002: Meta-Thinker anomaly detection and signal generation.
"""

import pytest

from animus.meta.anomalies import (
    AnomalyDetector,
    CircularToolUse,
    RepeatedFailures,
    GoalDrift,
    Stagnation,
)
from animus.meta.events import (
    Event,
    IterationStarted,
    ToolExecution,
    ResponseReceived,
    LoopCompleted,
    MaxIterationsReached,
)
from animus.meta.signals import (
    SignalType,
    ReplanSignal,
    InjectBriefSignal,
    EscalateSignal,
    HaltSignal,
    ReplanStrategy,
)
from animus.meta.thinker import MetaThinker, MetaThinkerConfig


class TestAnomalyDetectors:
    """Unit tests for individual anomaly detectors."""

    def test_circular_tool_use_detected(self):
        detector = CircularToolUse(threshold=0.5, min_repetitions=3)
        for i in range(4):
            detector.observe(
                ToolExecution(
                    iteration=i + 1,
                    tool_name="read_file",
                    params={"path": "/tmp/test.txt"},
                    success=True,
                )
            )
        report = detector.check()
        assert report is not None
        assert report.detector == "CircularToolUse"
        assert report.confidence >= 0.5

    def test_circular_tool_use_not_detected_different_params(self):
        detector = CircularToolUse(threshold=0.6, min_repetitions=3)
        for i in range(3):
            detector.observe(
                ToolExecution(
                    iteration=i + 1,
                    tool_name="read_file",
                    params={"path": f"/tmp/file{i}.txt"},
                    success=True,
                )
            )
        report = detector.check()
        assert report is None

    def test_repeated_failures_detected(self):
        detector = RepeatedFailures(threshold=0.5, min_consecutive=2)
        detector.observe(
            ToolExecution(
                iteration=1,
                tool_name="run_command",
                params={"command": "ls"},
                success=False,
                error="Permission denied",
            )
        )
        detector.observe(
            ToolExecution(
                iteration=2,
                tool_name="run_command",
                params={"command": "ls"},
                success=False,
                error="Permission denied",
            )
        )
        report = detector.check()
        assert report is not None
        assert report.detector == "RepeatedFailures"

    def test_repeated_failures_reset_on_success(self):
        detector = RepeatedFailures(threshold=0.5, min_consecutive=2)
        detector.observe(
            ToolExecution(
                iteration=1, tool_name="cmd", params={}, success=False
            )
        )
        detector.observe(
            ToolExecution(
                iteration=2, tool_name="cmd", params={}, success=True
            )
        )
        detector.observe(
            ToolExecution(
                iteration=3, tool_name="cmd", params={}, success=False
            )
        )
        report = detector.check()
        assert report is None

    def test_goal_drift_detected(self):
        detector = GoalDrift(threshold=0.5, window_size=5, min_drift_iterations=2)
        detector.set_original_prompt("read the file and summarize it")
        # First response is aligned (mentions file, summarize)
        detector.observe(
            ResponseReceived(
                iteration=1, text="Here is the file content. I will summarize it now."
            )
        )
        # Subsequent responses drift completely (weather, sports, unrelated)
        for i in range(2, 5):
            detector.observe(
                ResponseReceived(
                    iteration=i, text="The weather is sunny today. Maybe it will rain later."
                )
            )
        report = detector.check()
        assert report is not None
        assert report.detector == "GoalDrift"

    def test_goal_drift_not_detected_aligned(self):
        detector = GoalDrift(threshold=0.6, window_size=5, min_drift_iterations=2)
        detector.set_original_prompt("read the file and summarize it")
        for i in range(3):
            detector.observe(
                ResponseReceived(
                    iteration=i + 1,
                    text="Reading the file... found content. Summarizing: it says hello",
                )
            )
        report = detector.check()
        assert report is None

    def test_stagnation_detected(self):
        detector = Stagnation(threshold=0.5, window_size=6, min_repeats=3)
        for i in range(6):
            detector.observe(
                ToolExecution(iteration=i + 1, tool_name="read_file", params={}, success=True)
            )
        report = detector.check()
        assert report is not None
        assert report.detector == "Stagnation"


class TestMetaThinkerSignals:
    """Integration tests for Meta-Thinker signal generation."""

    def test_circular_tool_produces_replan(self):
        thinker = MetaThinker(MetaThinkerConfig(enabled=True, circular_tool_threshold=0.5))
        thinker.set_original_prompt("read and summarize")
        for i in range(4):
            thinker.observe(
                IterationStarted(iteration=i + 1, max_iterations=5)
            )
            thinker.observe(
                ResponseReceived(iteration=i + 1, text="calling read_file", tool_calls=[])
            )
            thinker.observe(
                ToolExecution(
                    iteration=i + 1,
                    tool_name="read_file",
                    params={"path": "/tmp/test.txt"},
                    success=True,
                )
            )

        signals = thinker.check()
        assert any(s.signal_type == SignalType.REPLAN for s in signals)

    def test_repeated_failures_produces_escalate(self):
        config = MetaThinkerConfig(
            enabled=True, escalate_on_repeated_failures=True
        )
        thinker = MetaThinker(config)
        thinker.set_original_prompt("run a command")
        for i in range(2):
            thinker.observe(
                ToolExecution(
                    iteration=i + 1,
                    tool_name="run_command",
                    params={"command": "rm /"},
                    success=False,
                    error="Permission denied",
                )
            )

        signals = thinker.check()
        assert any(s.signal_type == SignalType.ESCALATE for s in signals)

    def test_max_iterations_produces_halt(self):
        config = MetaThinkerConfig(enabled=True, halt_on_max_iterations=True)
        thinker = MetaThinker(config)
        thinker.set_original_prompt("do something")
        thinker.observe(
            MaxIterationsReached(
                iteration=5, final_response="partial result", total_iterations=5
            )
        )

        signals = thinker.check()
        assert any(s.signal_type == SignalType.HALT for s in signals)

    def test_disabled_thinker_returns_empty(self):
        thinker = MetaThinker(MetaThinkerConfig(enabled=False))
        thinker.observe(
            ToolExecution(iteration=1, tool_name="cmd", params={}, success=False)
        )
        assert thinker.check() == []

    def test_get_brief_returns_summary(self):
        thinker = MetaThinker(MetaThinkerConfig(enabled=True))
        thinker.observe(
            ToolExecution(iteration=1, tool_name="read_file", params={}, success=True)
        )
        thinker.observe(
            ToolExecution(
                iteration=2, tool_name="web_search", params={}, success=False
            )
        )
        brief = thinker.get_brief(iteration=2)
        assert brief is not None
        assert "Execution Summary" in brief
        assert "Successes: 1" in brief
        assert "Failures: 1" in brief

    def test_reset_clears_state(self):
        thinker = MetaThinker(MetaThinkerConfig(enabled=True))
        thinker.set_original_prompt("test")
        thinker.observe(
            ToolExecution(iteration=1, tool_name="cmd", params={}, success=True)
        )
        thinker.reset()
        assert thinker._events == []
        assert thinker._original_prompt == ""
        assert thinker.check() == []

    def test_event_history_limit(self):
        thinker = MetaThinker(MetaThinkerConfig(enabled=True))
        for i in range(MetaThinker._MAX_EVENTS + 50):
            thinker.observe(
                ToolExecution(iteration=i + 1, tool_name="cmd", params={}, success=True)
            )
        assert len(thinker._events) == MetaThinker._MAX_EVENTS
        # Verify oldest events were pruned (FIFO)
        assert thinker._events[0].iteration == 51


class TestSignalTypes:
    """Tests for signal dataclasses."""

    def test_replan_signal_fields(self):
        signal = ReplanSignal(
            confidence=0.8,
            reason="circular tool use",
            strategy=ReplanStrategy.SIMPLIFY_TASK,
            suggested_approach="break into steps",
        )
        assert signal.signal_type == SignalType.REPLAN
        assert signal.strategy == ReplanStrategy.SIMPLIFY_TASK
        assert signal.suggested_approach == "break into steps"

    def test_inject_brief_signal_fields(self):
        signal = InjectBriefSignal(
            confidence=0.7,
            reason="goal drift",
            brief_text="refocus on task",
            priority="high",
        )
        assert signal.signal_type == SignalType.INJECT_BRIEF
        assert signal.priority == "high"

    def test_escalate_signal_fields(self):
        signal = EscalateSignal(
            confidence=0.9, reason="repeated failures", target_mode="deep"
        )
        assert signal.signal_type == SignalType.ESCALATE
        assert signal.target_mode == "deep"
        assert signal.preserve_context is True

    def test_halt_signal_fields(self):
        signal = HaltSignal(
            confidence=1.0,
            reason="max iterations",
            partial_result="partial",
            explanation="stopped early",
        )
        assert signal.signal_type == SignalType.HALT
        assert signal.partial_result == "partial"
