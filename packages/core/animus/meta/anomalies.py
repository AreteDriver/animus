"""Anomaly detectors for Meta-Thinker strategic oversight.

Each detector implements a specific pattern from the COMPASS taxonomy:
- CircularToolUse: repeated identical/similar tool calls
- RepeatedFailures: consecutive failures on same tool
- GoalDrift: response text diverges from original task
- Stagnation: no meaningful progress over multiple iterations
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from animus.meta.events import Event, ToolExecution, ResponseReceived, MaxIterationsReached


@dataclass
class AnomalyReport:
    """Report generated when an anomaly is detected."""

    detector: str
    confidence: float
    description: str
    evidence: list[str]
    affected_iterations: list[int]


class AnomalyDetector(ABC):
    """Base class for anomaly detectors."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._events: list[Event] = []

    def observe(self, event: Event) -> None:
        """Record an event for analysis."""
        self._events.append(event)

    def reset(self) -> None:
        """Clear event history."""
        self._events = []

    @abstractmethod
    def check(self) -> AnomalyReport | None:
        """Check for anomalies in observed events.

        Returns AnomalyReport if anomaly detected, None otherwise.
        """
        ...


class CircularToolUse(AnomalyDetector):
    """Detects when the same tool is called repeatedly with similar parameters.

    Pattern: tool X called >=3 times with >=80% param similarity within a window.
    """

    def __init__(self, threshold: float = 0.7, min_repetitions: int = 3, window_size: int = 10):
        super().__init__(threshold)
        self.min_repetitions = min_repetitions
        self.window_size = window_size

    def _param_similarity(self, p1: dict, p2: dict) -> float:
        """Compute Jaccard similarity of parameter value strings."""
        s1 = set(str(v).lower() for v in p1.values() if v)
        s2 = set(str(v).lower() for v in p2.values() if v)
        if not s1 or not s2:
            return 0.0
        overlap = len(s1 & s2)
        union = len(s1 | s2)
        return overlap / union if union > 0 else 0.0

    def check(self) -> AnomalyReport | None:
        tool_execs = [
            e for e in self._events[-self.window_size :]
            if isinstance(e, ToolExecution)
        ]
        if len(tool_execs) < self.min_repetitions:
            return None

        from collections import Counter
        counts = Counter(e.tool_name for e in tool_execs)
        for tool_name, count in counts.items():
            if count < self.min_repetitions:
                continue
            # Check if params are similar
            tool_events = [e for e in tool_execs if e.tool_name == tool_name]
            similarities = []
            for i in range(len(tool_events) - 1):
                sim = self._param_similarity(tool_events[i].params, tool_events[i + 1].params)
                similarities.append(sim)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            if avg_sim >= 0.8:
                confidence = min(1.0, 0.5 + (count - self.min_repetitions) * 0.15)
                if confidence >= self.threshold:
                    iterations = sorted(set(e.iteration for e in tool_events))
                    return AnomalyReport(
                        detector="CircularToolUse",
                        confidence=confidence,
                        description=f"Tool '{tool_name}' called {count} times with similar params (avg similarity: {avg_sim:.2f})",
                        evidence=[f"Call {i+1}: {e.params}" for i, e in enumerate(tool_events)],
                        affected_iterations=iterations,
                    )
        return None


class RepeatedFailures(AnomalyDetector):
    """Detects consecutive failures on the same tool.

    Pattern: >=2 consecutive failures on same tool.
    """

    def __init__(self, threshold: float = 0.6, min_consecutive: int = 2, window_size: int = 10):
        super().__init__(threshold)
        self.min_consecutive = min_consecutive
        self.window_size = window_size

    def check(self) -> AnomalyReport | None:
        tool_execs = [
            e for e in self._events[-self.window_size :]
            if isinstance(e, ToolExecution)
        ]
        if not tool_execs:
            return None

        # Find consecutive failures per tool
        from collections import defaultdict
        failures_by_tool: dict[str, list[ToolExecution]] = defaultdict(list)
        for e in tool_execs:
            if not e.success:
                failures_by_tool[e.tool_name].append(e)
            else:
                # Reset streak on success
                failures_by_tool[e.tool_name] = []

        for tool_name, failures in failures_by_tool.items():
            if len(failures) >= self.min_consecutive:
                confidence = min(1.0, 0.4 + len(failures) * 0.2)
                if confidence >= self.threshold:
                    iterations = sorted(set(e.iteration for e in failures))
                    errors = [e.error for e in failures if e.error]
                    return AnomalyReport(
                        detector="RepeatedFailures",
                        confidence=confidence,
                        description=f"Tool '{tool_name}' failed {len(failures)} consecutive times",
                        evidence=errors[:5],
                        affected_iterations=iterations,
                    )
        return None


class GoalDrift(AnomalyDetector):
    """Detects when response text deviates from the original task.

    Pattern: response content shows low keyword overlap with original prompt.
    """

    def __init__(self, threshold: float = 0.7, window_size: int = 5, min_drift_iterations: int = 2):
        super().__init__(threshold)
        self.window_size = window_size
        self.min_drift_iterations = min_drift_iterations
        self._original_prompt = ""

    def set_original_prompt(self, prompt: str) -> None:
        self._original_prompt = prompt.lower()

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return set(w for w in words if len(w) > 2)

    def _drift_score(self, response_text: str) -> float:
        """Return drift score where 1.0 = complete drift, 0.0 = aligned."""
        if not self._original_prompt:
            return 0.0
        prompt_tokens = self._tokenize(self._original_prompt)
        response_tokens = self._tokenize(response_text)
        if not prompt_tokens or not response_tokens:
            return 0.0
        overlap = len(prompt_tokens & response_tokens)
        return 1.0 - (overlap / len(prompt_tokens))

    def check(self) -> AnomalyReport | None:
        responses = [
            e for e in self._events[-self.window_size :]
            if isinstance(e, ResponseReceived)
        ]
        drift_scores = []
        for r in responses:
            score = self._drift_score(r.text)
            drift_scores.append((r.iteration, score))

        high_drift = [(it, s) for it, s in drift_scores if s > 0.7]
        if len(high_drift) >= self.min_drift_iterations:
            confidence = min(1.0, sum(s for _, s in high_drift) / len(high_drift))
            if confidence >= self.threshold:
                iterations = [it for it, _ in high_drift]
                return AnomalyReport(
                    detector="GoalDrift",
                    confidence=confidence,
                    description=f"Response drifted from original task in {len(high_drift)} iterations",
                    evidence=[f"Iteration {it}: drift score {s:.2f}" for it, s in high_drift],
                    affected_iterations=iterations,
                )
        return None


class Stagnation(AnomalyDetector):
    """Detects when the agent makes no meaningful progress.

    Pattern: same tools called in same order, or no new information gained.
    """

    def __init__(self, threshold: float = 0.7, window_size: int = 6, min_repeats: int = 3):
        super().__init__(threshold)
        self.window_size = window_size
        self.min_repeats = min_repeats

    def check(self) -> AnomalyReport | None:
        tool_execs = [
            e for e in self._events[-self.window_size :]
            if isinstance(e, ToolExecution)
        ]
        if len(tool_execs) < self.min_repeats:
            return None

        # Extract tool call sequences
        sequences = []
        current = []
        for e in tool_execs:
            current.append(e.tool_name)
            if len(current) >= 3:
                sequences.append(tuple(current[-3:]))

        from collections import Counter
        counts = Counter(sequences)
        for seq, count in counts.items():
            if count >= 2:  # Same 3-tool pattern repeated
                confidence = min(1.0, 0.5 + count * 0.15)
                if confidence >= self.threshold:
                    related_events = [e for e in tool_execs if e.tool_name in seq]
                    iterations = sorted(set(e.iteration for e in related_events))
                    return AnomalyReport(
                        detector="Stagnation",
                        confidence=confidence,
                        description=f"Stagnation detected: repeating tool sequence {seq} {count} times",
                        evidence=[f"Sequence: {seq}"],
                        affected_iterations=iterations,
                    )
        return None