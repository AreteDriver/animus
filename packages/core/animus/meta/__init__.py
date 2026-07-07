"""Meta-Thinker: Strategic oversight layer for Animus Head.

Implements P-20260706-002. Inspired by COMPASS (ACL 2026):
- Runs asynchronously alongside the tactical Head loop
- Monitors execution trajectory for anomalies
- Generates strategic briefs and triggers replanning

Key design: event-driven observer. The Head loop emits events;
Meta-Thinker subscribes, detects patterns, emits signals back.
"""

from animus.meta.thinker import (
    MetaThinker,
    MetaThinkerConfig,
    ReplanStrategy,
)
from animus.meta.events import (
    Event,
    IterationStarted,
    ToolExecution,
    ResponseReceived,
    LoopCompleted,
    MaxIterationsReached,
)
from animus.meta.anomalies import (
    AnomalyDetector,
    CircularToolUse,
    RepeatedFailures,
    GoalDrift,
    Stagnation,
)
from animus.meta.signals import (
    Signal,
    ReplanSignal,
    InjectBriefSignal,
    EscalateSignal,
    HaltSignal,
)

__all__ = [
    "MetaThinker",
    "MetaThinkerConfig",
    "ReplanStrategy",
    "Event",
    "IterationStarted",
    "ToolExecution",
    "ResponseReceived",
    "LoopCompleted",
    "MaxIterationsReached",
    "AnomalyDetector",
    "CircularToolUse",
    "RepeatedFailures",
    "GoalDrift",
    "Stagnation",
    "Signal",
    "ReplanSignal",
    "InjectBriefSignal",
    "EscalateSignal",
    "HaltSignal",
]