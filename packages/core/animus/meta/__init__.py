"""Meta-Thinker: Strategic oversight layer for Animus Head.

Implements P-20260706-002. Inspired by COMPASS (ACL 2026):
- Runs asynchronously alongside the tactical Head loop
- Monitors execution trajectory for anomalies
- Generates strategic briefs and triggers replanning

Key design: event-driven observer. The Head loop emits events;
Meta-Thinker subscribes, detects patterns, emits signals back.
"""

from animus.meta.anomalies import (
    AnomalyDetector,
    CircularToolUse,
    GoalDrift,
    RepeatedFailures,
    Stagnation,
)
from animus.meta.events import (
    Event,
    IterationStarted,
    LoopCompleted,
    MaxIterationsReached,
    ResponseReceived,
    ToolExecution,
)
from animus.meta.signals import (
    EscalateSignal,
    HaltSignal,
    InjectBriefSignal,
    ReplanSignal,
    Signal,
)
from animus.meta.thinker import (
    MetaThinker,
    MetaThinkerConfig,
    ReplanStrategy,
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
