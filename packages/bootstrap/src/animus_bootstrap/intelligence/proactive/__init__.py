"""Proactive engine — scheduled checks, nudges, and reminders."""

from __future__ import annotations

from animus_bootstrap.intelligence.proactive.engine import (
    DEFAULT_OUTCOME_WINDOW_SECONDS,
    ActionObserver,
    NudgePrediction,
    NudgeRecord,
    ProactiveCheck,
    ProactiveEngine,
)
from animus_bootstrap.intelligence.proactive.metrics import (
    HitRateReport,
    hit_rate,
    hit_rate_all,
)
from animus_bootstrap.intelligence.proactive.outcomes import (
    FireRecord,
    OutcomeRecord,
    OutcomeStore,
)
from animus_bootstrap.intelligence.proactive.schedule import (
    ScheduleParser,
    ScheduleResult,
)

__all__ = [
    "DEFAULT_OUTCOME_WINDOW_SECONDS",
    "ActionObserver",
    "FireRecord",
    "HitRateReport",
    "NudgePrediction",
    "NudgeRecord",
    "OutcomeRecord",
    "OutcomeStore",
    "ProactiveCheck",
    "ProactiveEngine",
    "ScheduleParser",
    "ScheduleResult",
    "hit_rate",
    "hit_rate_all",
]
