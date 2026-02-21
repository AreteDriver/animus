"""Automation pipeline — trigger -> conditions -> actions engine."""

from __future__ import annotations

from animus_bootstrap.intelligence.automations.engine import AutomationEngine
from animus_bootstrap.intelligence.automations.models import (
    ActionConfig,
    AutomationResult,
    AutomationRule,
    Condition,
    TriggerConfig,
)

__all__ = [
    "ActionConfig",
    "AutomationEngine",
    "AutomationResult",
    "AutomationRule",
    "Condition",
    "TriggerConfig",
]
