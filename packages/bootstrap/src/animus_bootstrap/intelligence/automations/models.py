"""Automation pipeline data models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class TriggerConfig:
    """What starts an automation rule."""

    type: str  # "message" | "schedule" | "webhook" | "event"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Condition:
    """A condition that must be true for the rule to fire."""

    type: str  # "contains" | "from_channel" | "time_range" | "regex" | "sender_is"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionConfig:
    """An action to execute when the rule fires."""

    type: str  # "reply" | "forward" | "run_tool" | "store_memory" | "webhook"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationRule:
    """A trigger -> conditions -> actions pipeline."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    enabled: bool = True
    trigger: TriggerConfig = field(default_factory=lambda: TriggerConfig(type="message"))
    conditions: list[Condition] = field(default_factory=list)
    actions: list[ActionConfig] = field(default_factory=list)
    cooldown_seconds: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_fired: datetime | None = None


@dataclass
class AutomationResult:
    """Result of evaluating/executing an automation rule."""

    rule_id: str
    rule_name: str
    triggered: bool
    actions_executed: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
