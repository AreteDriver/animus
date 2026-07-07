"""Session controller — lifecycle management for HeadREPL sessions.

Wraps a HeadREPL with token-utilization and wall-clock monitoring.
When thresholds are breached, gracefully finalizes the session
(generates a summary, checkpoints state) and optionally restarts
a new session bootstrapped from the checkpoint.

Intended for long-running local Ollama sessions where context-window
drift or session staleness would degrade output quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionLifecycleEvent(Enum):
    """Lifecycle states for a managed session."""

    RUNNING = auto()
    WRAPPING_UP = auto()
    CHECKPOINTING = auto()
    RESTARTING = auto()
    FINISHED = auto()


@dataclass
class SessionPolicy:
    """Policy governing session lifecycle limits.

    Args:
        wrapup_threshold: Token utilization fraction (0.0–1.0) that triggers
            graceful finalize. 1.0 disables token-based wrap-up.
        session_timer: Max wall-clock duration. None disables timer.
        auto_restart: Whether to spin up a new session after wrap-up.
        wrapup_prompt: System prompt injected before the wrap-up turn.
        model_overrides: Per-model policy overrides keyed by model name.
    """

    wrapup_threshold: float = 0.96
    session_timer: timedelta | None = field(default_factory=lambda: timedelta(minutes=30))
    auto_restart: bool = True
    wrapup_prompt: str = (
        "You are approaching the session limit. "
        "Please provide a concise summary of: (1) our current task, "
        "(2) key decisions made, and (3) what should happen next."
    )
    model_overrides: dict[str, "SessionPolicy"] = field(default_factory=dict)

    def resolve_for_model(self, model: str) -> SessionPolicy:
        """Return a policy with model-specific overrides applied."""
        if model in self.model_overrides:
            base = self.model_overrides[model]
            return SessionPolicy(
                wrapup_threshold=base.wrapup_threshold,
                session_timer=base.session_timer,
                auto_restart=base.auto_restart,
                wrapup_prompt=base.wrapup_prompt or self.wrapup_prompt,
                model_overrides={},  # Prevent nested override recursion
            )
        return self

    @property
    def token_wrapup_enabled(self) -> bool:
        return self.wrapup_threshold < 1.0

    @property
    def timer_enabled(self) -> bool:
        return self.session_timer is not None and self.session_timer.total_seconds() > 0


@dataclass
class SessionTelemetry:
    """Snapshot of session telemetry at a point in time."""

    session_id: str
    event: SessionLifecycleEvent
    utilization_percent: float
    elapsed_seconds: float
    turns: int
    timestamp: datetime
    message: str = ""


class SessionController:
    """Manages a HeadREPL session lifecycle with limits and graceful wrap-up.

    Can operate in two modes:

    1. **Embedded** (default): The controller hooks into HeadREPL's turn loop
       via ``_check_session_limits()`` and ``_graceful_finalize()``.
    2. **Daemon watchdog**: A background thread polls a running REPL and
       triggers wrap-up when thresholds are breached.

    The controller itself is stateless with respect to conversation content;
    all state lives in the HeadREPL and HeadCheckpointStore.
    """

    DEFAULT_WRAPUP_PROMPT: str = (
        "You are approaching the session limit. "
        "Please provide a concise summary of: (1) our current task, "
        "(2) key decisions made, and (3) what should happen next."
    )

    def __init__(
        self,
        policy: SessionPolicy | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.policy = policy or SessionPolicy()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._telemetry_log: list[SessionTelemetry] = []
        self._active: bool = False

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def log_event(
        self,
        session_id: str,
        event: SessionLifecycleEvent,
        utilization_percent: float,
        elapsed_seconds: float,
        turns: int,
        message: str = "",
    ) -> SessionTelemetry:
        """Record a lifecycle event."""
        telemetry = SessionTelemetry(
            session_id=session_id,
            event=event,
            utilization_percent=utilization_percent,
            elapsed_seconds=elapsed_seconds,
            turns=turns,
            timestamp=datetime.now(UTC),
            message=message,
        )
        self._telemetry_log.append(telemetry)
        logger.info(
            "Session %s: %s (util=%.1f%% elapsed=%.0fs turns=%d) %s",
            session_id,
            event.name,
            utilization_percent,
            elapsed_seconds,
            turns,
            message,
        )
        return telemetry

    def get_telemetry(self, session_id: str | None = None) -> list[SessionTelemetry]:
        """Return telemetry entries, optionally filtered by session."""
        if session_id is None:
            return list(self._telemetry_log)
        return [t for t in self._telemetry_log if t.session_id == session_id]

    def get_summary_stats(self) -> dict[str, Any]:
        """Return aggregate statistics across all logged sessions."""
        if not self._telemetry_log:
            return {}

        utilizations = [t.utilization_percent for t in self._telemetry_log if t.event == SessionLifecycleEvent.WRAPPING_UP]
        elapsed_times = [t.elapsed_seconds for t in self._telemetry_log if t.event == SessionLifecycleEvent.WRAPPING_UP]
        restarts = sum(1 for t in self._telemetry_log if t.event == SessionLifecycleEvent.RESTARTING)

        return {
            "total_sessions": len({t.session_id for t in self._telemetry_log}),
            "total_wrapups": len(utilizations),
            "total_restarts": restarts,
            "avg_utilization_at_wrapup": sum(utilizations) / len(utilizations) if utilizations else 0.0,
            "min_utilization_at_wrapup": min(utilizations) if utilizations else 0.0,
            "max_utilization_at_wrapup": max(utilizations) if utilizations else 0.0,
            "avg_elapsed_seconds": sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0,
        }

    # ------------------------------------------------------------------
    # Limit checking
    # ------------------------------------------------------------------

    def check_limits(
        self,
        session_id: str,
        utilization_percent: float,
        elapsed_seconds: float,
        turns: int,
    ) -> tuple[bool, str]:
        """Check whether any session limit has been breached.

        Returns:
            (breached, reason) — reason is empty if not breached.
        """
        policy = self.policy

        if policy.token_wrapup_enabled and utilization_percent >= policy.wrapup_threshold * 100:
            return True, (
                f"token utilization {utilization_percent:.1f}% >= "
                f"threshold {policy.wrapup_threshold * 100:.1f}%"
            )

        if policy.timer_enabled and policy.session_timer:
            if elapsed_seconds >= policy.session_timer.total_seconds():
                return True, (
                    f"session timer expired ({elapsed_seconds:.0f}s >= "
                    f"{policy.session_timer.total_seconds():.0f}s)"
                )

        return False, ""

    # ------------------------------------------------------------------
    # Graceful finalize
    # ------------------------------------------------------------------

    def should_finalize(
        self,
        session_id: str,
        utilization_percent: float,
        elapsed_seconds: float,
        turns: int,
    ) -> tuple[bool, str]:
        """Public alias for check_limits with consistent return shape."""
        breached, reason = self.check_limits(session_id, utilization_percent, elapsed_seconds, turns)
        return breached, reason
