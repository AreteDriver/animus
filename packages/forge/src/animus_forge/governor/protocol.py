"""Pydantic v2 mirrors of the 5 consumer-side Governor JSON schemas.

**Why local models, not in-process import.** The adapter shells out to
``alg`` and never imports ``animus_loop_governor.*``. Local mirrors
preserve that boundary — a Governor version bump cannot break the
adapter at import time, only at runtime.

The 5 mirrored schemas (``completion-decision``, ``watchdog-report``,
``run-ledger``, ``run-event``, plus the run-state contract used by
``ensure_run``). Drift is caught by ``tests/test_governor/`` which
round-trips each fixture against the corresponding local model.

``TaskContract`` is **not** mirrored — the adapter passes contract YAML
paths through to ``alg compile``/``start`` and never reads the contract
itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """Local base: extra fields forbidden, assignment-validated.

    Matches the Governor's ``StrictModel`` semantics so we reject
    schema drift at parse time rather than silently dropping fields.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# CompletionDecision — completion-decision.schema.json
# ---------------------------------------------------------------------------


class CompletionDecision(_StrictModel):
    """Outcome of ``alg verify``.

    ``done=True`` is the only acceptable completion signal and requires
    at least one ``reason`` (the Governor's contract: a successful
    completion must document *why* it succeeded). ``done=False`` maps
    to :class:`VerifyDeniedError` and drives the retry path.
    """

    done: bool
    reasons: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    blocking_findings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _done_true_requires_reasons(self) -> CompletionDecision:
        if self.done and not self.reasons:
            raise ValueError(
                "CompletionDecision with done=true must list at least "
                "one reason (Governor contract: completion must "
                "document why it succeeded)"
            )
        return self


# ---------------------------------------------------------------------------
# WatchdogReport + WatchdogFinding — watchdog-report.schema.json
# ---------------------------------------------------------------------------


WatchdogSeverity = Literal["info", "warning", "error", "halt"]


class WatchdogFinding(_StrictModel):
    code: str
    severity: WatchdogSeverity
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class WatchdogReport(_StrictModel):
    drift_score: float = Field(ge=0.0, le=1.0)
    stagnation: bool
    findings: list[WatchdogFinding] = Field(default_factory=list)
    required_action: str | None = None


# ---------------------------------------------------------------------------
# RunEvent — run-event.schema.json
# ---------------------------------------------------------------------------


GovernorRole = Literal[
    "planner",
    "worker",
    "inspector",
    "test_operator",
    "adversarial_reviewer",
    "release_authority",
    "system",
]


class RunEvent(_StrictModel):
    sequence: int
    run_id: str
    timestamp: datetime | None = None
    actor_role: GovernorRole
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    contract_hash: str
    ledger_version: int


# ---------------------------------------------------------------------------
# RunLedger — run-ledger.schema.json
# ---------------------------------------------------------------------------


class AcceptanceState(_StrictModel):
    satisfied: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class RunMetrics(_StrictModel):
    iterations: int = 0
    failed_attempts: dict[str, int] = Field(default_factory=dict)
    commands_run: int = 0
    files_changed_count: int = 0
    acceptance_satisfied_count: int = 0
    last_progress_at: datetime | None = None
    drift_score: float = 0.0
    stagnation_detected: bool = False


RunPhase = Literal[
    "created",
    "contracted",
    "planned",
    "implementation",
    "blocked",
    "escalated",
    "review",
    "complete",
    "failed",
    "aborted",
]


class RunLedger(_StrictModel):
    ledger_version: int = 1
    run_id: str
    task_id: str
    contract_hash: str
    phase: RunPhase = "contracted"
    current_goal: str = ""
    completed: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    requirement_map: dict[str, list[str]] = Field(default_factory=dict)
    acceptance_status: dict[str, AcceptanceState] = Field(
        default_factory=dict
    )
    open_escalations: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    started_at: datetime | None = None
    updated_at: datetime | None = None


__all__ = [
    "AcceptanceState",
    "CompletionDecision",
    "GovernorRole",
    "RunEvent",
    "RunLedger",
    "RunMetrics",
    "RunPhase",
    "WatchdogFinding",
    "WatchdogReport",
    "WatchdogSeverity",
]
