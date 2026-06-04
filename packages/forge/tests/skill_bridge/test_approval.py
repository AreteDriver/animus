"""Approval pipeline tests — A5, A6, A7."""

from __future__ import annotations

from animus_forge.skill_bridge import (
    APPROVAL_DEFAULT_TIMEOUT_S,
    SkillExecutor,
    StepContext,
)
from animus_forge.skill_bridge.executor import APPROVAL_POLL_INTERVAL_S

from .conftest import MockApprovalStore


def _ctx():
    return StepContext(
        execution_id="exec-1",
        workflow_id="wf-1",
        step_id="s-destructive",
        next_step_id="s-next",
    )


def _build(mock_provider_manager, mock_budget, approval, fake_clock):
    return SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=approval,
        clock=fake_clock,
    )


# ---------------------------------------------------------------------------
# A5 — destructive blocks until approved
# ---------------------------------------------------------------------------


def test_destructive_blocks_until_approved(
    mock_provider_manager,
    mock_budget,
    resolver,
    fake_clock,
):
    approval = MockApprovalStore(sequence=["pending", "pending", "approved"])
    ex = _build(mock_provider_manager, mock_budget, approval, fake_clock)
    skill = resolver.resolve("demolish", "agent")
    result = ex.execute(skill, {}, _ctx())
    assert result.success
    # Polled twice before approval landed
    assert len(fake_clock.sleeps) == 2
    assert len(approval.tokens_created) == 1


# ---------------------------------------------------------------------------
# A6 — denied path returns failure with namespace
# ---------------------------------------------------------------------------


def test_destructive_denied_returns_denied_status(
    mock_provider_manager,
    mock_budget,
    resolver,
    fake_clock,
    mock_provider,
):
    approval = MockApprovalStore(sequence=["rejected"])
    ex = _build(mock_provider_manager, mock_budget, approval, fake_clock)
    skill = resolver.resolve("demolish", "agent")
    result = ex.execute(skill, {}, _ctx())
    assert not result.success
    assert result.error.startswith("skill.approval_denied")
    # No provider call — denial happens before execution
    assert len(mock_provider.calls) == 0
    # No budget allocation either (gate blocks before allocate)
    assert len(mock_budget.allocate_calls) == 0


# ---------------------------------------------------------------------------
# A7 — approval timeout
# ---------------------------------------------------------------------------


def test_destructive_approval_timeout(
    mock_provider_manager,
    mock_budget,
    resolver,
    fake_clock,
):
    # FakeClock with very large step so we advance past the deadline fast
    fake_clock._step = APPROVAL_DEFAULT_TIMEOUT_S  # one sleep blows the budget
    approval = MockApprovalStore(sequence=["pending", "pending", "pending"])
    ex = _build(mock_provider_manager, mock_budget, approval, fake_clock)
    skill = resolver.resolve("demolish", "agent")
    result = ex.execute(skill, {}, _ctx())
    assert not result.success
    assert result.error.startswith("skill.approval_timeout")


# ---------------------------------------------------------------------------
# Non-destructive skills do NOT touch approval_store
# ---------------------------------------------------------------------------


def test_safe_skill_skips_approval(
    mock_provider_manager,
    mock_budget,
    resolver,
    fake_clock,
):
    approval = MockApprovalStore(sequence=["approved"])
    ex = _build(mock_provider_manager, mock_budget, approval, fake_clock)
    skill = resolver.resolve("code-reviewer", "persona")  # risk_level safe
    result = ex.execute(skill, {"prompt": "x"}, _ctx())
    assert result.success
    assert len(approval.tokens_created) == 0


# ---------------------------------------------------------------------------
# Persona registered destructive in registry overrides default safe
# ---------------------------------------------------------------------------


def test_registry_destructive_persona_triggers_approval(
    mock_provider_manager,
    mock_budget,
    resolver,
    fake_clock,
):
    approval = MockApprovalStore(sequence=["approved"])
    ex = _build(mock_provider_manager, mock_budget, approval, fake_clock)
    skill = resolver.resolve("demolisher", "persona")
    assert skill.risk_level == "destructive"
    ex.execute(skill, {"prompt": "x"}, _ctx())
    assert len(approval.tokens_created) == 1


# ---------------------------------------------------------------------------
# Poll cadence matches APPROVAL_POLL_INTERVAL_S
# ---------------------------------------------------------------------------


def test_poll_cadence(
    mock_provider_manager,
    mock_budget,
    resolver,
    fake_clock,
):
    approval = MockApprovalStore(sequence=["pending", "approved"])
    ex = _build(mock_provider_manager, mock_budget, approval, fake_clock)
    skill = resolver.resolve("demolish", "agent")
    ex.execute(skill, {}, _ctx())
    assert fake_clock.sleeps == [APPROVAL_POLL_INTERVAL_S]
