"""Step handler wiring tests — covers the YAML-step → SkillExecutor bridge."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from animus_forge.skill_bridge import (
    SkillExecutor,
    build_skill_step_handler,
)


@dataclass
class FakeStep:
    id: str = "review-step"
    type: str = "skill"
    params: dict = field(default_factory=dict)


def _build_handler(resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock):
    executor = SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
    )
    return build_skill_step_handler(
        resolver=resolver,
        executor=executor,
        execution_id_provider=lambda: "exec-7",
        workflow_id_provider=lambda: "wf-7",
    )


def test_yaml_step_routes_persona(
    resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
):
    handler = _build_handler(
        resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
    )
    step = FakeStep(params={"skill": "code-reviewer", "tier": "persona", "inputs": {"prompt": "x"}})
    out = handler(step, {})
    assert out["success"] is True
    assert out["output"]["text"] == "MOCK_OUTPUT"


def test_yaml_step_missing_skill_raises(
    resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
):
    from animus_forge.skill_bridge import SkillConfigError

    handler = _build_handler(
        resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
    )
    step = FakeStep(params={"tier": "persona"})
    with pytest.raises(SkillConfigError, match="missing required 'skill'"):
        handler(step, {})


def test_yaml_step_bad_tier_raises(
    resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
):
    from animus_forge.skill_bridge import SkillConfigError

    handler = _build_handler(
        resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
    )
    step = FakeStep(params={"skill": "code-reviewer", "tier": "nope"})
    with pytest.raises(SkillConfigError, match="tier must be one of"):
        handler(step, {})


def test_yaml_step_resolution_error_returns_failure(
    resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
):
    handler = _build_handler(
        resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
    )
    step = FakeStep(params={"skill": "nonexistent", "tier": "persona"})
    out = handler(step, {})
    assert out["success"] is False
    assert out["error_category"] == "skill.resolution"


def test_yaml_step_provider_hint_propagates(
    resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
):
    handler = _build_handler(
        resolver, mock_provider_manager, mock_budget, mock_approval_approve, fake_clock
    )
    step = FakeStep(
        params={
            "skill": "code-reviewer",
            "tier": "persona",
            "inputs": {"prompt": "x"},
            "provider_hint": "anthropic",
        }
    )
    handler(step, {})
    assert mock_provider_manager.last_hint == "anthropic"
