"""Workflow-tier + nesting depth tests — A10, A14, A14b."""

from __future__ import annotations

from animus_forge.skill_bridge import (
    SkillExecutor,
    StepContext,
)


def _ctx(depth=0):
    return StepContext(
        execution_id="exec-1",
        workflow_id="wf-1",
        step_id="s1",
        nesting_depth=depth,
    )


# ---------------------------------------------------------------------------
# A10 — workflow tier delegates to nested runner
# ---------------------------------------------------------------------------


def test_workflow_tier_invokes_nested_workflow_executor(
    resolver,
    mock_provider_manager,
    mock_budget,
    mock_approval_approve,
    fake_clock,
):
    runner_calls: list = []

    def runner(skill_md_path, inputs, child_ctx):
        runner_calls.append((skill_md_path, inputs, child_ctx))
        return {"steps": ["x", "y"], "total_tokens": 42}

    ex = SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
        nested_workflow_runner=runner,
    )
    skill = resolver.resolve("context-mapping", "workflow")
    result = ex.execute(skill, {"target": "repo"}, _ctx(depth=0))
    assert result.success
    assert result.tokens_used == 42
    assert len(runner_calls) == 1
    # Nesting depth incremented for child
    assert runner_calls[0][2].nesting_depth == 1


# ---------------------------------------------------------------------------
# A14 — depth cap enforced
# ---------------------------------------------------------------------------


def test_nested_workflow_depth_cap_enforced(
    resolver,
    mock_provider_manager,
    mock_budget,
    mock_approval_approve,
    fake_clock,
):
    runner_called = []

    def runner(*a, **kw):  # noqa: ARG001
        runner_called.append(True)
        return {}

    ex = SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
        nested_workflow_runner=runner,
    )
    skill = resolver.resolve("context-mapping", "workflow")
    # At depth=3 with cap=3, we refuse before recursing
    result = ex.execute(skill, {}, _ctx(depth=3))
    assert not result.success
    assert result.error.startswith("skill.nesting_depth_exceeded")
    assert runner_called == []
    # No budget consumed when refused at the gate
    assert len(mock_budget.allocate_calls) == 0


# ---------------------------------------------------------------------------
# A14b — env var override raises the cap
# ---------------------------------------------------------------------------


def test_nested_workflow_depth_override_via_env(
    resolver,
    mock_provider_manager,
    mock_budget,
    mock_approval_approve,
    fake_clock,
    monkeypatch,
):
    monkeypatch.setenv("SKILLBRIDGE_MAX_NESTING_DEPTH", "10")

    def runner(*a, **kw):  # noqa: ARG001
        return {}

    ex = SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
        nested_workflow_runner=runner,
    )
    skill = resolver.resolve("context-mapping", "workflow")
    # Depth 5 — would fail at default 3, but env override allows it
    result = ex.execute(skill, {}, _ctx(depth=5))
    assert result.success


# ---------------------------------------------------------------------------
# Missing runner with workflow tier → graceful error
# ---------------------------------------------------------------------------


def test_workflow_tier_without_runner_returns_failure(
    resolver,
    mock_provider_manager,
    mock_budget,
    mock_approval_approve,
    fake_clock,
):
    ex = SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
        nested_workflow_runner=None,
    )
    skill = resolver.resolve("context-mapping", "workflow")
    result = ex.execute(skill, {}, _ctx())
    assert not result.success
    assert result.error.startswith("skill.no_workflow_runner")
