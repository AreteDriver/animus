"""Executor tests — A1, A4, A12, A13, A17, A18, A19, A20, A21."""

from __future__ import annotations

import concurrent.futures

import pytest

from animus_forge.skill_bridge import (
    TIER_OUTPUT_CEILING,
    SkillCallResult,
    SkillExecutor,
    StepContext,
)
from animus_forge.skill_bridge.executor import (
    _chars_to_tokens,
    _estimate_allocation,
)


def _ctx(step_id="s1", **kw):
    return StepContext(
        execution_id="exec-1",
        workflow_id="wf-1",
        step_id=step_id,
        **kw,
    )


# ---------------------------------------------------------------------------
# A1 — persona step executes via single provider call (also OQ1)
# ---------------------------------------------------------------------------


def test_persona_step_executes(executor, resolver, mock_provider, mock_budget):
    skill = resolver.resolve("code-reviewer", "persona")
    result = executor.execute(skill, {"prompt": "review my file"}, _ctx())
    assert result.success
    assert result.output["text"] == "MOCK_OUTPUT"
    assert len(mock_provider.calls) == 1
    # Budget pre-allocated AND recorded
    assert len(mock_budget.allocate_calls) == 1
    assert len(mock_budget.usage_calls) == 1


def test_persona_multi_voice_single_call(executor, resolver, mock_provider):
    """OQ1 — multi-perspective personas are still one Provider call."""
    skill = resolver.resolve("board-of-advisors", "persona")
    executor.execute(skill, {"prompt": "should we ship?"}, _ctx())
    assert len(mock_provider.calls) == 1


# ---------------------------------------------------------------------------
# A4 — budget allocate + record_usage on every code path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inputs", [{"prompt": "hello"}, {"a": 1}, {}])
def test_budget_recorded_on_success(executor, resolver, mock_budget, inputs):
    skill = resolver.resolve("code-reviewer", "persona")
    executor.execute(skill, inputs, _ctx())
    assert len(mock_budget.allocate_calls) == 1
    assert len(mock_budget.usage_calls) == 1


def test_budget_recorded_on_provider_error(executor, resolver, mock_provider, mock_budget):
    mock_provider.raise_on_complete = RuntimeError("provider exploded")
    skill = resolver.resolve("code-reviewer", "persona")
    result = executor.execute(skill, {"prompt": "x"}, _ctx())
    assert not result.success
    assert result.error.startswith("skill.provider_error")
    assert len(mock_budget.allocate_calls) == 1
    assert len(mock_budget.usage_calls) == 1


def test_budget_NOT_consumed_on_input_validation_failure(executor, resolver, mock_budget):
    """A2/A3 — agent input validation runs before allocate()."""
    skill = resolver.resolve("api-client", "agent")
    result = executor.execute(skill, {}, _ctx())  # missing required
    assert not result.success
    assert result.error.startswith("skill.invalid_input")
    assert len(mock_budget.allocate_calls) == 0
    assert len(mock_budget.usage_calls) == 0


# ---------------------------------------------------------------------------
# A2 / A3 — schema input validation
# ---------------------------------------------------------------------------


def test_agent_missing_required_input_rejected(executor, resolver):
    skill = resolver.resolve("api-client", "agent")
    result = executor.execute(skill, {"operation": "request"}, _ctx())  # missing url
    assert not result.success
    assert "missing required" in result.error.lower()


def test_agent_enum_violation_rejected(executor, resolver):
    skill = resolver.resolve("api-client", "agent")
    result = executor.execute(
        skill,
        {"operation": "INVALID_OP", "url": "https://x"},
        _ctx(),
    )
    assert not result.success
    assert "enum" in result.error.lower()


# ---------------------------------------------------------------------------
# A12 — provider_hint override
# ---------------------------------------------------------------------------


def test_provider_hint_overrides_default(executor, resolver, mock_provider_manager):
    skill = resolver.resolve("code-reviewer", "persona")
    executor.execute(skill, {"prompt": "x"}, _ctx(provider_hint="ollama"))
    assert mock_provider_manager.last_hint == "ollama"


# ---------------------------------------------------------------------------
# A13 — concurrent invocation safety
# ---------------------------------------------------------------------------


def test_concurrent_skill_calls(executor, resolver, mock_budget):
    skill = resolver.resolve("code-reviewer", "persona")

    def _call(i):
        return executor.execute(skill, {"prompt": f"call-{i}"}, _ctx(step_id=f"s-{i}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_call, range(10)))

    assert all(r.success for r in results)
    assert len(mock_budget.allocate_calls) == 10
    assert len(mock_budget.usage_calls) == 10


# ---------------------------------------------------------------------------
# A17 — agent WITHOUT output.schema.yaml returns free-form text
# ---------------------------------------------------------------------------


def test_agent_without_output_schema_returns_free_form(executor, resolver):
    skill = resolver.resolve("api-client", "agent")
    result = executor.execute(
        skill,
        {"operation": "request", "url": "https://x"},
        _ctx(),
    )
    assert result.success
    assert result.output.get("text") == "MOCK_OUTPUT"


# ---------------------------------------------------------------------------
# A18 — agent WITH output.schema.yaml triggers SCHEMA_DRIFT on empty output
# ---------------------------------------------------------------------------


def test_agent_with_output_schema_enforces_it(executor, resolver, mock_provider):
    mock_provider.response_text = ""  # force empty → schema drift
    skill = resolver.resolve("scorer", "agent")
    result = executor.execute(skill, {}, _ctx())
    assert not result.success
    assert result.error.startswith("skill.schema_drift")


# ---------------------------------------------------------------------------
# A19 — allocate uses tier-aware ceiling
# ---------------------------------------------------------------------------


def test_allocate_uses_input_estimate_plus_tier_ceiling(executor, resolver, mock_budget):
    skill = resolver.resolve("code-reviewer", "persona")
    user_prompt = "review my file"
    executor.execute(skill, {"prompt": user_prompt}, _ctx())
    allocated_tokens, _agent_id = mock_budget.allocate_calls[0]
    assert allocated_tokens > TIER_OUTPUT_CEILING["persona"]
    assert allocated_tokens >= TIER_OUTPUT_CEILING["persona"]


# ---------------------------------------------------------------------------
# A20 — record_usage reconciles to provider response
# ---------------------------------------------------------------------------


def test_record_usage_reconciles_to_actual(executor, resolver, mock_budget):
    skill = resolver.resolve("code-reviewer", "persona")
    executor.execute(skill, {"prompt": "x"}, _ctx())
    usage = mock_budget.usage_calls[0]
    assert usage["input_tokens"] == 50  # from MockCompletionResponse
    assert usage["output_tokens"] == 200


# ---------------------------------------------------------------------------
# A21 — fallback to chars-to-tokens math is exposed
# ---------------------------------------------------------------------------


def test_provider_without_estimator_falls_back_to_chars():
    """No Provider has estimate_input_tokens today; the math is just
    len(prompt) // 4, exercised here for explicit OQ5 verification."""
    text = "x" * 400
    assert _chars_to_tokens(text) == 100


def test_estimate_allocation_persona():
    user, system = "user", "system"
    result = _estimate_allocation(user, system, "persona")
    expected = _chars_to_tokens(user) + _chars_to_tokens(system) + TIER_OUTPUT_CEILING["persona"]
    assert result == expected


# ---------------------------------------------------------------------------
# Budget refused → BUDGET_EXHAUSTED status
# ---------------------------------------------------------------------------


def test_budget_refused_returns_budget_exhausted(
    mock_provider_manager,
    mock_approval_approve,
    resolver,
    fake_clock,
):
    refusing_budget = type(
        "RefBudget",
        (),
        {
            "allocate": lambda self, t, agent_id=None: False,  # noqa: ARG005
            "record_usage": lambda self, agent_id, **kw: None,  # noqa: ARG005
        },
    )()
    ex = SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=refusing_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
    )
    skill = resolver.resolve("code-reviewer", "persona")
    result = ex.execute(skill, {"prompt": "x"}, _ctx())
    assert not result.success
    assert result.error.startswith("skill.budget_exhausted")


# ---------------------------------------------------------------------------
# SkillCallResult shape sanity
# ---------------------------------------------------------------------------


def test_callresult_dataclass():
    r = SkillCallResult(success=True, output={"text": "x"}, tokens_used=12)
    assert r.success and r.tokens_used == 12 and r.error is None
