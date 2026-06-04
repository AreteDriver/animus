"""SkillExecutor — executes a ResolvedSkill via the Forge Provider stack.

Wraps every call in BudgetManager.allocate + record_usage. For agents
declared risk_level=destructive, routes through ApprovalStore and
blocks until decision (or C4 timeout).

Status mapping to Forge's StepStatus (SUCCESS / FAILED / AWAITING_APPROVAL):
the spec's richer error categories land in StepResult.error as a
namespaced string so callers can match on the prefix.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    SkillApprovalDeniedError,
    SkillApprovalTimeoutError,
    SkillBudgetError,
    SkillInputError,
    SkillNestingDepthError,
    SkillOutputDriftError,
)
from .types import TIER_OUTPUT_CEILING, ResolvedSkill

logger = logging.getLogger(__name__)

DEFAULT_MAX_NESTING_DEPTH = 3
APPROVAL_POLL_INTERVAL_S = 0.1
APPROVAL_DEFAULT_TIMEOUT_S = 30 * 60  # C4 — 30 minutes


@dataclass
class StepContext:
    """Per-call execution context passed to SkillExecutor.

    Carries identifiers required by BudgetManager / ApprovalStore plus
    the running nesting-depth counter for workflow-tier recursion (R17).
    """

    execution_id: str
    workflow_id: str
    step_id: str
    next_step_id: str = ""
    nesting_depth: int = 0
    provider_hint: str | None = None


@dataclass
class SkillCallResult:
    """Outcome of a single SkillExecutor.execute() call.

    Returned to the step handler, which translates it into a Forge
    StepResult.
    """

    success: bool
    output: dict
    error: str | None = None
    awaiting_approval: bool = False
    tokens_used: int = 0


class SkillExecutor:
    """Execute a ResolvedSkill end-to-end with budget + approval enforcement."""

    def __init__(
        self,
        *,
        provider_manager: Any,
        budget: Any,
        approval_store: Any = None,
        nested_workflow_runner: Any = None,
        clock: Any = time,
    ) -> None:
        """Construct an executor.

        provider_manager: must expose .get_provider(hint=None) -> Provider
            and Provider.complete(CompletionRequest) -> CompletionResponse.
        budget: BudgetManager-like (allocate(tokens, agent_id) -> bool;
            record_usage(agent_id, *, input_tokens=..., output_tokens=...,
            model=..., operation=...) -> Any).
        approval_store: ApprovalStore-like with create_token() and
            get_by_token(). Optional — required only when calling a
            destructive-risk skill.
        nested_workflow_runner: callable invoked for tier=workflow steps.
            Signature: (skill_md_path, inputs, parent_context) -> dict.
            Optional — required only when calling a workflow-tier skill.
        clock: indirection for testability (.time(), .sleep(s)).
        """
        self.provider_manager = provider_manager
        self.budget = budget
        self.approval_store = approval_store
        self.nested_workflow_runner = nested_workflow_runner
        self._clock = clock

    @property
    def max_nesting_depth(self) -> int:
        raw = os.environ.get("SKILLBRIDGE_MAX_NESTING_DEPTH")
        if raw:
            with suppress(ValueError):
                value = int(raw)
                if value >= 1:
                    return value
        return DEFAULT_MAX_NESTING_DEPTH

    def execute(
        self,
        skill: ResolvedSkill,
        inputs: dict,
        ctx: StepContext,
    ) -> SkillCallResult:
        """Dispatch by tier."""
        # R17 — nesting cap on workflow tier only; agents/personas never recurse.
        if skill.tier == "workflow" and ctx.nesting_depth >= self.max_nesting_depth:
            return _err(
                "skill.nesting_depth_exceeded",
                f"workflow nesting depth {ctx.nesting_depth} >= cap {self.max_nesting_depth}",
            )

        try:
            if skill.tier == "persona":
                return self._execute_persona(skill, inputs, ctx)
            if skill.tier == "agent":
                return self._execute_agent(skill, inputs, ctx)
            return self._execute_workflow(skill, inputs, ctx)
        except SkillInputError as exc:
            return _err("skill.invalid_input", str(exc))
        except SkillOutputDriftError as exc:
            return _err("skill.schema_drift", str(exc), tokens_used=getattr(exc, "tokens_used", 0))
        except SkillBudgetError as exc:
            return _err("skill.budget_exhausted", str(exc))
        except SkillApprovalDeniedError as exc:
            return _err("skill.approval_denied", str(exc))
        except SkillApprovalTimeoutError as exc:
            return _err("skill.approval_timeout", str(exc))
        except SkillNestingDepthError as exc:
            return _err("skill.nesting_depth_exceeded", str(exc))

    def _execute_persona(
        self,
        skill: ResolvedSkill,
        inputs: dict,
        ctx: StepContext,
    ) -> SkillCallResult:
        approval_gate = self._maybe_request_approval(skill, ctx)
        if approval_gate is not None:
            return approval_gate

        provider = self._select_provider(ctx.provider_hint)
        system_prompt = _read_text(skill.skill_md_path)
        user_prompt = _format_user_prompt(inputs)

        tokens_to_allocate = _estimate_allocation(user_prompt, system_prompt, "persona")
        self._allocate_or_raise(tokens_to_allocate, ctx.step_id)

        return self._call_provider(provider, system_prompt, user_prompt, "persona", ctx)

    def _execute_agent(
        self,
        skill: ResolvedSkill,
        inputs: dict,
        ctx: StepContext,
    ) -> SkillCallResult:
        # R8 — validate inputs BEFORE any budget allocation
        _validate_inputs(skill, inputs)

        approval_gate = self._maybe_request_approval(skill, ctx)
        if approval_gate is not None:
            return approval_gate

        provider = self._select_provider(ctx.provider_hint)
        system_prompt = _read_text(skill.skill_md_path)
        user_prompt = _format_user_prompt(inputs)

        tokens_to_allocate = _estimate_allocation(user_prompt, system_prompt, "agent")
        self._allocate_or_raise(tokens_to_allocate, ctx.step_id)

        result = self._call_provider(provider, system_prompt, user_prompt, "agent", ctx)

        # R6 / R18 — output schema enforcement is opt-in
        if result.success and skill.schema and skill.schema.has_output_schema:
            try:
                _validate_output(skill, result.output.get("text", ""))
            except SkillOutputDriftError as exc:
                exc.tokens_used = result.tokens_used  # propagate cost info
                raise

        return result

    def _execute_workflow(
        self,
        skill: ResolvedSkill,
        inputs: dict,
        ctx: StepContext,
    ) -> SkillCallResult:
        if self.nested_workflow_runner is None:
            return _err(
                "skill.no_workflow_runner",
                "tier=workflow requires nested_workflow_runner to be configured",
            )

        approval_gate = self._maybe_request_approval(skill, ctx)
        if approval_gate is not None:
            return approval_gate

        # Workflow-tier reserves 0 tokens at the parent level; nested
        # steps allocate per-call.
        self._allocate_or_raise(0, ctx.step_id)

        child_ctx = StepContext(
            execution_id=ctx.execution_id,
            workflow_id=ctx.workflow_id,
            step_id=ctx.step_id,
            next_step_id=ctx.next_step_id,
            nesting_depth=ctx.nesting_depth + 1,
            provider_hint=ctx.provider_hint,
        )

        try:
            child_output = self.nested_workflow_runner(
                skill.skill_md_path,
                inputs,
                child_ctx,
            )
        except Exception as exc:  # noqa: BLE001 — nested errors map to failure result
            return _err("skill.workflow_failed", f"nested workflow error: {exc}")

        tokens_used = (
            int(child_output.get("total_tokens", 0)) if isinstance(child_output, dict) else 0
        )
        with suppress(Exception):
            self.budget.record_usage(
                agent_id=ctx.step_id,
                operation="skill:workflow",
                output_tokens=tokens_used,
            )

        return SkillCallResult(
            success=True,
            output=child_output if isinstance(child_output, dict) else {"result": child_output},
            tokens_used=tokens_used,
        )

    # --- helpers ---------------------------------------------------------

    def _select_provider(self, hint: str | None):
        if hint and hasattr(self.provider_manager, "get_provider"):
            return self.provider_manager.get_provider(hint)
        if hasattr(self.provider_manager, "get_provider"):
            return self.provider_manager.get_provider()
        return self.provider_manager  # tests may pass a raw provider

    def _allocate_or_raise(self, tokens: int, agent_id: str) -> None:
        if not self.budget.allocate(tokens, agent_id=agent_id):
            raise SkillBudgetError(f"budget refused {tokens} tokens for agent_id={agent_id}")

    def _maybe_request_approval(
        self,
        skill: ResolvedSkill,
        ctx: StepContext,
    ) -> SkillCallResult | None:
        """Returns an AWAITING_APPROVAL SkillCallResult if the approval
        is still pending after a configured wait; otherwise None
        (= approved, proceed) or raises Denied/Timeout.
        """
        if skill.risk_level != "destructive":
            return None
        if self.approval_store is None:
            raise SkillApprovalDeniedError(
                f"approval_store not configured but skill {skill.name!r} is destructive"
            )

        token = self.approval_store.create_token(
            execution_id=ctx.execution_id,
            workflow_id=ctx.workflow_id,
            step_id=ctx.step_id,
            next_step_id=ctx.next_step_id or ctx.step_id,
            prompt=f"Approve destructive skill {skill.name!r}?",
            preview={"skill": skill.name, "tier": skill.tier},
        )

        deadline = self._clock.time() + APPROVAL_DEFAULT_TIMEOUT_S
        while True:
            record = self.approval_store.get_by_token(token)
            if record is None:
                raise SkillApprovalTimeoutError(f"approval token {token} expired before decision")
            status = record.get("status")
            if status == "approved":
                return None
            if status == "rejected":
                raise SkillApprovalDeniedError(f"approval rejected for skill {skill.name!r}")
            if self._clock.time() >= deadline:
                raise SkillApprovalTimeoutError(
                    f"approval timeout after {APPROVAL_DEFAULT_TIMEOUT_S}s"
                )
            self._clock.sleep(APPROVAL_POLL_INTERVAL_S)

    def _call_provider(
        self,
        provider,
        system_prompt: str,
        user_prompt: str,
        tier_for_metadata: str,
        ctx: StepContext,
    ) -> SkillCallResult:
        from animus_forge.providers.base import CompletionRequest

        request = CompletionRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            agent_id=ctx.step_id,
            workflow_id=ctx.workflow_id,
            metadata={"skill_bridge_tier": tier_for_metadata},
        )
        try:
            response = provider.complete(request)
        except Exception as exc:  # noqa: BLE001 — wrap provider failures
            # Record best-effort zero-output usage so budget is honest
            with suppress(Exception):
                self.budget.record_usage(
                    agent_id=ctx.step_id,
                    operation=f"skill:{tier_for_metadata}:provider_error",
                    input_tokens=_chars_to_tokens(user_prompt + system_prompt),
                )
            return _err("skill.provider_error", f"{type(exc).__name__}: {exc}")

        # Record actual usage (R4)
        with suppress(Exception):
            self.budget.record_usage(
                agent_id=ctx.step_id,
                operation=f"skill:{tier_for_metadata}",
                input_tokens=int(getattr(response, "input_tokens", 0) or 0),
                output_tokens=int(getattr(response, "output_tokens", 0) or 0),
                model=getattr(response, "model", None),
            )

        return SkillCallResult(
            success=True,
            output={"text": response.content, "model": response.model},
            tokens_used=int(getattr(response, "tokens_used", 0) or 0),
        )


# --- module-level helpers ------------------------------------------------


def _err(category: str, message: str, *, tokens_used: int = 0) -> SkillCallResult:
    return SkillCallResult(
        success=False,
        output={"error_category": category},
        error=f"{category}: {message}",
        tokens_used=tokens_used,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise SkillInputError(f"could not read SKILL.md at {path}: {exc}") from exc


def _format_user_prompt(inputs: dict) -> str:
    """Render workflow YAML inputs as a single user-prompt block.

    For now: stable key-sorted YAML-ish dump. Personas typically supply
    free-text via inputs.prompt; agents supply structured fields.
    """
    if not inputs:
        return "(no inputs provided)"
    if isinstance(inputs.get("prompt"), str) and len(inputs) == 1:
        return inputs["prompt"]
    lines = [f"{k}: {inputs[k]}" for k in sorted(inputs)]
    return "\n".join(lines)


def _chars_to_tokens(text: str) -> int:
    """Provider-agnostic token approximation (~4 chars / token, OQ5 fallback)."""
    return max(1, len(text) // 4)


def _estimate_allocation(user_prompt: str, system_prompt: str, tier: str) -> int:
    """OQ5 — input estimate + tier static output ceiling."""
    in_tokens = _chars_to_tokens(user_prompt) + _chars_to_tokens(system_prompt)
    return in_tokens + TIER_OUTPUT_CEILING.get(tier, 0)


def _validate_inputs(skill: ResolvedSkill, inputs: dict) -> None:
    """Minimal R8 validation: required, enum, basic type.

    Schema format mirrors ai-skills/agents/*/schema.yaml — each input
    is keyed by name and has optional fields: type, required, enum.
    """
    if skill.schema is None:
        return
    for field_name, spec in (skill.schema.inputs or {}).items():
        if not isinstance(spec, dict):
            continue
        present = field_name in inputs
        if spec.get("required") and not present:
            raise SkillInputError(f"missing required input: {field_name!r}")
        if not present:
            continue
        value = inputs[field_name]
        enum_values = spec.get("enum")
        if enum_values and value not in enum_values:
            raise SkillInputError(
                f"invalid enum value for {field_name!r}: {value!r} not in {enum_values}"
            )
        declared_type = spec.get("type")
        if declared_type and not _type_matches(value, declared_type):
            raise SkillInputError(
                f"type mismatch for {field_name!r}: got {type(value).__name__}, expected {declared_type}"
            )


def _type_matches(value: Any, declared: str) -> bool:
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = mapping.get(declared)
    return expected is None or isinstance(value, expected)


def _validate_output(skill: ResolvedSkill, output_text: str) -> None:
    """Stub for OQ4 output validation when output.schema.yaml exists.

    v1 only verifies that something was produced; structural validation
    is added when an actual caller demands it (lazy schema philosophy).
    """
    if not output_text or not output_text.strip():
        raise SkillOutputDriftError(
            f"agent {skill.name!r} produced empty output but declared output.schema.yaml"
        )
