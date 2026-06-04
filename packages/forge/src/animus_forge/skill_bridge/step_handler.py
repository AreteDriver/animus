"""Wires SkillBridge into Forge's StepHandler dispatch.

Workflow YAML usage::

    steps:
      - id: review
        type: skill
        skill: code-reviewer
        tier: persona
        inputs:
          files: ["src/foo.py"]
        provider_hint: anthropic    # optional
"""

from __future__ import annotations

import logging
from typing import Any

from .errors import SkillConfigError, SkillResolutionError
from .executor import SkillCallResult, SkillExecutor, StepContext
from .resolver import TIERS, SkillResolver

logger = logging.getLogger(__name__)

STEP_TYPE = "skill"


def build_skill_step_handler(
    *,
    resolver: SkillResolver,
    executor: SkillExecutor,
    execution_id_provider=lambda: "",
    workflow_id_provider=lambda: "",
):
    """Return a handler `(step, context) -> dict` registered as `type: skill`."""

    def _handler(step: Any, context: dict) -> dict:
        params = getattr(step, "params", None) or {}
        name = params.get("skill")
        tier = params.get("tier")
        if not name:
            raise SkillConfigError("step.params missing required 'skill'")
        if tier not in TIERS:
            raise SkillConfigError(f"step.params.tier must be one of {TIERS!r}, got {tier!r}")

        inputs = params.get("inputs", {}) or {}
        provider_hint = params.get("provider_hint")

        try:
            resolved = resolver.resolve(name, tier)
        except SkillResolutionError as exc:
            logger.warning("skill resolution failed: %s", exc)
            return {
                "success": False,
                "error": f"skill.resolution: {exc}",
                "error_category": "skill.resolution",
            }

        # Tier mismatch — registry says it's elsewhere
        if resolved.tier != tier:
            return {
                "success": False,
                "error": f"skill.tier_mismatch: registry has {resolved.tier!r}, YAML says {tier!r}",
                "error_category": "skill.tier_mismatch",
            }

        ctx = StepContext(
            execution_id=execution_id_provider() or context.get("execution_id", ""),
            workflow_id=workflow_id_provider() or context.get("workflow_id", ""),
            step_id=getattr(step, "id", "") or context.get("step_id", ""),
            next_step_id=context.get("next_step_id", ""),
            nesting_depth=int(context.get("_skill_nesting_depth", 0)),
            provider_hint=provider_hint,
        )

        result: SkillCallResult = executor.execute(resolved, inputs, ctx)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "tokens_used": result.tokens_used,
        }

    return _handler
