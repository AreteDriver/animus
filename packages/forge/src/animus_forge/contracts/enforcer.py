"""Automatic contract enforcement with retry-on-violation.

When an agent's output fails contract validation, the enforcer re-invokes
the agent with corrective feedback, creating a quality guarantee layer.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from animus_forge.contracts.base import AgentRole, ContractViolation
from animus_forge.contracts.definitions import get_contract

logger = logging.getLogger(__name__)


@dataclass
class EnforcementResult:
    """Result of a validate-and-retry enforcement cycle.

    Attributes:
        role: The agent role that was enforced.
        valid: Whether the final output satisfies the contract.
        attempts: Total number of attempts (1 = passed first try).
        violations: Description of each violation encountered across attempts.
        corrected: Whether a violation was fixed via retry.
    """

    role: str
    valid: bool
    attempts: int
    violations: list[str]
    corrected: bool


@dataclass
class EnforcementStats:
    """Aggregate enforcement statistics.

    Attributes:
        total_validations: Total number of validation calls.
        total_violations: Total number of initial validation failures.
        total_corrections: Violations that were fixed via retry.
        violation_rate: Fraction of validations that resulted in a violation.
        correction_rate: Of violations, fraction that were corrected.
        by_role: Per-role breakdown with keys: validations, violations, corrections.
    """

    total_validations: int
    total_violations: int
    total_corrections: int
    violation_rate: float
    correction_rate: float
    by_role: dict[str, dict]


@dataclass
class _RoleCounters:
    """Internal per-role tracking counters."""

    validations: int = 0
    violations: int = 0
    corrections: int = 0


class ContractEnforcer:
    """Enforces agent contracts with automatic retry on violation.

    Acts as a drop-in replacement for the contract_validator interface
    expected by WorkflowExecutor, and additionally provides retry logic
    that re-invokes agents with corrective feedback when outputs fail
    validation.

    Args:
        max_retries: Default maximum retry attempts for validate_and_retry.
    """

    def __init__(self, max_retries: int = 2) -> None:
        self._default_max_retries = max_retries
        self._counters: dict[str, _RoleCounters] = defaultdict(_RoleCounters)

    def validate_output(self, role: AgentRole | str, output: dict[str, Any]) -> bool:
        """Validate agent output against its registered contract.

        This method satisfies the contract_validator interface expected by
        WorkflowExecutor.

        Args:
            role: The agent role whose contract to validate against.
            output: The agent's output data.

        Returns:
            True if the output is valid.

        Raises:
            ContractViolation: If the output violates the contract.
        """
        role_str = role.value if isinstance(role, AgentRole) else role
        counters = self._counters[role_str]
        counters.validations += 1

        contract = get_contract(role)
        try:
            contract.validate_output(output)
            return True
        except ContractViolation:
            counters.violations += 1
            raise

    async def validate_and_retry(
        self,
        role: AgentRole | str,
        output: dict[str, Any],
        step_config: dict[str, Any],
        executor_callback: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        max_retries: int | None = None,
    ) -> tuple[dict[str, Any], EnforcementResult]:
        """Validate output and retry the agent on contract violation.

        If the output fails validation, constructs a correction prompt and
        re-invokes the agent via executor_callback. Repeats up to max_retries
        times. Returns the best output obtained along with enforcement metadata.

        Args:
            role: The agent role to enforce.
            output: The initial output to validate.
            step_config: The workflow step configuration, must contain at least
                an ``original_prompt`` key used to build correction prompts.
            executor_callback: An async callable that re-invokes the agent.
                Signature: (prompt: str, step_config: dict) -> output dict.
            max_retries: Maximum retry attempts. Defaults to the instance default.

        Returns:
            A tuple of (final_output, EnforcementResult).
        """
        if max_retries is None:
            max_retries = self._default_max_retries

        role_enum = AgentRole(role) if isinstance(role, str) else role
        role_str = role_enum.value
        contract = get_contract(role_enum)
        counters = self._counters[role_str]

        violations: list[str] = []
        best_output = output
        attempt = 0

        for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
            try:
                counters.validations += 1
                contract.validate_output(best_output)
                logger.info(
                    "Contract satisfied for role '%s' on attempt %d",
                    role_str,
                    attempt,
                )
                corrected = attempt > 1 and len(violations) > 0
                if corrected:
                    counters.corrections += 1
                return best_output, EnforcementResult(
                    role=role_str,
                    valid=True,
                    attempts=attempt,
                    violations=violations,
                    corrected=corrected,
                )
            except ContractViolation as exc:
                counters.violations += 1
                violation_msg = str(exc)
                violations.append(violation_msg)
                logger.warning(
                    "Contract violation for role '%s' (attempt %d/%d): %s",
                    role_str,
                    attempt,
                    max_retries + 1,
                    violation_msg,
                )

                if attempt > max_retries:
                    # Exhausted retries, return best output as-is
                    break

                original_prompt = step_config.get("original_prompt", "")
                correction_prompt = self.build_correction_prompt(original_prompt, exc, attempt)

                try:
                    best_output = await executor_callback(correction_prompt, step_config)
                except Exception:
                    logger.exception(
                        "Executor callback failed during retry %d for role '%s'",
                        attempt,
                        role_str,
                    )
                    break

        return best_output, EnforcementResult(
            role=role_str,
            valid=False,
            attempts=attempt,
            violations=violations,
            corrected=False,
        )

    def build_correction_prompt(
        self,
        original_prompt: str,
        violation: ContractViolation,
        attempt: int,
    ) -> str:
        """Build a follow-up prompt explaining what the agent got wrong.

        Args:
            original_prompt: The prompt that produced the invalid output.
            violation: The ContractViolation that was raised.
            attempt: The current attempt number (1-indexed).

        Returns:
            A correction prompt instructing the agent to fix its output.
        """
        role_str = violation.role or "unknown"
        try:
            contract = get_contract(role_str)
            schema_str = json.dumps(contract.output_schema, indent=2)
        except ValueError:
            schema_str = "(schema unavailable)"

        required_fields: list[str] = []
        try:
            contract = get_contract(role_str)
            required_fields = contract.output_schema.get("required", [])
        except ValueError:
            pass  # Graceful degradation: unknown role has no required fields, use empty list

        parts = [
            f"[Retry attempt {attempt + 1}] Your previous output did not satisfy "
            f"the contract for the '{role_str}' role.",
            f"Specifically: {violation}",
        ]

        if violation.field:
            parts.append(f"Problem field: {violation.field}")

        if violation.details:
            parts.append(f"Details: {json.dumps(violation.details)}")

        parts.append(f"Required output schema:\n{schema_str}")

        if required_fields:
            parts.append(
                "Please ensure your output includes these required fields: "
                + ", ".join(required_fields)
            )

        parts.append("Please try again with a corrected output that satisfies the schema.")

        if original_prompt:
            parts.append(f"Original request:\n{original_prompt}")

        return "\n\n".join(parts)

    def get_enforcement_stats(self) -> EnforcementStats:
        """Compute aggregate enforcement statistics.

        Returns:
            An EnforcementStats snapshot of all tracked validations.
        """
        total_validations = 0
        total_violations = 0
        total_corrections = 0
        by_role: dict[str, dict] = {}

        for role_str, counters in self._counters.items():
            total_validations += counters.validations
            total_violations += counters.violations
            total_corrections += counters.corrections
            by_role[role_str] = {
                "validations": counters.validations,
                "violations": counters.violations,
                "corrections": counters.corrections,
            }

        violation_rate = total_violations / total_validations if total_validations > 0 else 0.0
        correction_rate = total_corrections / total_violations if total_violations > 0 else 0.0

        return EnforcementStats(
            total_validations=total_validations,
            total_violations=total_violations,
            total_corrections=total_corrections,
            violation_rate=violation_rate,
            correction_rate=correction_rate,
            by_role=by_role,
        )
