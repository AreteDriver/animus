"""Pre-flight budget validation for workflows.

Validates that workflows have sufficient budget before execution starts.
Provides cost estimation and detailed validation results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from animus_forge.budget.manager import BudgetConfig, BudgetManager

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Pre-flight validation status."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class StepEstimate:
    """Estimated token usage for a workflow step."""

    step_name: str
    agent_type: str
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    confidence: float = 0.8  # 0.0 to 1.0, how confident the estimate is

    @property
    def total_tokens(self) -> int:
        """Total estimated tokens for this step."""
        return self.estimated_input_tokens + self.estimated_output_tokens


@dataclass
class WorkflowEstimate:
    """Estimated total budget for a workflow."""

    workflow_id: str
    steps: list[StepEstimate] = field(default_factory=list)
    overhead_tokens: int = 1000  # Parsing, validation, etc.
    retry_buffer_percent: float = 0.2  # 20% buffer for retries

    @property
    def base_tokens(self) -> int:
        """Base token estimate (steps + overhead)."""
        return sum(s.total_tokens for s in self.steps) + self.overhead_tokens

    @property
    def total_with_buffer(self) -> int:
        """Total tokens including retry buffer."""
        return int(self.base_tokens * (1 + self.retry_buffer_percent))

    @property
    def min_tokens(self) -> int:
        """Minimum tokens (best case)."""
        return self.base_tokens

    @property
    def max_tokens(self) -> int:
        """Maximum tokens (worst case with retries)."""
        return int(self.base_tokens * (1 + self.retry_buffer_percent * 2))

    @property
    def average_confidence(self) -> float:
        """Average confidence of step estimates."""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)


@dataclass
class ValidationResult:
    """Result of pre-flight budget validation."""

    status: ValidationStatus
    workflow_id: str
    estimate: WorkflowEstimate
    current_budget: int
    current_used: int
    messages: list[str] = field(default_factory=list)
    step_validations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available_budget(self) -> int:
        """Currently available budget."""
        return max(0, self.current_budget - self.current_used)

    @property
    def budget_sufficient(self) -> bool:
        """Check if budget is sufficient for estimated usage."""
        return self.available_budget >= self.estimate.total_with_buffer

    @property
    def margin(self) -> int:
        """Budget margin (positive = surplus, negative = deficit)."""
        return self.available_budget - self.estimate.total_with_buffer

    @property
    def margin_percent(self) -> float:
        """Budget margin as percentage of estimate."""
        if self.estimate.total_with_buffer == 0:
            return 100.0
        return (self.margin / self.estimate.total_with_buffer) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "workflow_id": self.workflow_id,
            "budget_sufficient": self.budget_sufficient,
            "estimate": {
                "base_tokens": self.estimate.base_tokens,
                "total_with_buffer": self.estimate.total_with_buffer,
                "min_tokens": self.estimate.min_tokens,
                "max_tokens": self.estimate.max_tokens,
                "confidence": round(self.estimate.average_confidence, 2),
                "steps": len(self.estimate.steps),
            },
            "budget": {
                "total": self.current_budget,
                "used": self.current_used,
                "available": self.available_budget,
                "margin": self.margin,
                "margin_percent": round(self.margin_percent, 1),
            },
            "messages": self.messages,
            "step_validations": self.step_validations,
        }


# Default token estimates by agent type
DEFAULT_ESTIMATES = {
    "planner": {"input": 2000, "output": 1500},
    "builder": {"input": 3000, "output": 4000},
    "tester": {"input": 2500, "output": 2000},
    "reviewer": {"input": 3000, "output": 1500},
    "architect": {"input": 3500, "output": 3000},
    "documenter": {"input": 2000, "output": 3000},
    "analyst": {"input": 4000, "output": 2500},
    "visualizer": {"input": 2000, "output": 1000},
    "reporter": {"input": 3000, "output": 2000},
    "default": {"input": 2500, "output": 2000},
}


class PreflightValidator:
    """Validates workflow budget requirements before execution."""

    def __init__(
        self,
        budget_manager: BudgetManager | None = None,
        custom_estimates: dict[str, dict[str, int]] | None = None,
    ):
        """Initialize validator.

        Args:
            budget_manager: Budget manager to validate against
            custom_estimates: Custom token estimates by agent type
        """
        self.budget_manager = budget_manager or BudgetManager()
        self.estimates = {**DEFAULT_ESTIMATES, **(custom_estimates or {})}

    def estimate_step(
        self,
        step_name: str,
        agent_type: str,
        prompt_length: int = 0,
        context_length: int = 0,
    ) -> StepEstimate:
        """Estimate tokens for a single step.

        Args:
            step_name: Name of the step
            agent_type: Type of agent executing the step
            prompt_length: Length of prompt in characters (optional)
            context_length: Length of context in characters (optional)

        Returns:
            Step estimate
        """
        agent_type_lower = agent_type.lower()
        defaults = self.estimates.get(agent_type_lower, self.estimates["default"])

        # Adjust based on prompt/context if provided
        input_tokens = defaults["input"]
        if prompt_length:
            # Rough estimate: ~4 chars per token
            input_tokens = max(input_tokens, prompt_length // 4)
        if context_length:
            input_tokens += context_length // 4

        output_tokens = defaults["output"]

        # Confidence decreases with customization
        confidence = 0.8
        if prompt_length or context_length:
            confidence = 0.6

        return StepEstimate(
            step_name=step_name,
            agent_type=agent_type,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            confidence=confidence,
        )

    def estimate_workflow(
        self,
        workflow_id: str,
        steps: list[dict[str, Any]],
        overhead_tokens: int = 1000,
        retry_buffer_percent: float = 0.2,
    ) -> WorkflowEstimate:
        """Estimate total tokens for a workflow.

        Args:
            workflow_id: Workflow identifier
            steps: List of step configurations
            overhead_tokens: Additional tokens for overhead
            retry_buffer_percent: Buffer for potential retries

        Returns:
            Workflow estimate
        """
        step_estimates = []

        for step in steps:
            name = step.get("name", step.get("step", "unknown"))
            agent_type = step.get("agent", step.get("agent_type", "default"))
            prompt_length = len(str(step.get("prompt", "")))
            context_length = len(str(step.get("context", "")))

            estimate = self.estimate_step(
                step_name=name,
                agent_type=agent_type,
                prompt_length=prompt_length,
                context_length=context_length,
            )
            step_estimates.append(estimate)

        return WorkflowEstimate(
            workflow_id=workflow_id,
            steps=step_estimates,
            overhead_tokens=overhead_tokens,
            retry_buffer_percent=retry_buffer_percent,
        )

    def validate(
        self,
        workflow_id: str,
        steps: list[dict[str, Any]],
        strict: bool = False,
    ) -> ValidationResult:
        """Validate workflow budget requirements.

        Args:
            workflow_id: Workflow identifier
            steps: List of step configurations
            strict: If True, fail on warnings (margin < 25%)

        Returns:
            Validation result
        """
        estimate = self.estimate_workflow(workflow_id, steps)
        stats = self.budget_manager.get_stats()

        messages = []
        step_validations = []
        status = ValidationStatus.PASS

        # Validate overall budget
        available = stats["available"]
        required = estimate.total_with_buffer

        if required > available:
            status = ValidationStatus.FAIL
            messages.append(
                f"Insufficient budget: need ~{required:,} tokens, only {available:,} available"
            )
        elif required > available * 0.75:  # Less than 25% margin
            status = ValidationStatus.WARN if not strict else ValidationStatus.FAIL
            messages.append(
                f"Tight budget margin: need ~{required:,} tokens, {available:,} available "
                f"({round((available - required) / required * 100, 1)}% margin)"
            )
        else:
            messages.append(
                f"Budget sufficient: need ~{required:,} tokens, {available:,} available"
            )

        # Validate per-step limits if configured
        per_step_limit = self.budget_manager.config.per_step_limit
        for step_est in estimate.steps:
            validation = {
                "step": step_est.step_name,
                "agent": step_est.agent_type,
                "estimated_tokens": step_est.total_tokens,
                "status": "ok",
            }

            if per_step_limit and step_est.total_tokens > per_step_limit:
                validation["status"] = "exceeds_limit"
                validation["limit"] = per_step_limit
                if status != ValidationStatus.FAIL:
                    status = ValidationStatus.WARN
                messages.append(
                    f"Step '{step_est.step_name}' estimate ({step_est.total_tokens:,}) "
                    f"exceeds per-step limit ({per_step_limit:,})"
                )

            step_validations.append(validation)

        # Check confidence
        if estimate.average_confidence < 0.5:
            if status == ValidationStatus.PASS:
                status = ValidationStatus.WARN
            messages.append(
                f"Low estimation confidence ({round(estimate.average_confidence * 100)}%). "
                "Actual usage may vary significantly."
            )

        logger.info(
            f"Pre-flight validation for '{workflow_id}': {status.value} "
            f"(estimated: {estimate.total_with_buffer:,}, available: {available:,})"
        )

        return ValidationResult(
            status=status,
            workflow_id=workflow_id,
            estimate=estimate,
            current_budget=stats["total_budget"],
            current_used=stats["used"],
            messages=messages,
            step_validations=step_validations,
        )

    def quick_check(
        self,
        workflow_id: str,
        steps: list[dict[str, Any]],
    ) -> bool:
        """Quick budget sufficiency check.

        Args:
            workflow_id: Workflow identifier
            steps: List of step configurations

        Returns:
            True if budget is likely sufficient
        """
        result = self.validate(workflow_id, steps)
        return result.status != ValidationStatus.FAIL


def validate_workflow_budget(
    workflow_id: str,
    steps: list[dict[str, Any]],
    budget_config: BudgetConfig | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Convenience function for one-shot validation.

    Args:
        workflow_id: Workflow identifier
        steps: List of step configurations
        budget_config: Optional budget configuration
        strict: If True, fail on warnings

    Returns:
        Validation result
    """
    manager = BudgetManager(config=budget_config)
    validator = PreflightValidator(budget_manager=manager)
    return validator.validate(workflow_id, steps, strict=strict)
