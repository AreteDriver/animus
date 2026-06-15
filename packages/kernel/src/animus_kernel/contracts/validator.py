"""Contract validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .base import AgentRole, ContractViolation
from .definitions import get_contract


@dataclass
class ValidationResult:
    """Result of a contract validation."""

    valid: bool
    role: str
    direction: str  # "input" or "output"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "role": self.role,
            "direction": self.direction,
            "errors": self.errors,
            "warnings": self.warnings,
            "validated_at": self.validated_at.isoformat(),
        }


class ContractValidator:
    """Validates data against agent contracts with detailed reporting."""

    def __init__(self, strict: bool = True):
        """Initialize validator.

        Args:
            strict: If True, raise exceptions on validation failure.
                   If False, return ValidationResult with errors.
        """
        self.strict = strict
        self._validation_history: list[ValidationResult] = []

    def validate_input(
        self,
        role: AgentRole | str,
        data: dict,
        context: dict = None,
    ) -> ValidationResult:
        """Validate input data for an agent.

        Args:
            role: The agent role
            data: Input data to validate
            context: Optional context to check for required fields

        Returns:
            ValidationResult with validation status

        Raises:
            ContractViolation: If strict=True and validation fails
        """
        contract = get_contract(role)
        role_name = role.value if isinstance(role, AgentRole) else role

        errors = []
        warnings = []

        # Validate against schema
        try:
            contract.validate_input(data)
        except ContractViolation as e:
            errors.append(str(e))

        # Check required context
        if context is not None:
            missing = contract.check_context(context)
            if missing:
                warnings.append(f"Missing context fields: {', '.join(missing)}")

        result = ValidationResult(
            valid=len(errors) == 0,
            role=role_name,
            direction="input",
            errors=errors,
            warnings=warnings,
        )

        self._validation_history.append(result)

        if self.strict and not result.valid:
            raise ContractViolation(
                f"Input validation failed for {role_name}: {'; '.join(errors)}",
                role=role_name,
            )

        return result

    def validate_output(
        self,
        role: AgentRole | str,
        data: dict,
    ) -> ValidationResult:
        """Validate output data from an agent.

        Args:
            role: The agent role
            data: Output data to validate

        Returns:
            ValidationResult with validation status

        Raises:
            ContractViolation: If strict=True and validation fails
        """
        contract = get_contract(role)
        role_name = role.value if isinstance(role, AgentRole) else role

        errors = []
        warnings = []

        # Validate against schema
        try:
            contract.validate_output(data)
        except ContractViolation as e:
            errors.append(str(e))

        # Additional output quality checks
        warnings.extend(self._check_output_quality(role_name, data))

        result = ValidationResult(
            valid=len(errors) == 0,
            role=role_name,
            direction="output",
            errors=errors,
            warnings=warnings,
        )

        self._validation_history.append(result)

        if self.strict and not result.valid:
            raise ContractViolation(
                f"Output validation failed for {role_name}: {'; '.join(errors)}",
                role=role_name,
            )

        return result

    def validate_handoff(
        self,
        from_role: AgentRole | str,
        to_role: AgentRole | str,
        data: dict,
    ) -> tuple[ValidationResult, ValidationResult]:
        """Validate a handoff between two agents.

        Ensures the output from one agent is valid input for the next.

        Args:
            from_role: The sending agent
            to_role: The receiving agent
            data: Data being passed

        Returns:
            Tuple of (output_validation, input_validation) results
        """
        # Validate as output from sender
        output_result = self.validate_output(from_role, data)

        # Validate as input to receiver
        input_result = self.validate_input(to_role, data)

        return output_result, input_result

    def _check_planner_quality(self, data: dict) -> list[str]:
        """Check planner output quality."""
        warnings = []
        tasks = data.get("tasks", [])
        if len(tasks) > 10:
            warnings.append(f"Plan has {len(tasks)} tasks - consider breaking into smaller chunks")
        if not data.get("risks"):
            warnings.append("Plan has no identified risks")
        return warnings

    def _check_builder_quality(self, data: dict) -> list[str]:
        """Check builder output quality."""
        warnings = []
        if len(data.get("code", "")) > 10000:
            warnings.append("Large code output - consider smaller increments")
        if data.get("status") == "partial" and not data.get("notes"):
            warnings.append("Partial completion without notes explaining what's missing")
        return warnings

    def _check_tester_quality(self, data: dict) -> list[str]:
        """Check tester output quality."""
        warnings = []
        if data.get("tests_run", 0) == 0:
            warnings.append("No tests were run")
        coverage = data.get("coverage")
        if coverage is not None and coverage < 50:
            warnings.append(f"Low test coverage: {coverage}%")
        return warnings

    def _check_reviewer_quality(self, data: dict) -> list[str]:
        """Check reviewer output quality."""
        warnings = []
        if data.get("approved") and data.get("findings"):
            critical = sum(1 for f in data["findings"] if f.get("severity") == "critical")
            if critical > 0:
                warnings.append(f"Approved with {critical} critical findings")
        return warnings

    def _check_output_quality(self, role: str, data: dict) -> list[str]:
        """Check output data quality beyond schema validation."""
        checkers = {
            "planner": self._check_planner_quality,
            "builder": self._check_builder_quality,
            "tester": self._check_tester_quality,
            "reviewer": self._check_reviewer_quality,
        }
        checker = checkers.get(role)
        return checker(data) if checker else []

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent validation history.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of ValidationResult dictionaries
        """
        return [r.to_dict() for r in self._validation_history[-limit:]]

    def get_stats(self) -> dict:
        """Get validation statistics.

        Returns:
            Dictionary with validation stats
        """
        total = len(self._validation_history)
        if total == 0:
            return {"total": 0, "success_rate": 0}

        valid_count = sum(1 for r in self._validation_history if r.valid)

        by_role: dict[str, dict] = {}
        for r in self._validation_history:
            if r.role not in by_role:
                by_role[r.role] = {"total": 0, "valid": 0}
            by_role[r.role]["total"] += 1
            if r.valid:
                by_role[r.role]["valid"] += 1

        return {
            "total": total,
            "valid": valid_count,
            "invalid": total - valid_count,
            "success_rate": round((valid_count / total) * 100, 1),
            "by_role": by_role,
        }


def validate_workflow_contracts(workflow_steps: list[dict]) -> list[str]:
    """Validate that workflow steps have compatible contracts.

    Checks that each step's output can feed into the next step's input.

    Args:
        workflow_steps: List of workflow step definitions

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for i, step in enumerate(workflow_steps):
        step_type = step.get("type")
        if step_type in ("claude_code", "openai"):
            role = step.get("params", {}).get("role")
            if role:
                try:
                    get_contract(role)
                except ValueError as e:
                    errors.append(f"Step {i + 1} ({step.get('id')}): {e}")

    return errors
