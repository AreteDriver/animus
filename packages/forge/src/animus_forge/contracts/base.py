"""Base classes for agent contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate


class AgentRole(Enum):
    """Defined agent roles in the orchestration system."""

    PLANNER = "planner"
    BUILDER = "builder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    ANALYST = "analyst"
    VISUALIZER = "visualizer"
    REPORTER = "reporter"
    DATA_ANALYST = "data_analyst"
    DEVOPS = "devops"
    SECURITY_AUDITOR = "security_auditor"
    MIGRATOR = "migrator"
    MODEL_BUILDER = "model_builder"


class ContractViolation(Exception):
    """Raised when agent input/output violates its contract."""

    def __init__(self, message: str, role: str = None, field: str = None, details: dict = None):
        self.role = role
        self.field = field
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error": "contract_violation",
            "message": str(self),
            "role": self.role,
            "field": self.field,
            "details": self.details,
        }


@dataclass
class AgentContract:
    """Defines what an agent expects as input and guarantees as output.

    Each agent has a contract specifying:
    - input_schema: JSON Schema for valid inputs
    - output_schema: JSON Schema for valid outputs
    - required_context: Optional context fields the agent needs
    """

    role: AgentRole
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    description: str = ""
    required_context: list[str] = field(default_factory=list)

    def validate_input(self, data: dict) -> bool:
        """Validate input data against the contract's input schema.

        Args:
            data: Input data to validate

        Returns:
            True if valid

        Raises:
            ContractViolation: If validation fails
        """
        try:
            validate(instance=data, schema=self.input_schema)
            return True
        except JsonSchemaError as e:
            raise ContractViolation(
                f"{self.role.value} input invalid: {e.message}",
                role=self.role.value,
                field=e.json_path if hasattr(e, "json_path") else str(e.path),
                details={
                    "schema_path": list(e.schema_path) if e.schema_path else [],
                    "validator": e.validator,
                    "validator_value": str(e.validator_value)[:100],
                },
            )

    def validate_output(self, data: dict) -> bool:
        """Validate output data against the contract's output schema.

        Args:
            data: Output data to validate

        Returns:
            True if valid

        Raises:
            ContractViolation: If validation fails
        """
        try:
            validate(instance=data, schema=self.output_schema)
            return True
        except JsonSchemaError as e:
            raise ContractViolation(
                f"{self.role.value} output invalid: {e.message}",
                role=self.role.value,
                field=e.json_path if hasattr(e, "json_path") else str(e.path),
                details={
                    "schema_path": list(e.schema_path) if e.schema_path else [],
                    "validator": e.validator,
                },
            )

    def check_context(self, context: dict) -> list[str]:
        """Check if required context fields are present.

        Args:
            context: Context dictionary to check

        Returns:
            List of missing field names (empty if all present)
        """
        missing = []
        for field_name in self.required_context:
            if field_name not in context:
                missing.append(field_name)
        return missing

    def to_dict(self) -> dict:
        """Convert contract to dictionary for serialization."""
        return {
            "role": self.role.value,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "required_context": self.required_context,
        }
