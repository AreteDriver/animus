"""Tests for the contracts module."""

import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.contracts import (
    AgentContract,
    AgentRole,
    ContractViolation,
    get_contract,
)
from animus_forge.contracts.validator import ContractValidator, ValidationResult


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_roles_exist(self):
        """All expected roles are defined."""
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.BUILDER.value == "builder"
        assert AgentRole.TESTER.value == "tester"
        assert AgentRole.REVIEWER.value == "reviewer"
        assert AgentRole.MODEL_BUILDER.value == "model_builder"

    def test_role_from_string(self):
        """Can create role from string value."""
        assert AgentRole("planner") == AgentRole.PLANNER
        assert AgentRole("model_builder") == AgentRole.MODEL_BUILDER


class TestAgentContract:
    """Tests for AgentContract class."""

    def test_create_contract(self):
        """Can create a contract with schema."""
        contract = AgentContract(
            role=AgentRole.PLANNER,
            input_schema={
                "type": "object",
                "required": ["task"],
                "properties": {"task": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "required": ["plan"],
                "properties": {"plan": {"type": "string"}},
            },
            description="Test contract",
        )
        assert contract.role == AgentRole.PLANNER
        assert "task" in contract.input_schema["required"]

    def test_validate_input_success(self):
        """Valid input passes validation."""
        contract = get_contract(AgentRole.PLANNER)
        # PLANNER requires 'request' and 'context' fields
        valid_input = {"request": "Build a REST API", "context": {}}
        assert contract.validate_input(valid_input) is True

    def test_validate_input_failure(self):
        """Invalid input raises ContractViolation."""
        contract = get_contract(AgentRole.PLANNER)
        invalid_input = {}  # Missing required fields
        with pytest.raises(ContractViolation):
            contract.validate_input(invalid_input)

    def test_validate_output_success(self):
        """Valid output passes validation."""
        contract = get_contract(AgentRole.PLANNER)
        valid_output = {
            "tasks": [{"id": "1", "description": "Step 1", "dependencies": []}],
            "architecture": "Simple architecture",
            "success_criteria": ["Tests pass"],
        }
        assert contract.validate_output(valid_output) is True

    def test_validate_output_failure(self):
        """Invalid output raises ContractViolation."""
        contract = get_contract(AgentRole.PLANNER)
        invalid_output = {"wrong": "data"}
        with pytest.raises(ContractViolation):
            contract.validate_output(invalid_output)

    def test_check_context(self):
        """Context checking identifies missing fields."""
        contract = AgentContract(
            role=AgentRole.BUILDER,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_context=["plan", "requirements"],
        )
        missing = contract.check_context({"plan": "..."})
        assert "requirements" in missing
        assert "plan" not in missing


class TestGetContract:
    """Tests for get_contract function."""

    def test_get_by_enum(self):
        """Can get contract by AgentRole enum."""
        contract = get_contract(AgentRole.BUILDER)
        assert contract.role == AgentRole.BUILDER

    def test_get_by_string(self):
        """Can get contract by string name."""
        contract = get_contract("tester")
        assert contract.role == AgentRole.TESTER

    def test_unknown_role_raises(self):
        """Unknown role raises ValueError."""
        with pytest.raises(ValueError):
            get_contract("unknown_role")

    def test_core_roles_have_contracts(self):
        """Core AgentRole values have contracts defined."""
        core_roles = [
            AgentRole.PLANNER,
            AgentRole.BUILDER,
            AgentRole.TESTER,
            AgentRole.REVIEWER,
            AgentRole.MODEL_BUILDER,
        ]
        for role in core_roles:
            contract = get_contract(role)
            assert contract is not None


class TestContractValidator:
    """Tests for ContractValidator class."""

    def test_validate_input_strict(self):
        """Strict mode raises on validation failure."""
        validator = ContractValidator(strict=True)
        with pytest.raises(ContractViolation):
            validator.validate_input("planner", {})

    def test_validate_input_non_strict(self):
        """Non-strict mode returns ValidationResult."""
        validator = ContractValidator(strict=False)
        result = validator.validate_input("planner", {})
        assert isinstance(result, ValidationResult)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_output(self):
        """Output validation works."""
        validator = ContractValidator(strict=False)
        result = validator.validate_output(
            "builder",
            {
                "code": "print('hello')",
                "files_created": ["main.py"],
                "status": "complete",
            },
        )
        assert result.valid is True

    def test_validate_handoff(self):
        """Handoff validation checks both sides."""
        validator = ContractValidator(strict=False)
        data = {
            "tasks": [{"id": "1", "description": "Task 1", "dependencies": []}],
            "architecture": "Simple architecture",
            "success_criteria": ["Tests pass"],
        }
        output_result, input_result = validator.validate_handoff("planner", "builder", data)
        assert output_result.valid is True
        # Builder input might have different requirements
        assert isinstance(input_result, ValidationResult)

    def test_validation_history(self):
        """Validator tracks validation history."""
        validator = ContractValidator(strict=False)
        validator.validate_input("planner", {"request": "test", "context": {}})
        validator.validate_output(
            "planner",
            {
                "tasks": [{"id": "1", "description": "Task", "dependencies": []}],
                "architecture": "Simple",
                "success_criteria": ["Pass"],
            },
        )
        history = validator.get_history()
        assert len(history) == 2

    def test_validation_stats(self):
        """Validator provides statistics."""
        validator = ContractValidator(strict=False)
        validator.validate_input("planner", {"request": "test", "context": {}})  # Valid
        validator.validate_input("planner", {})  # Invalid
        stats = validator.get_stats()
        assert stats["total"] == 2
        assert stats["valid"] == 1
        assert stats["invalid"] == 1


class TestContractViolation:
    """Tests for ContractViolation exception."""

    def test_violation_message(self):
        """Violation includes message."""
        violation = ContractViolation("Test error", role="planner")
        assert "Test error" in str(violation)

    def test_violation_role(self):
        """Violation includes role."""
        violation = ContractViolation("Error", role="builder")
        assert violation.role == "builder"


class TestModelBuilderContract:
    """Tests for the MODEL_BUILDER agent contract."""

    def test_model_builder_contract_exists(self):
        """MODEL_BUILDER contract is registered."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        assert contract is not None
        assert contract.role == AgentRole.MODEL_BUILDER

    def test_model_builder_by_string(self):
        """Can get MODEL_BUILDER contract by string."""
        contract = get_contract("model_builder")
        assert contract.role == AgentRole.MODEL_BUILDER

    def test_model_builder_input_validation_success(self):
        """Valid MODEL_BUILDER input passes validation."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        valid_input = {
            "request": "Create a procedural terrain generator",
            "target_platform": "unity",
            "asset_type": "script",
        }
        assert contract.validate_input(valid_input) is True

    def test_model_builder_input_validation_minimal(self):
        """MODEL_BUILDER accepts minimal required fields."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        minimal_input = {
            "request": "Create a shader",
            "target_platform": "blender",
        }
        assert contract.validate_input(minimal_input) is True

    def test_model_builder_input_validation_all_platforms(self):
        """MODEL_BUILDER accepts all valid platforms."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        platforms = ["unity", "blender", "unreal", "godot", "threejs", "generic"]
        for platform in platforms:
            valid_input = {
                "request": "Create an asset",
                "target_platform": platform,
            }
            assert contract.validate_input(valid_input) is True

    def test_model_builder_input_validation_failure_missing_fields(self):
        """MODEL_BUILDER rejects input missing required fields."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        invalid_input = {"request": "Create something"}  # Missing target_platform
        with pytest.raises(ContractViolation):
            contract.validate_input(invalid_input)

    def test_model_builder_input_validation_failure_invalid_platform(self):
        """MODEL_BUILDER rejects invalid platform."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        invalid_input = {
            "request": "Create an asset",
            "target_platform": "invalid_platform",
        }
        with pytest.raises(ContractViolation):
            contract.validate_input(invalid_input)

    def test_model_builder_output_validation_success(self):
        """Valid MODEL_BUILDER output passes validation."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        valid_output = {
            "assets": [
                {
                    "name": "TerrainGenerator.cs",
                    "type": "script",
                    "content": "using UnityEngine; public class TerrainGenerator {}",
                }
            ],
            "instructions": [{"step": 1, "action": "Import the script into Unity"}],
            "status": "complete",
        }
        assert contract.validate_output(valid_output) is True

    def test_model_builder_output_validation_partial_status(self):
        """MODEL_BUILDER accepts partial status."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        valid_output = {
            "assets": [],
            "instructions": [{"step": 1, "action": "Open Blender and create a new mesh"}],
            "status": "needs_manual_work",
        }
        assert contract.validate_output(valid_output) is True

    def test_model_builder_output_validation_failure(self):
        """MODEL_BUILDER rejects invalid output."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        invalid_output = {"wrong": "data"}
        with pytest.raises(ContractViolation):
            contract.validate_output(invalid_output)

    def test_model_builder_description(self):
        """MODEL_BUILDER has a meaningful description."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        assert "3D" in contract.description
        assert len(contract.description) > 20

    def test_model_builder_required_context(self):
        """MODEL_BUILDER has appropriate required context."""
        contract = get_contract(AgentRole.MODEL_BUILDER)
        assert "target_platform" in contract.required_context
