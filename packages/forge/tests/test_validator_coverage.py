"""Additional coverage tests for ContractValidator and validate_workflow_contracts."""

import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.contracts.base import AgentRole, ContractViolation
from animus_forge.contracts.validator import (
    ContractValidator,
    ValidationResult,
    validate_workflow_contracts,
)


class TestValidationResult:
    def test_to_dict(self):
        result = ValidationResult(valid=True, role="planner", direction="input")
        d = result.to_dict()
        assert d["valid"] is True
        assert d["role"] == "planner"
        assert d["direction"] == "input"
        assert "validated_at" in d

    def test_with_errors(self):
        result = ValidationResult(
            valid=False,
            role="builder",
            direction="output",
            errors=["Missing code"],
            warnings=["Large output"],
        )
        d = result.to_dict()
        assert d["valid"] is False
        assert len(d["errors"]) == 1
        assert len(d["warnings"]) == 1


class TestContractValidatorInputValidation:
    def test_validate_valid_input(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_input("planner", {"request": "Build a feature", "context": {}})
        assert result.valid is True
        assert result.direction == "input"

    def test_validate_invalid_input(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_input("planner", {"bad_field": "no request"})
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_with_context(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_input(
            "planner",
            {"request": "test", "context": {}},
            context={"workflow_id": "abc"},
        )
        assert result.role == "planner"

    def test_validate_with_agent_role_enum(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_input(AgentRole.BUILDER, {"request": "test", "context": {}})
        assert result.role == "builder"

    def test_strict_mode_raises(self):
        validator = ContractValidator(strict=True)
        with pytest.raises(ContractViolation):
            validator.validate_input("planner", {"bad": "data"})


class TestContractValidatorOutputValidation:
    def test_validate_valid_output(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("planner", {"tasks": [{"id": 1}], "risks": ["none"]})
        assert result.direction == "output"

    def test_planner_quality_many_tasks(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("planner", {"tasks": list(range(15)), "risks": []})
        assert any("15 tasks" in w for w in result.warnings)

    def test_planner_quality_no_risks(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("planner", {"tasks": [1, 2], "risks": []})
        assert any("risks" in w.lower() for w in result.warnings)

    def test_builder_quality_large_code(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("builder", {"code": "x" * 20000})
        assert any("large code" in w.lower() for w in result.warnings)

    def test_builder_quality_partial_no_notes(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("builder", {"code": "print(1)", "status": "partial"})
        assert any("partial" in w.lower() for w in result.warnings)

    def test_tester_quality_no_tests_run(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("tester", {"tests_run": 0})
        assert any("no tests" in w.lower() for w in result.warnings)

    def test_tester_quality_low_coverage(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output("tester", {"tests_run": 10, "coverage": 30})
        assert any("coverage" in w.lower() for w in result.warnings)

    def test_reviewer_quality_approved_with_critical(self):
        validator = ContractValidator(strict=False)
        result = validator.validate_output(
            "reviewer",
            {
                "approved": True,
                "findings": [{"severity": "critical", "message": "SQL injection"}],
            },
        )
        assert any("critical" in w.lower() for w in result.warnings)

    def test_check_output_quality_unknown_role(self):
        """_check_output_quality returns empty list for roles without specific checkers."""
        validator = ContractValidator(strict=False)
        warnings = validator._check_output_quality("migrator", {"data": "test"})
        assert warnings == []


class TestContractValidatorHandoff:
    def test_validate_handoff(self):
        validator = ContractValidator(strict=False)
        out_result, in_result = validator.validate_handoff(
            "planner",
            "builder",
            {"tasks": [1, 2], "risks": ["none"], "request": "test", "context": {}},
        )
        assert out_result.direction == "output"
        assert in_result.direction == "input"


class TestContractValidatorHistory:
    def test_get_history(self):
        validator = ContractValidator(strict=False)
        validator.validate_input("planner", {"request": "test", "context": {}})
        validator.validate_output("builder", {"code": "x"})

        history = validator.get_history()
        assert len(history) == 2
        assert history[0]["direction"] == "input"
        assert history[1]["direction"] == "output"

    def test_get_history_limit(self):
        validator = ContractValidator(strict=False)
        for i in range(30):
            validator.validate_input("planner", {"request": f"test {i}", "context": {}})

        history = validator.get_history(limit=10)
        assert len(history) == 10

    def test_get_stats(self):
        validator = ContractValidator(strict=False)
        validator.validate_input("planner", {"request": "test", "context": {}})
        validator.validate_output("builder", {"code": "x"})

        stats = validator.get_stats()
        assert stats["total"] == 2
        assert "success_rate" in stats
        assert "by_role" in stats

    def test_get_stats_empty(self):
        validator = ContractValidator(strict=False)
        stats = validator.get_stats()
        assert stats["total"] == 0
        assert stats["success_rate"] == 0


class TestValidateWorkflowContracts:
    def test_valid_workflow(self):
        steps = [
            {"id": "plan", "type": "claude_code", "params": {"role": "planner"}},
            {"id": "build", "type": "claude_code", "params": {"role": "builder"}},
        ]
        errors = validate_workflow_contracts(steps)
        assert len(errors) == 0

    def test_invalid_role(self):
        steps = [
            {
                "id": "bad",
                "type": "claude_code",
                "params": {"role": "nonexistent_role_xyz"},
            },
        ]
        errors = validate_workflow_contracts(steps)
        assert len(errors) > 0

    def test_non_ai_steps_skipped(self):
        steps = [
            {"id": "shell", "type": "shell", "params": {"command": "echo hi"}},
        ]
        errors = validate_workflow_contracts(steps)
        assert len(errors) == 0

    def test_steps_without_role(self):
        steps = [
            {"id": "ai", "type": "claude_code", "params": {}},
        ]
        errors = validate_workflow_contracts(steps)
        assert len(errors) == 0  # No role = no validation needed
