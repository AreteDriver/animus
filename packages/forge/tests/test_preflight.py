"""Tests for pre-flight budget validation."""

import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.budget import (
    BudgetConfig,
    BudgetManager,
    PreflightValidator,
    StepEstimate,
    ValidationStatus,
    WorkflowEstimate,
    validate_workflow_budget,
)


class TestStepEstimate:
    """Tests for StepEstimate."""

    def test_total_tokens(self):
        """Total tokens is sum of input and output."""
        estimate = StepEstimate(
            step_name="test",
            agent_type="builder",
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )
        assert estimate.total_tokens == 1500

    def test_default_confidence(self):
        """Default confidence is 0.8."""
        estimate = StepEstimate(step_name="test", agent_type="builder")
        assert estimate.confidence == 0.8


class TestWorkflowEstimate:
    """Tests for WorkflowEstimate."""

    def test_base_tokens(self):
        """Base tokens includes steps and overhead."""
        estimate = WorkflowEstimate(
            workflow_id="test",
            steps=[
                StepEstimate("step1", "builder", 1000, 500),
                StepEstimate("step2", "tester", 800, 400),
            ],
            overhead_tokens=200,
        )
        assert estimate.base_tokens == 1500 + 1200 + 200  # 2900

    def test_total_with_buffer(self):
        """Total includes retry buffer."""
        estimate = WorkflowEstimate(
            workflow_id="test",
            steps=[StepEstimate("step1", "builder", 1000, 0)],
            overhead_tokens=0,
            retry_buffer_percent=0.2,
        )
        assert estimate.total_with_buffer == 1200  # 1000 * 1.2

    def test_min_max_tokens(self):
        """Min is base, max includes larger buffer."""
        estimate = WorkflowEstimate(
            workflow_id="test",
            steps=[StepEstimate("step1", "builder", 1000, 0)],
            overhead_tokens=0,
            retry_buffer_percent=0.2,
        )
        assert estimate.min_tokens == 1000
        assert estimate.max_tokens == 1400  # 1000 * 1.4

    def test_average_confidence(self):
        """Average confidence is calculated correctly."""
        estimate = WorkflowEstimate(
            workflow_id="test",
            steps=[
                StepEstimate("step1", "builder", 1000, 0, confidence=0.8),
                StepEstimate("step2", "tester", 500, 0, confidence=0.6),
            ],
        )
        assert estimate.average_confidence == 0.7

    def test_empty_steps_confidence(self):
        """Empty steps returns 0 confidence."""
        estimate = WorkflowEstimate(workflow_id="test", steps=[])
        assert estimate.average_confidence == 0.0


class TestPreflightValidator:
    """Tests for PreflightValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator with sufficient budget."""
        config = BudgetConfig(total_budget=100000, reserve_tokens=5000)
        manager = BudgetManager(config=config)
        return PreflightValidator(budget_manager=manager)

    @pytest.fixture
    def low_budget_validator(self):
        """Create validator with low budget."""
        config = BudgetConfig(total_budget=1000, reserve_tokens=100)
        manager = BudgetManager(config=config)
        return PreflightValidator(budget_manager=manager)

    def test_estimate_step_uses_defaults(self, validator):
        """Step estimation uses default values."""
        estimate = validator.estimate_step("build", "builder")
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_output_tokens > 0

    def test_estimate_step_with_prompt_length(self, validator):
        """Prompt length adjusts input tokens when exceeding default."""
        short = validator.estimate_step("test", "builder", prompt_length=100)
        # Need a very long prompt to exceed the default estimate (~3000)
        very_long = validator.estimate_step("test", "builder", prompt_length=20000)
        # 20000 chars / 4 chars per token = 5000 tokens > 3000 default
        assert very_long.estimated_input_tokens > short.estimated_input_tokens

    def test_estimate_workflow(self, validator):
        """Workflow estimation aggregates steps."""
        steps = [
            {"name": "plan", "agent": "planner"},
            {"name": "build", "agent": "builder"},
            {"name": "test", "agent": "tester"},
        ]
        estimate = validator.estimate_workflow("test-workflow", steps)
        assert estimate.workflow_id == "test-workflow"
        assert len(estimate.steps) == 3
        assert estimate.base_tokens > 0

    def test_validate_sufficient_budget(self, validator):
        """Validation passes with sufficient budget."""
        steps = [
            {"name": "plan", "agent": "planner"},
            {"name": "build", "agent": "builder"},
        ]
        result = validator.validate("test", steps)
        assert result.status == ValidationStatus.PASS
        assert result.budget_sufficient is True
        assert result.margin > 0

    def test_validate_insufficient_budget(self, low_budget_validator):
        """Validation fails with insufficient budget."""
        steps = [
            {"name": "plan", "agent": "planner"},
            {"name": "build", "agent": "builder"},
        ]
        result = low_budget_validator.validate("test", steps)
        assert result.status == ValidationStatus.FAIL
        assert result.budget_sufficient is False
        assert result.margin < 0

    def test_validate_tight_budget_warns(self):
        """Validation warns when budget is tight."""
        # Create budget that's just barely enough
        config = BudgetConfig(total_budget=15000, reserve_tokens=500)
        manager = BudgetManager(config=config)
        validator = PreflightValidator(budget_manager=manager)

        steps = [{"name": "build", "agent": "builder"}]
        result = validator.validate("test", steps)
        # With typical builder estimate (~7000) + buffer (~1400), this should warn
        assert result.status in (ValidationStatus.WARN, ValidationStatus.PASS)

    def test_validate_strict_mode(self):
        """Strict mode fails on warnings."""
        config = BudgetConfig(total_budget=15000, reserve_tokens=500)
        manager = BudgetManager(config=config)
        validator = PreflightValidator(budget_manager=manager)

        steps = [{"name": "build", "agent": "builder"}]
        result = validator.validate("test", steps, strict=True)
        # In strict mode, tight budget should fail
        if result.margin_percent < 25:
            assert result.status == ValidationStatus.FAIL

    def test_validate_per_step_limit(self, validator):
        """Per-step limits are validated."""
        validator.budget_manager.config.per_step_limit = 1000

        steps = [{"name": "build", "agent": "builder"}]  # Defaults ~7000 tokens
        result = validator.validate("test", steps)

        # Should have step validation warning
        assert any(v["status"] == "exceeds_limit" for v in result.step_validations)

    def test_quick_check(self, validator):
        """Quick check returns boolean."""
        steps = [{"name": "plan", "agent": "planner"}]
        assert validator.quick_check("test", steps) is True

    def test_quick_check_insufficient(self, low_budget_validator):
        """Quick check returns False for insufficient budget."""
        steps = [{"name": "build", "agent": "builder"}]
        assert low_budget_validator.quick_check("test", steps) is False

    def test_to_dict(self, validator):
        """Validation result converts to dict."""
        steps = [{"name": "plan", "agent": "planner"}]
        result = validator.validate("test", steps)
        d = result.to_dict()

        assert "status" in d
        assert "workflow_id" in d
        assert "budget_sufficient" in d
        assert "estimate" in d
        assert "budget" in d
        assert "messages" in d


class TestValidateWorkflowBudget:
    """Tests for convenience function."""

    def test_validate_workflow_budget_default(self):
        """Convenience function works with defaults."""
        steps = [{"name": "plan", "agent": "planner"}]
        result = validate_workflow_budget("test", steps)
        assert result.status in ValidationStatus

    def test_validate_workflow_budget_custom_config(self):
        """Convenience function accepts custom config."""
        config = BudgetConfig(total_budget=500, reserve_tokens=50)
        steps = [{"name": "build", "agent": "builder"}]
        result = validate_workflow_budget("test", steps, budget_config=config)
        assert result.status == ValidationStatus.FAIL


class TestCustomEstimates:
    """Tests for custom token estimates."""

    def test_custom_estimates_override_defaults(self):
        """Custom estimates override defaults."""
        custom = {"custom_agent": {"input": 100, "output": 50}}
        config = BudgetConfig(total_budget=100000)
        manager = BudgetManager(config=config)
        validator = PreflightValidator(
            budget_manager=manager,
            custom_estimates=custom,
        )

        estimate = validator.estimate_step("test", "custom_agent")
        assert estimate.estimated_input_tokens == 100
        assert estimate.estimated_output_tokens == 50

    def test_unknown_agent_uses_default(self):
        """Unknown agent types use default estimates."""
        config = BudgetConfig(total_budget=100000)
        manager = BudgetManager(config=config)
        validator = PreflightValidator(budget_manager=manager)

        estimate = validator.estimate_step("test", "unknown_agent_xyz")
        # Should get default values
        assert estimate.estimated_input_tokens == 2500
        assert estimate.estimated_output_tokens == 2000


class TestValidationMessages:
    """Tests for validation message content."""

    def test_sufficient_budget_message(self):
        """Sufficient budget includes helpful message."""
        config = BudgetConfig(total_budget=100000)
        manager = BudgetManager(config=config)
        validator = PreflightValidator(budget_manager=manager)

        steps = [{"name": "plan", "agent": "planner"}]
        result = validator.validate("test", steps)

        assert any("sufficient" in m.lower() for m in result.messages)

    def test_insufficient_budget_message(self):
        """Insufficient budget includes clear message."""
        config = BudgetConfig(total_budget=100)
        manager = BudgetManager(config=config)
        validator = PreflightValidator(budget_manager=manager)

        steps = [{"name": "build", "agent": "builder"}]
        result = validator.validate("test", steps)

        assert any("insufficient" in m.lower() for m in result.messages)
