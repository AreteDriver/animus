"""Tests for enhanced error recovery in workflow executor."""

import pytest

from animus_forge.workflow.executor import (
    StepStatus,
    WorkflowExecutor,
    configure_circuit_breaker,
    reset_circuit_breakers,
)
from animus_forge.workflow.loader import (
    FallbackConfig,
    StepConfig,
    WorkflowConfig,
)


@pytest.fixture(autouse=True)
def cleanup_circuit_breakers():
    """Reset circuit breakers before and after each test."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


class TestOnFailureModes:
    """Tests for different on_failure modes."""

    def test_on_failure_abort(self):
        """on_failure='abort' stops execution."""
        workflow = WorkflowConfig(
            name="test_abort",
            version="1.0",
            description="Test abort mode",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="abort",
                    max_retries=0,
                ),
                StepConfig(
                    id="never_runs",
                    type="shell",
                    params={"command": "echo done"},
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "failed"
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.FAILED

    def test_on_failure_skip(self):
        """on_failure='skip' continues to next step."""
        workflow = WorkflowConfig(
            name="test_skip",
            version="1.0",
            description="Test skip mode",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="skip",
                    max_retries=0,
                ),
                StepConfig(
                    id="runs_anyway",
                    type="shell",
                    params={"command": "echo done"},
                    outputs=["stdout"],
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"
        assert len(result.steps) == 2
        assert result.steps[0].status == StepStatus.FAILED
        assert result.steps[1].status == StepStatus.SUCCESS

    def test_on_failure_continue_with_default(self):
        """on_failure='continue_with_default' uses default output."""
        workflow = WorkflowConfig(
            name="test_default",
            version="1.0",
            description="Test continue_with_default mode",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="continue_with_default",
                    default_output={"result": "default_value", "status": "fallback"},
                    max_retries=0,
                    outputs=["result"],
                ),
            ],
            outputs=["result"],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"
        assert result.steps[0].status == StepStatus.SUCCESS
        assert result.steps[0].output["result"] == "default_value"
        assert result.outputs["result"] == "default_value"


class TestFallbackStrategies:
    """Tests for fallback strategies."""

    def test_fallback_default_value(self):
        """Fallback with default_value returns configured value."""
        workflow = WorkflowConfig(
            name="test_fallback_value",
            version="1.0",
            description="Test fallback default value",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="fallback",
                    fallback=FallbackConfig(
                        type="default_value",
                        value={"key": "fallback_data"},
                    ),
                    max_retries=0,
                    outputs=["fallback_value"],
                ),
            ],
            outputs=["fallback_value"],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"
        assert result.steps[0].output["fallback_used"] is True
        assert result.steps[0].output["fallback_value"]["key"] == "fallback_data"

    def test_fallback_alternate_step(self):
        """Fallback executes alternate step."""
        workflow = WorkflowConfig(
            name="test_fallback_step",
            version="1.0",
            description="Test fallback alternate step",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="fallback",
                    fallback=FallbackConfig(
                        type="alternate_step",
                        step={
                            "id": "alt_step",
                            "type": "shell",
                            "params": {"command": "echo fallback_success"},
                            "outputs": ["stdout"],
                        },
                    ),
                    max_retries=0,
                    outputs=["stdout"],
                ),
            ],
            outputs=["stdout"],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"
        assert "fallback_success" in result.steps[0].output["stdout"]

    def test_fallback_callback(self):
        """Fallback invokes registered callback."""
        callback_called = []

        def my_callback(step, context, error):
            callback_called.append(step.id)
            return {"callback_result": "handled"}

        workflow = WorkflowConfig(
            name="test_fallback_callback",
            version="1.0",
            description="Test fallback callback",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="fallback",
                    fallback=FallbackConfig(
                        type="callback",
                        callback="my_handler",
                    ),
                    max_retries=0,
                    outputs=["callback_result"],
                ),
            ],
            outputs=["callback_result"],
        )

        executor = WorkflowExecutor(fallback_callbacks={"my_handler": my_callback})
        result = executor.execute(workflow)

        assert result.status == "success"
        assert "failing_step" in callback_called
        assert result.steps[0].output["callback_result"] == "handled"

    def test_fallback_callback_not_registered(self):
        """Fallback fails if callback not registered."""
        workflow = WorkflowConfig(
            name="test_fallback_missing",
            version="1.0",
            description="Test missing callback",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="fallback",
                    fallback=FallbackConfig(
                        type="callback",
                        callback="nonexistent",
                    ),
                    max_retries=0,
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "failed"
        assert "fallback failed" in result.error


class TestErrorCallback:
    """Tests for error callback notifications."""

    def test_error_callback_called_on_failure(self):
        """Error callback is invoked when step fails."""
        errors = []

        def on_error(step_id, workflow_id, error):
            errors.append({"step_id": step_id, "error": str(error)})

        workflow = WorkflowConfig(
            name="test_error_callback",
            version="1.0",
            description="Test error callback",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="abort",
                    max_retries=0,
                ),
            ],
        )

        executor = WorkflowExecutor(error_callback=on_error)
        executor.execute(workflow)

        assert len(errors) == 1
        assert errors[0]["step_id"] == "failing_step"

    def test_error_callback_exception_logged(self):
        """Error callback exceptions are caught and logged."""

        def bad_callback(step_id, workflow_id, error):
            raise RuntimeError("Callback failed")

        workflow = WorkflowConfig(
            name="test_bad_callback",
            version="1.0",
            description="Test bad callback",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="abort",
                    max_retries=0,
                ),
            ],
        )

        executor = WorkflowExecutor(error_callback=bad_callback)
        # Should not raise despite callback error
        result = executor.execute(workflow)

        assert result.status == "failed"


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_blocks_after_failures(self):
        """Circuit breaker opens after failure threshold."""
        # Configure circuit breaker with low threshold
        configure_circuit_breaker("test_cb", failure_threshold=2, recovery_timeout=60)

        workflow = WorkflowConfig(
            name="test_cb",
            version="1.0",
            description="Test circuit breaker",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "exit 1"},
                    circuit_breaker_key="test_cb",
                    on_failure="skip",
                    max_retries=0,
                ),
            ],
        )

        executor = WorkflowExecutor()

        # First failure
        executor.execute(workflow)
        # Second failure - should trip circuit
        executor.execute(workflow)
        # Third attempt - circuit should be open
        result = executor.execute(workflow)

        assert result.steps[0].status == StepStatus.FAILED
        assert "Circuit breaker open" in result.steps[0].error

    def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker resets failure count on success."""
        configure_circuit_breaker("test_reset", failure_threshold=3, recovery_timeout=60)

        # Execute failing step
        fail_workflow = WorkflowConfig(
            name="fail_wf",
            version="1.0",
            description="Failing workflow",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "exit 1"},
                    circuit_breaker_key="test_reset",
                    on_failure="skip",
                    max_retries=0,
                ),
            ],
        )

        # Execute successful step
        success_workflow = WorkflowConfig(
            name="success_wf",
            version="1.0",
            description="Successful workflow",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo ok"},
                    circuit_breaker_key="test_reset",
                    on_failure="skip",
                    max_retries=0,
                ),
            ],
        )

        executor = WorkflowExecutor()

        # Fail once
        executor.execute(fail_workflow)
        # Succeed - resets counter
        executor.execute(success_workflow)
        # Fail twice more - should NOT trip (counter was reset)
        executor.execute(fail_workflow)
        executor.execute(fail_workflow)

        # This should still work (only 2 failures since reset)
        result = executor.execute(success_workflow)
        assert result.steps[0].status == StepStatus.SUCCESS


class TestFallbackConfigParsing:
    """Tests for FallbackConfig parsing."""

    def test_fallback_config_from_dict(self):
        """FallbackConfig parses correctly from dict."""
        data = {
            "type": "default_value",
            "value": {"key": "value"},
        }

        config = FallbackConfig.from_dict(data)

        assert config.type == "default_value"
        assert config.value == {"key": "value"}
        assert config.step is None
        assert config.callback is None

    def test_fallback_config_alternate_step(self):
        """FallbackConfig parses alternate_step config."""
        data = {
            "type": "alternate_step",
            "step": {
                "id": "alt",
                "type": "shell",
                "params": {"command": "echo ok"},
            },
        }

        config = FallbackConfig.from_dict(data)

        assert config.type == "alternate_step"
        assert config.step["id"] == "alt"

    def test_step_config_with_fallback(self):
        """StepConfig parses fallback correctly."""
        data = {
            "id": "test_step",
            "type": "shell",
            "params": {"command": "echo test"},
            "on_failure": "fallback",
            "fallback": {
                "type": "default_value",
                "value": "fallback",
            },
        }

        step = StepConfig.from_dict(data)

        assert step.on_failure == "fallback"
        assert step.fallback is not None
        assert step.fallback.type == "default_value"
        assert step.fallback.value == "fallback"

    def test_step_config_with_default_output(self):
        """StepConfig parses default_output correctly."""
        data = {
            "id": "test_step",
            "type": "shell",
            "params": {"command": "echo test"},
            "on_failure": "continue_with_default",
            "default_output": {"result": "default"},
        }

        step = StepConfig.from_dict(data)

        assert step.on_failure == "continue_with_default"
        assert step.default_output == {"result": "default"}

    def test_step_config_with_circuit_breaker_key(self):
        """StepConfig parses circuit_breaker_key correctly."""
        data = {
            "id": "test_step",
            "type": "shell",
            "params": {"command": "echo test"},
            "circuit_breaker_key": "my_cb_key",
        }

        step = StepConfig.from_dict(data)

        assert step.circuit_breaker_key == "my_cb_key"
