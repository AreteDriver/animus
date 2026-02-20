"""Tests for async workflow execution."""

import asyncio

import pytest

from animus_forge.workflow.executor import (
    StepStatus,
    WorkflowExecutor,
    configure_circuit_breaker,
    reset_circuit_breakers,
)
from animus_forge.workflow.loader import (
    ConditionConfig,
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


class TestAsyncExecution:
    """Tests for async execute_async method."""

    @pytest.mark.asyncio
    async def test_async_execute_simple_workflow(self):
        """Async execution works for simple workflow."""
        workflow = WorkflowConfig(
            name="simple_async",
            version="1.0",
            description="Test async execution",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo hello"},
                    outputs=["stdout"],
                ),
            ],
            outputs=["stdout"],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.SUCCESS
        assert "hello" in result.steps[0].output["stdout"]

    @pytest.mark.asyncio
    async def test_async_execute_multiple_steps(self):
        """Async execution handles multiple sequential steps."""
        workflow = WorkflowConfig(
            name="multi_step_async",
            version="1.0",
            description="Test multiple async steps",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo first"},
                    outputs=["stdout"],
                ),
                StepConfig(
                    id="step2",
                    type="shell",
                    params={"command": "echo second"},
                    outputs=["stdout"],
                ),
                StepConfig(
                    id="step3",
                    type="shell",
                    params={"command": "echo third"},
                    outputs=["stdout"],
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert len(result.steps) == 3
        assert all(s.status == StepStatus.SUCCESS for s in result.steps)

    @pytest.mark.asyncio
    async def test_async_execute_with_inputs(self):
        """Async execution substitutes input values."""
        workflow = WorkflowConfig(
            name="input_async",
            version="1.0",
            description="Test async with inputs",
            inputs={"name": {"type": "string", "required": True}},
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo ${name}"},
                    outputs=["stdout"],
                ),
            ],
            outputs=["stdout"],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow, inputs={"name": "Alice"})

        assert result.status == "success"
        assert "Alice" in result.steps[0].output["stdout"]

    @pytest.mark.asyncio
    async def test_async_execute_missing_required_input(self):
        """Async execution fails on missing required input."""
        workflow = WorkflowConfig(
            name="missing_input_async",
            version="1.0",
            description="Test missing input",
            inputs={"name": {"type": "string", "required": True}},
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo test"},
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow, inputs={})

        assert result.status == "failed"
        assert "Missing required input" in result.error

    @pytest.mark.asyncio
    async def test_async_execute_with_default_input(self):
        """Async execution uses default input values."""
        workflow = WorkflowConfig(
            name="default_input_async",
            version="1.0",
            description="Test default input",
            inputs={"name": {"type": "string", "required": True, "default": "World"}},
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo ${name}"},
                    outputs=["stdout"],
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow, inputs={})

        assert result.status == "success"
        assert "World" in result.steps[0].output["stdout"]


class TestAsyncFailureHandling:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_async_on_failure_abort(self):
        """Async aborts workflow on failure with abort mode."""
        workflow = WorkflowConfig(
            name="abort_async",
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
        result = await executor.execute_async(workflow)

        assert result.status == "failed"
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_async_on_failure_skip(self):
        """Async continues workflow on failure with skip mode."""
        workflow = WorkflowConfig(
            name="skip_async",
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
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert len(result.steps) == 2
        assert result.steps[0].status == StepStatus.FAILED
        assert result.steps[1].status == StepStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_async_on_failure_continue_with_default(self):
        """Async uses default output on failure."""
        workflow = WorkflowConfig(
            name="default_async",
            version="1.0",
            description="Test continue_with_default mode",
            steps=[
                StepConfig(
                    id="failing_step",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="continue_with_default",
                    default_output={"result": "fallback_value"},
                    max_retries=0,
                    outputs=["result"],
                ),
            ],
            outputs=["result"],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert result.steps[0].output["result"] == "fallback_value"
        assert result.outputs["result"] == "fallback_value"


class TestAsyncFallback:
    """Tests for async fallback strategies."""

    @pytest.mark.asyncio
    async def test_async_fallback_default_value(self):
        """Async fallback with default_value."""
        workflow = WorkflowConfig(
            name="fallback_value_async",
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
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert result.steps[0].output["fallback_used"] is True
        assert result.steps[0].output["fallback_value"]["key"] == "fallback_data"

    @pytest.mark.asyncio
    async def test_async_fallback_alternate_step(self):
        """Async fallback executes alternate step."""
        workflow = WorkflowConfig(
            name="fallback_step_async",
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
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert "fallback_success" in result.steps[0].output["stdout"]

    @pytest.mark.asyncio
    async def test_async_fallback_callback(self):
        """Async fallback invokes registered callback."""
        callback_called = []

        def my_callback(step, context, error):
            callback_called.append(step.id)
            return {"callback_result": "handled"}

        workflow = WorkflowConfig(
            name="fallback_callback_async",
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
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert "failing_step" in callback_called
        assert result.steps[0].output["callback_result"] == "handled"


class TestAsyncConditions:
    """Tests for async conditional step execution."""

    @pytest.mark.asyncio
    async def test_async_condition_met(self):
        """Async executes step when condition is met."""
        workflow = WorkflowConfig(
            name="condition_met_async",
            version="1.0",
            description="Test condition met",
            steps=[
                StepConfig(
                    id="conditional_step",
                    type="shell",
                    params={"command": "echo executed"},
                    condition=ConditionConfig(
                        field="run_step",
                        operator="equals",
                        value=True,
                    ),
                    outputs=["stdout"],
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow, inputs={"run_step": True})

        assert result.status == "success"
        assert result.steps[0].status == StepStatus.SUCCESS
        assert "executed" in result.steps[0].output["stdout"]

    @pytest.mark.asyncio
    async def test_async_condition_not_met(self):
        """Async skips step when condition is not met."""
        workflow = WorkflowConfig(
            name="condition_skip_async",
            version="1.0",
            description="Test condition not met",
            steps=[
                StepConfig(
                    id="conditional_step",
                    type="shell",
                    params={"command": "echo executed"},
                    condition=ConditionConfig(
                        field="run_step",
                        operator="equals",
                        value=True,
                    ),
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow, inputs={"run_step": False})

        assert result.status == "success"
        assert result.steps[0].status == StepStatus.SKIPPED


class TestAsyncRetries:
    """Tests for async retry behavior."""

    @pytest.mark.asyncio
    async def test_async_retry_on_failure(self):
        """Async retries step on failure."""
        call_count = [0]

        def flaky_handler(step, context):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("Temporary failure")
            return {"result": "success"}

        workflow = WorkflowConfig(
            name="retry_async",
            version="1.0",
            description="Test async retry",
            steps=[
                StepConfig(
                    id="flaky_step",
                    type="custom",
                    params={},
                    max_retries=3,
                ),
            ],
        )

        executor = WorkflowExecutor()
        executor.register_handler("custom", flaky_handler)
        result = await executor.execute_async(workflow)

        assert result.status == "success"
        assert call_count[0] == 3
        assert result.steps[0].retries == 2  # 0-indexed, so 2 means 3rd attempt

    @pytest.mark.asyncio
    async def test_async_max_retries_exceeded(self):
        """Async fails after exceeding max retries."""
        call_count = [0]

        def always_fails(step, context):
            call_count[0] += 1
            raise RuntimeError("Always fails")

        workflow = WorkflowConfig(
            name="max_retry_async",
            version="1.0",
            description="Test max retries",
            steps=[
                StepConfig(
                    id="always_fails",
                    type="custom",
                    params={},
                    max_retries=2,
                    on_failure="abort",
                ),
            ],
        )

        executor = WorkflowExecutor()
        executor.register_handler("custom", always_fails)
        result = await executor.execute_async(workflow)

        assert result.status == "failed"
        assert call_count[0] == 3  # Initial + 2 retries


class TestAsyncCircuitBreaker:
    """Tests for async circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_blocks(self):
        """Async respects open circuit breaker."""
        configure_circuit_breaker("test_async_cb", failure_threshold=2, recovery_timeout=60)

        workflow = WorkflowConfig(
            name="cb_async",
            version="1.0",
            description="Test circuit breaker",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "exit 1"},
                    circuit_breaker_key="test_async_cb",
                    on_failure="skip",
                    max_retries=0,
                ),
            ],
        )

        executor = WorkflowExecutor()

        # Trip the circuit breaker
        await executor.execute_async(workflow)
        await executor.execute_async(workflow)

        # Circuit should now be open
        result = await executor.execute_async(workflow)

        assert result.steps[0].status == StepStatus.FAILED
        assert "Circuit breaker open" in result.steps[0].error


class TestAsyncResume:
    """Tests for async resume functionality."""

    @pytest.mark.asyncio
    async def test_async_resume_from_step(self):
        """Async resumes from specified step."""
        workflow = WorkflowConfig(
            name="resume_async",
            version="1.0",
            description="Test async resume",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo first"},
                ),
                StepConfig(
                    id="step2",
                    type="shell",
                    params={"command": "echo second"},
                    outputs=["stdout"],
                ),
                StepConfig(
                    id="step3",
                    type="shell",
                    params={"command": "echo third"},
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = await executor.execute_async(workflow, resume_from="step2")

        assert result.status == "success"
        assert len(result.steps) == 2
        assert result.steps[0].step_id == "step2"
        assert result.steps[1].step_id == "step3"


class TestAsyncErrorCallback:
    """Tests for async error callback notifications."""

    @pytest.mark.asyncio
    async def test_async_error_callback_called(self):
        """Async error callback is invoked on failure."""
        errors = []

        def on_error(step_id, workflow_id, error):
            errors.append({"step_id": step_id, "error": str(error)})

        workflow = WorkflowConfig(
            name="error_cb_async",
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
        await executor.execute_async(workflow)

        assert len(errors) == 1
        assert errors[0]["step_id"] == "failing_step"


class TestAsyncConcurrency:
    """Tests for async concurrent workflow execution."""

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Multiple workflows can execute concurrently."""
        workflow = WorkflowConfig(
            name="concurrent_async",
            version="1.0",
            description="Test concurrent execution",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo workflow"},
                    outputs=["stdout"],
                ),
            ],
        )

        executor = WorkflowExecutor()

        # Execute multiple workflows concurrently
        results = await asyncio.gather(
            executor.execute_async(workflow, inputs={"id": "1"}),
            executor.execute_async(workflow, inputs={"id": "2"}),
            executor.execute_async(workflow, inputs={"id": "3"}),
        )

        assert len(results) == 3
        assert all(r.status == "success" for r in results)

    @pytest.mark.asyncio
    async def test_async_does_not_block_event_loop(self):
        """Async execution doesn't block the event loop."""

        workflow = WorkflowConfig(
            name="nonblocking_async",
            version="1.0",
            description="Test non-blocking",
            steps=[
                StepConfig(
                    id="slow_step",
                    type="shell",
                    params={"command": "sleep 0.1"},
                ),
            ],
        )

        executor = WorkflowExecutor()

        # Track if other async tasks can run during execution
        other_task_ran = [False]

        async def other_task():
            await asyncio.sleep(0.05)
            other_task_ran[0] = True

        # Run workflow and other task concurrently
        await asyncio.gather(
            executor.execute_async(workflow),
            other_task(),
        )

        assert other_task_ran[0], "Other async task should have run during workflow execution"
