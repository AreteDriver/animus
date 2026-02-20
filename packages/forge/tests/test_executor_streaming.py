"""Tests for WorkflowExecutor → ExecutionManager streaming log wiring."""

import asyncio
from unittest.mock import MagicMock

from animus_forge.workflow.executor_core import WorkflowExecutor
from animus_forge.workflow.loader import StepConfig, WorkflowConfig


def _make_workflow(steps=None, name="Stream Test"):
    """Helper to build a minimal WorkflowConfig."""
    if steps is None:
        steps = [
            StepConfig(
                id="echo-step",
                type="shell",
                params={"command": "echo hello"},
            ),
        ]
    return WorkflowConfig(name=name, version="1.0", description="", steps=steps)


class TestExecutorStreamingBasic:
    """Verify execution_manager receives lifecycle events."""

    def test_create_and_start_called(self):
        """execution_manager.create_execution and start_execution called on execute."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-123"
        mock_em.create_execution.return_value = mock_execution

        executor = WorkflowExecutor(execution_manager=mock_em)
        executor.execute(_make_workflow())

        mock_em.create_execution.assert_called_once()
        mock_em.start_execution.assert_called_once_with("exec-123")

    def test_complete_execution_called_on_success(self):
        """complete_execution called with no error on successful workflow."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-ok"
        mock_em.create_execution.return_value = mock_execution

        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(_make_workflow())

        assert result.status == "success"
        mock_em.complete_execution.assert_called_once_with("exec-ok", error=None)

    def test_complete_execution_called_on_failure(self):
        """complete_execution called with error on failed workflow."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-fail"
        mock_em.create_execution.return_value = mock_execution

        workflow = _make_workflow(
            steps=[
                StepConfig(
                    id="fail-step",
                    type="shell",
                    params={"command": "exit 1"},
                ),
            ]
        )
        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(workflow)

        assert result.status == "failed"
        call_args = mock_em.complete_execution.call_args
        assert call_args[0][0] == "exec-fail"
        assert call_args[1]["error"] is not None


class TestExecutorStreamingStepEvents:
    """Verify step-level events emitted."""

    def test_add_log_called_for_step_start(self):
        """add_log called with 'Starting step' message before step execution."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-log"
        mock_em.create_execution.return_value = mock_execution

        executor = WorkflowExecutor(execution_manager=mock_em)
        executor.execute(_make_workflow())

        # At least one add_log call should mention "Starting step"
        log_calls = [c for c in mock_em.add_log.call_args_list if "Starting step" in str(c)]
        assert len(log_calls) >= 1

    def test_update_progress_called(self):
        """update_progress called with step progress."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-prog"
        mock_em.create_execution.return_value = mock_execution

        executor = WorkflowExecutor(execution_manager=mock_em)
        executor.execute(_make_workflow())

        assert mock_em.update_progress.called

    def test_update_metrics_called_on_step_completion(self):
        """update_metrics called after step completes."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-metrics"
        mock_em.create_execution.return_value = mock_execution

        executor = WorkflowExecutor(execution_manager=mock_em)
        executor.execute(_make_workflow())

        mock_em.update_metrics.assert_called_once()
        call_kwargs = mock_em.update_metrics.call_args
        assert call_kwargs[0][0] == "exec-metrics"

    def test_multi_step_emits_multiple_events(self):
        """Multiple steps emit multiple log and progress events."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-multi"
        mock_em.create_execution.return_value = mock_execution

        workflow = _make_workflow(
            steps=[
                StepConfig(id="s1", type="shell", params={"command": "echo a"}),
                StepConfig(id="s2", type="shell", params={"command": "echo b"}),
                StepConfig(id="s3", type="shell", params={"command": "echo c"}),
            ]
        )
        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(workflow)

        assert result.status == "success"
        # 3 steps → 3 update_progress + 3 update_metrics calls
        assert mock_em.update_progress.call_count == 3
        assert mock_em.update_metrics.call_count == 3


class TestExecutorStreamingResilience:
    """Verify execution_manager failures never break the pipeline."""

    def test_none_execution_manager_works(self):
        """Workflow executes normally with execution_manager=None."""
        executor = WorkflowExecutor(execution_manager=None)
        result = executor.execute(_make_workflow())
        assert result.status == "success"

    def test_create_execution_failure_non_fatal(self):
        """If create_execution raises, workflow still executes."""
        mock_em = MagicMock()
        mock_em.create_execution.side_effect = Exception("DB down")

        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(_make_workflow())

        assert result.status == "success"

    def test_add_log_failure_non_fatal(self):
        """If add_log raises, step still executes."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-log-fail"
        mock_em.create_execution.return_value = mock_execution
        mock_em.add_log.side_effect = Exception("Log write failed")

        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(_make_workflow())

        assert result.status == "success"

    def test_update_metrics_failure_non_fatal(self):
        """If update_metrics raises, workflow still completes."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-metric-fail"
        mock_em.create_execution.return_value = mock_execution
        mock_em.update_metrics.side_effect = Exception("Metrics write failed")

        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(_make_workflow())

        assert result.status == "success"

    def test_complete_execution_failure_non_fatal(self):
        """If complete_execution raises, result is still returned."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-complete-fail"
        mock_em.create_execution.return_value = mock_execution
        mock_em.complete_execution.side_effect = Exception("Complete failed")

        executor = WorkflowExecutor(execution_manager=mock_em)
        result = executor.execute(_make_workflow())

        assert result.status == "success"


class TestExecutorStreamingAsync:
    """Verify async execution emits same events."""

    def test_async_creates_execution(self):
        """execute_async creates and starts execution tracking."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-async"
        mock_em.create_execution.return_value = mock_execution

        executor = WorkflowExecutor(execution_manager=mock_em)
        result = asyncio.run(executor.execute_async(_make_workflow()))

        assert result.status == "success"
        mock_em.create_execution.assert_called_once()
        mock_em.start_execution.assert_called_once_with("exec-async")
        mock_em.complete_execution.assert_called_once_with("exec-async", error=None)

    def test_async_emits_step_events(self):
        """execute_async emits log + progress + metrics for each step."""
        mock_em = MagicMock()
        mock_execution = MagicMock()
        mock_execution.id = "exec-async-steps"
        mock_em.create_execution.return_value = mock_execution

        workflow = _make_workflow(
            steps=[
                StepConfig(id="a1", type="shell", params={"command": "echo x"}),
                StepConfig(id="a2", type="shell", params={"command": "echo y"}),
            ]
        )
        executor = WorkflowExecutor(execution_manager=mock_em)
        result = asyncio.run(executor.execute_async(workflow))

        assert result.status == "success"
        assert mock_em.update_progress.call_count == 2
        assert mock_em.update_metrics.call_count == 2
