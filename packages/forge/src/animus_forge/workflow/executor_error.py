"""Error handling logic for workflow executor.

Mixin class providing step failure handling and error recovery strategies.
"""

from __future__ import annotations

import logging

from .executor_results import ExecutionResult, StepResult, StepStatus
from .loader import StepConfig

logger = logging.getLogger(__name__)


class ErrorHandlerMixin:
    """Mixin providing step failure handling.

    Expects the following attributes from the host class:
    - error_callback: Callable | None
    - fallback_callbacks: dict
    - _execute_fallback(step, error, workflow_id) -> dict | None
    - _execute_fallback_async(step, error, workflow_id) -> dict | None
    """

    def _handle_step_failure(
        self,
        step: StepConfig,
        step_result: StepResult,
        result: ExecutionResult,
        workflow_id: str | None,
    ) -> str:
        """Handle step failure based on on_failure strategy.

        Args:
            step: The failed step configuration
            step_result: The step result to potentially update
            result: The workflow result to update on abort
            workflow_id: Current workflow ID

        Returns:
            Action to take: "abort", "skip", "continue", or "recovered"
        """
        # Notify error callback
        if self.error_callback:
            try:
                self.error_callback(
                    step.id,
                    workflow_id or "",
                    Exception(step_result.error or "Unknown error"),
                )
            except Exception as cb_err:
                logger.warning(f"Error callback failed: {cb_err}")

        if step.on_failure == "abort":
            result.status = "failed"
            result.error = f"Step '{step.id}' failed: {step_result.error}"
            return "abort"

        if step.on_failure == "skip":
            return "skip"

        if step.on_failure == "continue_with_default":
            step_result.status = StepStatus.SUCCESS
            step_result.output = step.default_output.copy()
            logger.info(f"Step '{step.id}' failed, using default output")
            return "continue"

        if step.on_failure == "fallback" and step.fallback:
            fallback_output = self._execute_fallback(step, step_result.error, workflow_id)
            if fallback_output is not None:
                step_result.status = StepStatus.SUCCESS
                step_result.output = fallback_output
                logger.info(f"Step '{step.id}' recovered via fallback")
                return "recovered"
            result.status = "failed"
            result.error = f"Step '{step.id}' failed and fallback failed"
            return "abort"

        # Default: abort
        result.status = "failed"
        result.error = f"Step '{step.id}' failed: {step_result.error}"
        return "abort"

    async def _handle_step_failure_async(
        self,
        step: StepConfig,
        step_result: StepResult,
        result: ExecutionResult,
        workflow_id: str | None,
    ) -> str:
        """Async version of _handle_step_failure."""
        # Notify error callback
        if self.error_callback:
            try:
                self.error_callback(
                    step.id,
                    workflow_id or "",
                    Exception(step_result.error or "Unknown error"),
                )
            except Exception as cb_err:
                logger.warning(f"Error callback failed: {cb_err}")

        if step.on_failure == "abort":
            result.status = "failed"
            result.error = f"Step '{step.id}' failed: {step_result.error}"
            return "abort"

        if step.on_failure == "skip":
            return "skip"

        if step.on_failure == "continue_with_default":
            step_result.status = StepStatus.SUCCESS
            step_result.output = step.default_output.copy()
            logger.info(f"Step '{step.id}' failed, using default output")
            return "continue"

        if step.on_failure == "fallback" and step.fallback:
            fallback_output = await self._execute_fallback_async(
                step, step_result.error, workflow_id
            )
            if fallback_output is not None:
                step_result.status = StepStatus.SUCCESS
                step_result.output = fallback_output
                logger.info(f"Step '{step.id}' recovered via fallback")
                return "recovered"
            result.status = "failed"
            result.error = f"Step '{step.id}' failed and fallback failed"
            return "abort"

        # Default: abort
        result.status = "failed"
        result.error = f"Step '{step.id}' failed: {step_result.error}"
        return "abort"
