"""Step execution logic for workflow executor.

Mixin class providing single-step execution, precondition checking,
handler invocation, and fallback strategies.
"""

from __future__ import annotations

import asyncio
import logging
import time

from animus_forge.utils.circuit_breaker import CircuitBreakerError

from .executor_clients import get_circuit_breaker
from .executor_results import StepResult, StepStatus
from .loader import StepConfig

logger = logging.getLogger(__name__)


class StepExecutionMixin:
    """Mixin providing single-step execution logic.

    Expects the following attributes from the host class:
    - _handlers: dict[str, StepHandler]
    - _context: dict
    - contract_validator: ContractValidator | None
    - checkpoint_manager: CheckpointManager | None
    - fallback_callbacks: dict
    """

    def _check_step_preconditions(self, step: StepConfig, result: StepResult) -> tuple:
        """Check step preconditions and return handler/circuit breaker.

        Returns:
            Tuple of (handler, circuit_breaker, error_message)
            If error_message is set, the step should fail with that message.
        """
        # Check condition
        if step.condition and not step.condition.evaluate(self._context):
            result.status = StepStatus.SKIPPED
            return None, None, None

        # Validate input contract if applicable
        role = step.params.get("role")
        if self.contract_validator and role:
            try:
                input_data = step.params.get("input", {})
                self.contract_validator.validate_input(role, input_data, self._context)
            except Exception as e:
                return None, None, f"Input validation failed: {e}"

        handler = self._handlers.get(step.type)
        if not handler:
            return None, None, f"Unknown step type: {step.type}"

        cb_key = step.circuit_breaker_key or step.type
        cb = get_circuit_breaker(cb_key)
        if cb and cb.is_open:
            logger.warning(f"Step '{step.id}' skipped: circuit breaker open")
            return None, None, f"Circuit breaker open for {cb_key}"

        return handler, cb, None

    def _invoke_handler(self, handler, step: StepConfig, cb) -> dict:
        """Invoke step handler with optional circuit breaker."""
        if cb:
            return cb.call(handler, step, self._context)
        return handler(step, self._context)

    async def _invoke_handler_async(self, handler, step: StepConfig, cb) -> dict:
        """Invoke step handler asynchronously with optional circuit breaker."""
        loop = asyncio.get_event_loop()
        if cb:
            return await loop.run_in_executor(None, cb.call, handler, step, self._context)
        return await loop.run_in_executor(None, handler, step, self._context)

    def _execute_step(self, step: StepConfig, workflow_id: str = None) -> StepResult:
        """Execute a single workflow step."""
        start_time = time.time()
        result = StepResult(step_id=step.id, status=StepStatus.PENDING)

        handler, cb, error = self._check_step_preconditions(step, result)
        if result.status == StepStatus.SKIPPED:
            return result
        if error:
            result.status = StepStatus.FAILED
            result.error = error
            return result

        role = step.params.get("role")
        last_error = None

        for attempt in range(step.max_retries + 1):
            result.retries = attempt
            result.status = StepStatus.RUNNING

            try:
                if self.checkpoint_manager and workflow_id:
                    with self.checkpoint_manager.stage(
                        step.id, input_data=step.params, workflow_id=workflow_id
                    ) as ctx:
                        output = self._invoke_handler(handler, step, cb)
                        ctx.output_data = output
                        ctx.tokens_used = output.get("tokens_used", 0)
                else:
                    output = self._invoke_handler(handler, step, cb)

                if self.contract_validator and role:
                    self.contract_validator.validate_output(role, output)

                result.status = StepStatus.SUCCESS
                result.output = output
                result.tokens_used = output.get("tokens_used", 0)
                break

            except CircuitBreakerError as e:
                result.status = StepStatus.FAILED
                result.error = str(e)
                break

            except Exception as e:
                last_error = str(e)
                if attempt < step.max_retries:
                    time.sleep(min(2**attempt, 30))
                else:
                    result.status = StepStatus.FAILED
                    result.error = last_error

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def _execute_step_async(self, step: StepConfig, workflow_id: str = None) -> StepResult:
        """Execute a single workflow step asynchronously."""
        start_time = time.time()
        result = StepResult(step_id=step.id, status=StepStatus.PENDING)

        handler, cb, error = self._check_step_preconditions(step, result)
        if result.status == StepStatus.SKIPPED:
            return result
        if error:
            result.status = StepStatus.FAILED
            result.error = error
            return result

        role = step.params.get("role")
        last_error = None

        for attempt in range(step.max_retries + 1):
            result.retries = attempt
            result.status = StepStatus.RUNNING

            try:
                if self.checkpoint_manager and workflow_id:
                    with self.checkpoint_manager.stage(
                        step.id, input_data=step.params, workflow_id=workflow_id
                    ) as ctx:
                        output = await self._invoke_handler_async(handler, step, cb)
                        ctx.output_data = output
                        ctx.tokens_used = output.get("tokens_used", 0)
                else:
                    output = await self._invoke_handler_async(handler, step, cb)

                if self.contract_validator and role:
                    self.contract_validator.validate_output(role, output)

                result.status = StepStatus.SUCCESS
                result.output = output
                result.tokens_used = output.get("tokens_used", 0)
                break

            except CircuitBreakerError as e:
                result.status = StepStatus.FAILED
                result.error = str(e)
                break

            except Exception as e:
                last_error = str(e)
                if attempt < step.max_retries:
                    await asyncio.sleep(min(2**attempt, 30))
                else:
                    result.status = StepStatus.FAILED
                    result.error = last_error

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    def _execute_fallback(
        self, step: StepConfig, error: str | None, workflow_id: str | None
    ) -> dict | None:
        """Execute fallback strategy for a failed step.

        Args:
            step: The failed step configuration
            error: The error message from the failed step
            workflow_id: Current workflow ID

        Returns:
            Fallback output dict, or None if fallback failed
        """
        if not step.fallback:
            return None

        fallback = step.fallback
        logger.info(f"Executing fallback ({fallback.type}) for step '{step.id}'")

        try:
            if fallback.type == "default_value":
                # Return the configured default value
                return {"fallback_value": fallback.value, "fallback_used": True}

            elif fallback.type == "alternate_step" and fallback.step:
                # Execute an alternate step
                alt_step = StepConfig.from_dict(fallback.step)
                alt_result = self._execute_step(alt_step, workflow_id)
                if alt_result.status == StepStatus.SUCCESS:
                    return alt_result.output
                return None

            elif fallback.type == "callback" and fallback.callback:
                # Invoke a registered callback
                if fallback.callback in self.fallback_callbacks:
                    callback = self.fallback_callbacks[fallback.callback]
                    return callback(step, self._context, Exception(error or "Unknown"))
                else:
                    logger.warning(f"Fallback callback '{fallback.callback}' not registered")
                    return None

        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            return None

        return None

    async def _execute_fallback_async(
        self, step: StepConfig, error: str | None, workflow_id: str | None
    ) -> dict | None:
        """Execute fallback strategy for a failed step asynchronously.

        Args:
            step: The failed step configuration
            error: The error message from the failed step
            workflow_id: Current workflow ID

        Returns:
            Fallback output dict, or None if fallback failed
        """
        if not step.fallback:
            return None

        fallback = step.fallback
        logger.info(f"Executing fallback ({fallback.type}) for step '{step.id}'")

        try:
            if fallback.type == "default_value":
                # Return the configured default value
                return {"fallback_value": fallback.value, "fallback_used": True}

            elif fallback.type == "alternate_step" and fallback.step:
                # Execute an alternate step
                alt_step = StepConfig.from_dict(fallback.step)
                alt_result = await self._execute_step_async(alt_step, workflow_id)
                if alt_result.status == StepStatus.SUCCESS:
                    return alt_result.output
                return None

            elif fallback.type == "callback" and fallback.callback:
                # Invoke a registered callback
                if fallback.callback in self.fallback_callbacks:
                    callback = self.fallback_callbacks[fallback.callback]
                    # Run callback in executor if not async
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        callback,
                        step,
                        self._context,
                        Exception(error or "Unknown"),
                    )
                else:
                    logger.warning(f"Fallback callback '{fallback.callback}' not registered")
                    return None

        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            return None

        return None
