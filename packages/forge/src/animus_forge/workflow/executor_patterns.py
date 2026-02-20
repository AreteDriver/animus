"""Distribution pattern step handlers for workflow execution.

Mixin class providing parallel, fan-out, fan-in, and map-reduce patterns.
"""

from __future__ import annotations

import logging
import time

from animus_forge.monitoring.parallel_tracker import (
    ParallelPatternType,
    get_parallel_tracker,
)

from .loader import StepConfig
from .parallel import ParallelExecutor, ParallelStrategy, ParallelTask
from .rate_limited_executor import RateLimitedParallelExecutor

logger = logging.getLogger(__name__)


class DistributionPatternsMixin:
    """Mixin providing distribution pattern step handlers.

    Expects the following attributes from the host class:
    - _handlers: dict[str, StepHandler]
    - _context: dict
    - budget_manager: BudgetManager | None
    - checkpoint_manager: CheckpointManager | None
    - _current_workflow_id: str | None
    - fallback_callbacks: dict
    """

    def _check_sub_step_budget(self, sub_step: StepConfig, stage_name: str) -> None:
        """Check budget allocation for a sub-step.

        Raises:
            RuntimeError: If budget is exceeded
        """
        if not self.budget_manager:
            return
        estimated_tokens = sub_step.params.get("estimated_tokens", 1000)
        if not self.budget_manager.can_allocate(estimated_tokens, agent_id=stage_name):
            raise RuntimeError(f"Budget exceeded for sub-step '{sub_step.id}'")

    def _record_sub_step_metrics(
        self,
        stage_name: str,
        sub_step: StepConfig,
        parent_step_id: str,
        tokens_used: int,
        duration_ms: int,
        retries_used: int,
        output: dict | None,
        error_msg: str | None,
    ) -> None:
        """Record budget and checkpoint metrics for a sub-step."""
        if self.budget_manager and tokens_used > 0:
            self.budget_manager.record_usage(
                agent_id=stage_name,
                tokens=tokens_used,
                operation=f"parallel:{sub_step.type}",
                metadata={
                    "parent_step": parent_step_id,
                    "sub_step": sub_step.id,
                    "step_type": sub_step.type,
                    "duration_ms": duration_ms,
                    "retries": retries_used,
                },
            )

        if self.checkpoint_manager and self._current_workflow_id:
            self.checkpoint_manager.checkpoint_now(
                stage=stage_name,
                status="success" if error_msg is None else "failed",
                input_data=sub_step.params,
                output_data=output if output else {"error": error_msg},
                tokens_used=tokens_used,
                duration_ms=duration_ms,
                workflow_id=self._current_workflow_id,
            )

    def _execute_sub_step_attempt(
        self,
        sub_step: StepConfig,
        stage_name: str,
        context: dict,
        context_updates: dict,
    ) -> tuple[dict, int]:
        """Execute a single sub-step attempt.

        Returns:
            Tuple of (output dict, tokens_used)
        """
        self._check_sub_step_budget(sub_step, stage_name)

        step_context = context.copy()
        step_context.update(context_updates)

        step_handler = self._handlers.get(sub_step.type)
        if not step_handler:
            raise ValueError(f"Unknown step type: {sub_step.type}")

        output = step_handler(sub_step, step_context)
        tokens_used = output.get("tokens_used", 0)

        for output_key in sub_step.outputs:
            if output_key in output:
                context_updates[output_key] = output[output_key]

        return output, tokens_used

    def _execute_with_retries(
        self,
        sub_step: StepConfig,
        stage_name: str,
        context: dict,
        context_updates: dict,
    ) -> tuple[dict | None, int, str | None, int]:
        """Execute a sub-step with retry logic.

        Args:
            sub_step: Step configuration
            stage_name: Stage name for metrics
            context: Execution context
            context_updates: Dictionary to collect output updates

        Returns:
            Tuple of (output, tokens_used, error_msg, retries_used)
        """
        output, error_msg, tokens_used, retries_used = None, None, 0, 0

        for attempt in range(sub_step.max_retries + 1):
            try:
                output, tokens_used = self._execute_sub_step_attempt(
                    sub_step, stage_name, context, context_updates
                )
                output["retries"] = retries_used
                return output, tokens_used, None, retries_used
            except Exception as e:
                error_msg = str(e)
                retries_used = attempt + 1
                if attempt < sub_step.max_retries:
                    time.sleep(min(2**attempt, 30))
                elif attempt == sub_step.max_retries:
                    raise

        return output, tokens_used, error_msg, retries_used

    def _execute_parallel(self, step: StepConfig, context: dict) -> dict:
        """Execute parallel sub-steps using ParallelExecutor.

        Params:
            steps: List of sub-step configurations
            strategy: "threading" | "asyncio" (default: "threading")
            max_workers: int (default: 4)
            fail_fast: bool - if True, abort on first failure (default: False)
            rate_limit: bool - if True, use rate-limited executor for AI steps (default: True)
            anthropic_concurrent: int - max concurrent Anthropic calls (default: 5)
            openai_concurrent: int - max concurrent OpenAI calls (default: 8)
        """
        sub_steps = step.params.get("steps", [])
        if not sub_steps:
            return {"parallel_results": {}, "tokens_used": 0}

        # Check if any sub-steps are AI steps
        ai_step_types = {"claude_code", "openai"}
        parsed_step_types = {s.get("type") for s in sub_steps}
        has_ai_steps = bool(parsed_step_types & ai_step_types)

        # Use rate-limited executor for AI steps (unless explicitly disabled)
        use_rate_limiting = step.params.get("rate_limit", True) and has_ai_steps

        strategy_map = {
            "threading": ParallelStrategy.THREADING,
            "asyncio": ParallelStrategy.ASYNCIO,
            "process": ParallelStrategy.PROCESS,
        }

        if use_rate_limiting:
            # Force asyncio strategy for rate limiting
            executor = RateLimitedParallelExecutor(
                strategy=ParallelStrategy.ASYNCIO,
                max_workers=step.params.get("max_workers", 4),
                timeout=step.timeout_seconds or 300.0,
                provider_limits={
                    "anthropic": step.params.get("anthropic_concurrent", 5),
                    "openai": step.params.get("openai_concurrent", 8),
                },
            )
            logger.debug(
                f"Using rate-limited executor for parallel step '{step.id}' "
                f"(AI steps detected: {parsed_step_types & ai_step_types})"
            )
        else:
            strategy = strategy_map.get(
                step.params.get("strategy", "threading"), ParallelStrategy.THREADING
            )
            executor = ParallelExecutor(
                strategy=strategy,
                max_workers=step.params.get("max_workers", 4),
                timeout=step.timeout_seconds,
            )

        parsed_steps = {StepConfig.from_dict(s).id: StepConfig.from_dict(s) for s in sub_steps}

        results: dict[str, dict] = {}
        total_tokens = 0
        first_error: Exception | None = None
        context_updates: dict[str, any] = {}
        parent_step_id = step.id
        fail_fast = step.params.get("fail_fast", False)

        def make_handler(sub_step: StepConfig):
            def handler(**kwargs):
                # kwargs may contain step_type from rate-limited executor
                nonlocal total_tokens
                start_time = time.time()
                stage_name = f"{parent_step_id}.{sub_step.id}"
                output, tokens_used, error_msg, retries_used = None, 0, None, 0

                try:
                    output, tokens_used, error_msg, retries_used = self._execute_with_retries(
                        sub_step, stage_name, context, context_updates
                    )
                    total_tokens += tokens_used
                    return output
                except Exception as e:
                    # Capture error message for checkpoint recording
                    error_msg = str(e)
                    raise
                finally:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._record_sub_step_metrics(
                        stage_name,
                        sub_step,
                        parent_step_id,
                        tokens_used,
                        duration_ms,
                        retries_used,
                        output,
                        error_msg,
                    )

            return handler

        tasks = [
            ParallelTask(
                id=sub_step.id,
                step_id=sub_step.id,
                handler=make_handler(sub_step),
                dependencies=sub_step.depends_on,
                kwargs={"step_type": sub_step.type},  # For rate limiter provider detection
            )
            for sub_step in parsed_steps.values()
        ]

        def on_complete(task_id: str, result: any):
            results[task_id] = result

        def on_error(task_id: str, error: Exception):
            nonlocal first_error
            results[task_id] = {"error": str(error)}
            if first_error is None:
                first_error = error

        try:
            parallel_result = executor.execute_parallel(
                tasks=tasks,
                on_complete=on_complete,
                on_error=on_error,
                fail_fast=fail_fast,
            )
        except ValueError as e:
            raise RuntimeError(f"Parallel execution failed: {e}")

        if fail_fast and first_error is not None:
            raise RuntimeError(f"Parallel step failed: {first_error}")

        self._context.update(context_updates)

        total_retries = sum(r.get("retries", 0) for r in results.values() if isinstance(r, dict))

        return {
            "parallel_results": results,
            "tokens_used": total_tokens,
            "successful": parallel_result.successful,
            "failed": parallel_result.failed,
            "cancelled": parallel_result.cancelled,
            "duration_ms": parallel_result.total_duration_ms,
            "total_retries": total_retries,
        }

    def _substitute_template_vars(self, template: str, item: any, index: int) -> str:
        """Substitute template variables with item value and index.

        Supports:
            ${item} - The current item value
            ${index} - The current item index (0-based)
            ${context_var} - Any variable from execution context
        """
        result = template.replace("${item}", str(item))
        result = result.replace("${index}", str(index))

        # Also substitute context variables
        for key, value in self._context.items():
            if isinstance(value, str):
                result = result.replace(f"${{{key}}}", value)

        return result

    def _execute_fan_out(self, step: StepConfig, context: dict) -> dict:
        """Execute a fan-out (scatter) step.

        Iterates over a list of items and executes a step template for each
        item concurrently with rate limiting.

        Params:
            items: Expression that resolves to a list (e.g., "${files}")
            max_concurrent: Maximum concurrent executions (default: 5)
            step_template: Step configuration template with ${item} placeholders
            fail_fast: If True, abort on first failure (default: False)
            collect_errors: If True, include errors in results (default: True)

        Returns:
            Dict with:
            - results: List of results from each item
            - successful: Count of successful executions
            - failed: Count of failed executions
            - tokens_used: Total tokens used
        """
        # Resolve items from context
        items_expr = step.params.get("items", [])
        if isinstance(items_expr, str) and items_expr.startswith("${") and items_expr.endswith("}"):
            var_name = items_expr[2:-1]
            items = context.get(var_name, [])
        else:
            items = items_expr

        if not isinstance(items, list):
            raise ValueError(f"fan_out items must be a list, got {type(items)}")

        if not items:
            return {
                "results": [],
                "successful": 0,
                "failed": 0,
                "tokens_used": 0,
            }

        max_concurrent = step.params.get("max_concurrent", 5)
        step_template = step.params.get("step_template", {})
        fail_fast = step.params.get("fail_fast", False)
        collect_errors = step.params.get("collect_errors", True)

        if not step_template:
            raise ValueError("fan_out requires step_template parameter")

        # Start parallel execution tracking
        tracker = get_parallel_tracker()
        execution_id = f"fan_out_{step.id}_{int(time.time() * 1000)}"
        tracker.start_execution(
            execution_id=execution_id,
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id=step.id,
            total_items=len(items),
            max_concurrent=max_concurrent,
            workflow_id=context.get("workflow_id"),
        )

        # Build parallel tasks for each item
        tasks = []
        for idx, item in enumerate(items):
            # Create a step config from template with substituted values
            template_copy = step_template.copy()
            params = template_copy.get("params", {}).copy()

            # Substitute ${item} and ${index} in prompt and other string params
            for key, value in params.items():
                if isinstance(value, str):
                    params[key] = self._substitute_template_vars(value, item, idx)

            template_copy["params"] = params
            template_copy["id"] = f"{step.id}_item_{idx}"
            template_copy["outputs"] = template_copy.get("outputs", [])

            sub_step = StepConfig.from_dict(template_copy)
            branch_id = f"{step.id}_item_{idx}"

            # Create task with tracking
            def make_handler(
                sub_step_config: StepConfig,
                item_value: any,
                item_idx: int,
                b_id: str,
            ):
                def handler(**kwargs):
                    # Start branch tracking
                    tracker.start_branch(execution_id, b_id, item_idx, item_value)

                    try:
                        step_handler = self._handlers.get(sub_step_config.type)
                        if not step_handler:
                            raise ValueError(f"Unknown step type: {sub_step_config.type}")

                        # Add item to context for this execution
                        item_context = context.copy()
                        item_context["item"] = item_value
                        item_context["index"] = item_idx

                        output = step_handler(sub_step_config, item_context)
                        tokens = output.get("tokens_used", 0) if output else 0

                        # Complete branch tracking
                        tracker.complete_branch(execution_id, b_id, tokens)

                        return {
                            "item": item_value,
                            "index": item_idx,
                            "output": output,
                        }
                    except Exception as e:
                        tracker.fail_branch(execution_id, b_id, str(e))
                        raise

                return handler

            task = ParallelTask(
                id=branch_id,
                step_id=sub_step.id,
                handler=make_handler(sub_step, item, idx, branch_id),
                kwargs={"step_type": sub_step.type},
            )
            tasks.append(task)

        # Rate limiting configuration from step params
        adaptive_enabled = step.params.get("adaptive_rate_limiting", True)
        distributed_enabled = step.params.get("distributed_rate_limiting", False)
        backoff_factor = step.params.get("rate_limit_backoff_factor", 0.5)
        recovery_threshold = step.params.get("rate_limit_recovery_threshold", 10)

        # Use rate-limited executor with full configuration
        from .rate_limited_executor import AdaptiveRateLimitConfig

        adaptive_config = (
            AdaptiveRateLimitConfig(
                backoff_factor=backoff_factor,
                recovery_threshold=recovery_threshold,
            )
            if adaptive_enabled
            else None
        )

        executor = RateLimitedParallelExecutor(
            strategy=ParallelStrategy.ASYNCIO,
            max_workers=max_concurrent,
            timeout=step.timeout_seconds or 300.0,
            provider_limits={
                "anthropic": step.params.get("anthropic_concurrent", 5),
                "openai": step.params.get("openai_concurrent", 8),
            },
            adaptive=adaptive_enabled,
            adaptive_config=adaptive_config,
            distributed=distributed_enabled,
            distributed_window=step.params.get("distributed_window", 60),
            distributed_rpm={
                "anthropic": step.params.get("anthropic_rpm", 60),
                "openai": step.params.get("openai_rpm", 90),
            },
        )

        results: list[dict] = []
        errors: list[dict] = []
        total_tokens = 0

        def on_complete(task_id: str, result: any):
            nonlocal total_tokens
            results.append(result)
            if isinstance(result, dict) and "output" in result:
                total_tokens += result["output"].get("tokens_used", 0)

        def on_error(task_id: str, error: Exception):
            if collect_errors:
                errors.append(
                    {
                        "task_id": task_id,
                        "error": str(error),
                    }
                )

        parallel_result = executor.execute_parallel(
            tasks=tasks,
            on_complete=on_complete,
            on_error=on_error,
            fail_fast=fail_fast,
        )

        # Capture rate limit stats before completing tracking
        provider_stats = executor.get_provider_stats()

        # Update tracker with rate limit state for each provider
        for provider, stats in provider_stats.items():
            if stats.get("total_429s", 0) > 0 or stats.get("is_throttled", False):
                tracker.update_rate_limit_state(
                    provider=provider,
                    current_limit=stats.get("current_limit", 0),
                    base_limit=stats.get("base_limit", 0),
                    total_429s=stats.get("total_429s", 0),
                    is_throttled=stats.get("is_throttled", False),
                )

        # Complete execution tracking
        if parallel_result.failed:
            tracker.fail_execution(execution_id)
        else:
            tracker.complete_execution(execution_id)

        # Sort results by index
        results.sort(key=lambda x: x.get("index", 0))

        # Collect all results for output variable
        all_results = [r.get("output", {}).get("response", r.get("output", {})) for r in results]

        return {
            "results": all_results,
            "detailed_results": results,
            "errors": errors if collect_errors else [],
            "successful": len(parallel_result.successful),
            "failed": len(parallel_result.failed),
            "cancelled": len(parallel_result.cancelled),
            "tokens_used": total_tokens,
            "duration_ms": parallel_result.total_duration_ms,
            "execution_id": execution_id,
            "rate_limit_stats": provider_stats,
        }

    def _execute_fan_in(self, step: StepConfig, context: dict) -> dict:
        """Execute a fan-in (gather) step.

        Aggregates results from a previous step (typically fan_out) and
        optionally processes them through an aggregation step.

        Params:
            input: Expression that resolves to the input list (e.g., "${reviews}")
            aggregation: "concat" | "claude_code" | "openai" | "custom"
            aggregate_prompt: Prompt template for AI aggregation
            separator: Separator for concat aggregation (default: "\\n")

        Returns:
            Dict with aggregated result
        """
        # Resolve input from context
        input_expr = step.params.get("input", [])
        if isinstance(input_expr, str) and input_expr.startswith("${") and input_expr.endswith("}"):
            var_name = input_expr[2:-1]
            input_data = context.get(var_name, [])
        else:
            input_data = input_expr

        if not isinstance(input_data, list):
            input_data = [input_data]

        aggregation = step.params.get("aggregation", "concat")
        aggregate_prompt = step.params.get("aggregate_prompt", "")

        if aggregation == "concat":
            separator = step.params.get("separator", "\n")
            result = separator.join(str(item) for item in input_data)
            return {
                "response": result,
                "aggregation_type": "concat",
                "item_count": len(input_data),
                "tokens_used": 0,
            }

        elif aggregation in ("claude_code", "openai"):
            # Use AI to aggregate results
            items_text = "\n---\n".join(str(item) for item in input_data)
            prompt = aggregate_prompt.replace("${items}", items_text)

            # Substitute other context variables
            for key, value in context.items():
                if isinstance(value, str):
                    prompt = prompt.replace(f"${{{key}}}", value)

            # Create sub-step for aggregation
            agg_step_config = {
                "id": f"{step.id}_aggregate",
                "type": aggregation,
                "params": {
                    "prompt": prompt,
                    "role": step.params.get("role", "analyst"),
                    "model": step.params.get("model"),
                    "max_tokens": step.params.get("max_tokens", 4096),
                },
            }
            agg_step = StepConfig.from_dict(agg_step_config)

            handler = self._handlers.get(aggregation)
            if not handler:
                raise ValueError(f"Unknown aggregation type: {aggregation}")

            output = handler(agg_step, context)
            return {
                "response": output.get("response", ""),
                "aggregation_type": aggregation,
                "item_count": len(input_data),
                "tokens_used": output.get("tokens_used", 0),
            }

        elif aggregation == "custom":
            # Custom aggregation via callback
            callback_name = step.params.get("callback")
            if callback_name and callback_name in self.fallback_callbacks:
                callback = self.fallback_callbacks[callback_name]
                result = callback(step, context, None)
                return {
                    "response": result,
                    "aggregation_type": "custom",
                    "item_count": len(input_data),
                    "tokens_used": 0,
                }
            raise ValueError(f"Custom callback '{callback_name}' not registered")

        else:
            raise ValueError(f"Unknown aggregation type: {aggregation}")

    def _execute_map_reduce(self, step: StepConfig, context: dict) -> dict:
        """Execute a map-reduce step.

        Combines fan_out (map) and fan_in (reduce) in a single step.

        Params:
            items: Expression that resolves to a list
            max_concurrent: Maximum concurrent map executions
            map_step: Step configuration for map phase
            reduce_step: Step configuration for reduce phase
            fail_fast: If True, abort map phase on first failure

        Returns:
            Dict with final reduced result
        """
        # Resolve items
        items_expr = step.params.get("items", [])
        if isinstance(items_expr, str) and items_expr.startswith("${") and items_expr.endswith("}"):
            var_name = items_expr[2:-1]
            items = context.get(var_name, [])
        else:
            items = items_expr

        if not isinstance(items, list):
            raise ValueError(f"map_reduce items must be a list, got {type(items)}")

        map_step_config = step.params.get("map_step", {})
        reduce_step_config = step.params.get("reduce_step", {})

        if not map_step_config:
            raise ValueError("map_reduce requires map_step parameter")
        if not reduce_step_config:
            raise ValueError("map_reduce requires reduce_step parameter")

        # Execute map phase using fan_out
        fan_out_step = StepConfig(
            id=f"{step.id}_map",
            type="fan_out",
            params={
                "items": items,
                "max_concurrent": step.params.get("max_concurrent", 5),
                "step_template": map_step_config,
                "fail_fast": step.params.get("fail_fast", False),
            },
            timeout_seconds=step.timeout_seconds,
        )

        map_result = self._execute_fan_out(fan_out_step, context)

        if map_result["failed"] > 0 and step.params.get("fail_fast", False):
            return {
                "response": None,
                "map_results": map_result["results"],
                "map_errors": map_result.get("errors", []),
                "phase": "map_failed",
                "tokens_used": map_result["tokens_used"],
            }

        # Execute reduce phase using fan_in
        # Put map results into context for reduce
        reduce_context = context.copy()
        reduce_context["map_results"] = map_result["results"]

        # Substitute ${map_results} in reduce prompt
        reduce_params = reduce_step_config.get("params", {}).copy()
        if "prompt" in reduce_params:
            items_text = "\n---\n".join(str(r) for r in map_result["results"])
            reduce_params["prompt"] = reduce_params["prompt"].replace("${map_results}", items_text)

        fan_in_step = StepConfig(
            id=f"{step.id}_reduce",
            type="fan_in",
            params={
                "input": map_result["results"],
                "aggregation": reduce_step_config.get("type", "claude_code"),
                "aggregate_prompt": reduce_params.get("prompt", ""),
                "role": reduce_params.get("role", "analyst"),
                "model": reduce_params.get("model"),
            },
            timeout_seconds=step.timeout_seconds,
        )

        reduce_result = self._execute_fan_in(fan_in_step, reduce_context)

        return {
            "response": reduce_result.get("response", ""),
            "map_results": map_result["results"],
            "map_successful": map_result["successful"],
            "map_failed": map_result["failed"],
            "tokens_used": map_result["tokens_used"] + reduce_result.get("tokens_used", 0),
            "duration_ms": map_result.get("duration_ms", 0),
        }
