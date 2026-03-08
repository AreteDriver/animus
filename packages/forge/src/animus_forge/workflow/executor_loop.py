"""Loop step handler for workflow executor.

Mixin class providing the loop step type. Supports counted loops,
for-each iteration over lists, and condition-based break logic.
"""

from __future__ import annotations

import logging
import time

from .loader import StepConfig

logger = logging.getLogger(__name__)


class LoopHandlerMixin:
    """Mixin providing loop step execution.

    Expects the following attributes from the host class:
    - _handlers: dict[str, StepHandler]
    - _context: dict
    - checkpoint_manager: CheckpointManager | None
    - budget_manager: BudgetManager | None
    - _current_workflow_id: str | None
    """

    def _execute_loop(self, step: StepConfig, context: dict) -> dict:
        """Execute a loop step.

        Iterates over body steps up to max_iterations times, with optional
        for-each item iteration and break conditions.

        Params:
            max_iterations: Maximum loop iterations (default: 10)
            items: Expression resolving to a list for for-each (e.g. "${files}")
            item_variable: Context variable name for current item (default: "item")
            steps: List of sub-step configurations to execute per iteration
            break_condition: Dict with field/operator/value to check after each iteration
                field: Context variable to check (e.g. "quality_score")
                operator: "eq", "gt", "lt", "gte", "lte", "contains"
                value: Value to compare against

        Returns:
            Dict with:
            - iterations: Number of iterations completed
            - results: List of per-iteration results
            - tokens_used: Total tokens across all iterations
            - break_reason: Why the loop stopped ("max_iterations", "break_condition",
              "items_exhausted", "no_items", "step_failed")
        """
        max_iterations = step.params.get("max_iterations", 10)
        sub_steps = step.params.get("steps", [])
        items_expr = step.params.get("items", None)
        item_variable = step.params.get("item_variable", "item")
        break_condition = step.params.get("break_condition", None)

        if not sub_steps:
            return {
                "iterations": 0,
                "results": [],
                "tokens_used": 0,
                "break_reason": "no_steps",
            }

        # Resolve items for for-each mode
        items = None
        if items_expr is not None:
            if (
                isinstance(items_expr, str)
                and items_expr.startswith("${")
                and items_expr.endswith("}")
            ):
                var_name = items_expr[2:-1]
                items = context.get(var_name, [])
            elif isinstance(items_expr, list):
                items = items_expr
            else:
                items = []

            if not items:
                return {
                    "iterations": 0,
                    "results": [],
                    "tokens_used": 0,
                    "break_reason": "no_items",
                }
            # Cap iterations to items length
            max_iterations = min(max_iterations, len(items))

        # Parse sub-steps once
        parsed_steps = [StepConfig.from_dict(s) for s in sub_steps]

        all_results = []
        total_tokens = 0
        break_reason = "max_iterations"

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Set loop variables in context
            context["_loop_iteration"] = iteration
            context["_loop_index"] = iteration
            if items is not None:
                context[item_variable] = items[iteration]

            # Checkpoint at iteration start
            if self.checkpoint_manager and self._current_workflow_id:
                try:
                    self.checkpoint_manager.checkpoint_now(
                        stage=f"{step.id}_iter_{iteration}",
                        status="running",
                        input_data={"iteration": iteration},
                        output_data={},
                        tokens_used=0,
                        duration_ms=0,
                        workflow_id=self._current_workflow_id,
                    )
                except Exception:
                    logger.debug("Loop checkpoint failed", exc_info=True)

            # Execute body steps sequentially
            iter_outputs = {}
            iter_tokens = 0
            iter_failed = False

            for sub_step in parsed_steps:
                # Budget check
                if self.budget_manager:
                    estimated = sub_step.params.get("estimated_tokens", 1000)
                    if not self.budget_manager.can_allocate(estimated):
                        break_reason = "budget_exceeded"
                        iter_failed = True
                        break

                # Substitute template variables in prompt
                step_context = context.copy()
                step_context.update(iter_outputs)

                handler = self._handlers.get(sub_step.type)
                if not handler:
                    logger.warning("Unknown step type in loop body: %s", sub_step.type)
                    iter_failed = True
                    break

                try:
                    output = handler(sub_step, step_context)
                    tokens = output.get("tokens_used", 0)
                    iter_tokens += tokens

                    # Store sub-step outputs
                    for output_key in sub_step.outputs:
                        if output_key in output:
                            iter_outputs[output_key] = output[output_key]
                            context[output_key] = output[output_key]
                        elif "response" in output:
                            iter_outputs[output_key] = output["response"]
                            context[output_key] = output["response"]
                        elif "stdout" in output:
                            iter_outputs[output_key] = output["stdout"]
                            context[output_key] = output["stdout"]
                except Exception as e:
                    logger.warning(
                        "Loop body step %s failed on iteration %d: %s",
                        sub_step.id,
                        iteration,
                        e,
                    )
                    iter_failed = True
                    break

            iter_duration = int((time.time() - iter_start) * 1000)
            total_tokens += iter_tokens

            all_results.append(
                {
                    "iteration": iteration,
                    "outputs": iter_outputs,
                    "tokens_used": iter_tokens,
                    "duration_ms": iter_duration,
                    "item": items[iteration] if items is not None else None,
                    "failed": iter_failed,
                }
            )

            if iter_failed:
                break_reason = "step_failed"
                break

            # Budget tracking
            if self.budget_manager and iter_tokens > 0:
                self.budget_manager.record_usage(f"{step.id}_iter_{iteration}", iter_tokens)

            # Check break condition
            if break_condition and self._check_break_condition(break_condition, context):
                break_reason = "break_condition"
                break

            # For-each: check if we've exhausted items
            if items is not None and iteration >= len(items) - 1:
                break_reason = "items_exhausted"
                break
        else:
            # Loop completed all iterations without breaking
            break_reason = "max_iterations" if items is None else "items_exhausted"

        # Clean up loop variables
        context.pop("_loop_iteration", None)
        context.pop("_loop_index", None)

        return {
            "iterations": len(all_results),
            "results": all_results,
            "tokens_used": total_tokens,
            "break_reason": break_reason,
        }

    def _check_break_condition(self, condition: dict, context: dict) -> bool:
        """Evaluate a break condition against the current context.

        Args:
            condition: Dict with field, operator, value keys.
            context: Current execution context.

        Returns:
            True if the break condition is met.
        """
        field_name = condition.get("field", "")
        operator = condition.get("operator", "eq")
        expected = condition.get("value")

        actual = context.get(field_name)
        if actual is None:
            return False

        try:
            if operator == "eq":
                return actual == expected
            elif operator == "gt":
                return float(actual) > float(expected)
            elif operator == "lt":
                return float(actual) < float(expected)
            elif operator == "gte":
                return float(actual) >= float(expected)
            elif operator == "lte":
                return float(actual) <= float(expected)
            elif operator == "contains":
                return str(expected) in str(actual)
            else:
                logger.warning("Unknown break condition operator: %s", operator)
                return False
        except (ValueError, TypeError):
            return False
