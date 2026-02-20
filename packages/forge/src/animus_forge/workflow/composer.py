"""Workflow Composability System.

Enables workflows to call other workflows as sub-steps, building a library
of composable building blocks. For example, a ``review-and-fix`` workflow
can invoke ``run-tests`` which itself invokes ``lint-check``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from animus_forge.workflow.executor import ExecutionResult, WorkflowExecutor
from animus_forge.workflow.loader import StepConfig, load_workflow

logger = logging.getLogger(__name__)

_DEFAULT_MAX_DEPTH = 5


@dataclass
class SubWorkflowResult:
    """Result of executing a sub-workflow step.

    Attributes:
        workflow_name: Name of the sub-workflow that was executed.
        status: Outcome status, either ``"success"`` or ``"failed"``.
        outputs: Output values produced by the sub-workflow.
        steps_executed: Number of steps that ran to completion.
        total_tokens: Aggregate token usage across all sub-workflow steps.
        total_duration_ms: Wall-clock duration of the sub-workflow in milliseconds.
        depth: Nesting depth at which this sub-workflow executed.
    """

    workflow_name: str
    status: str  # "success" or "failed"
    outputs: dict = field(default_factory=dict)
    steps_executed: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    depth: int = 0


class WorkflowComposer:
    """Enables workflow composition by allowing workflows to invoke sub-workflows.

    Sub-workflows are referenced via a ``sub_workflow`` step type in YAML
    definitions. The composer handles context propagation, depth limiting,
    and circular-reference detection.

    Args:
        max_depth: Maximum allowed nesting depth for sub-workflows.
            Prevents infinite recursion when workflows reference each other.

    Example YAML step::

        steps:
          - id: run-tests
            type: sub_workflow
            params:
              workflow: run-tests-v2
              inputs:
                code: ${builder_output}
              pass_context: true
            outputs:
              - test_results
    """

    def __init__(self, max_depth: int = _DEFAULT_MAX_DEPTH) -> None:
        self.max_depth = max_depth

    def execute_sub_workflow(
        self,
        step: StepConfig,
        parent_context: dict,
        *,
        depth: int = 1,
        parent_executor: WorkflowExecutor | None = None,
    ) -> dict:
        """Execute a sub-workflow step and return its outputs.

        This is the handler registered with ``WorkflowExecutor`` for the
        ``sub_workflow`` step type.

        Args:
            step: The step configuration containing ``params.workflow`` and
                optionally ``params.inputs`` and ``params.pass_context``.
            parent_context: The parent workflow's execution context.
            depth: Current nesting depth (1-indexed). Callers should not
                set this directly; it is managed internally.
            parent_executor: The parent ``WorkflowExecutor`` whose managers
                (checkpoint, budget, feedback) will be inherited by the
                child executor.

        Returns:
            A dictionary of outputs produced by the sub-workflow, keyed
            by output name. Also includes a ``_sub_workflow_result`` key
            with the full ``SubWorkflowResult``.

        Raises:
            RecursionError: If ``depth`` exceeds ``max_depth``.
            FileNotFoundError: If the referenced workflow YAML cannot be found.
            ValueError: If the step is missing required ``workflow`` param.
        """
        # --- Validate params ---
        workflow_name: str | None = step.params.get("workflow")
        if not workflow_name:
            raise ValueError(
                f"Step '{step.id}' is type 'sub_workflow' but missing required param 'workflow'"
            )

        # --- Depth check ---
        if depth > self.max_depth:
            raise RecursionError(
                f"Sub-workflow depth {depth} exceeds maximum of {self.max_depth}. "
                f"Workflow '{workflow_name}' at step '{step.id}' would exceed the limit."
            )

        logger.info(
            "Executing sub-workflow '%s' at depth %d (step '%s')",
            workflow_name,
            depth,
            step.id,
        )

        # --- Load the sub-workflow ---
        sub_workflow = load_workflow(workflow_name)

        # --- Build child context ---
        pass_context: bool = step.params.get("pass_context", False)
        child_inputs: dict[str, Any] = {}
        if pass_context:
            child_inputs.update(parent_context)

        # Map explicit input params, resolving ${var} references from parent
        explicit_inputs: dict[str, Any] = step.params.get("inputs", {})
        for key, value in explicit_inputs.items():
            child_inputs[key] = _resolve_value(value, parent_context)

        # --- Create child executor inheriting parent managers ---
        child_executor = WorkflowExecutor(
            checkpoint_manager=(parent_executor.checkpoint_manager if parent_executor else None),
            budget_manager=(parent_executor.budget_manager if parent_executor else None),
            feedback_engine=(parent_executor.feedback_engine if parent_executor else None),
            dry_run=parent_executor.dry_run if parent_executor else False,
        )

        # Register sub_workflow handler on child so nested sub-workflows work
        composer_for_child = self

        def child_sub_workflow_handler(child_step: StepConfig, ctx: dict) -> dict:
            return composer_for_child.execute_sub_workflow(
                child_step,
                ctx,
                depth=depth + 1,
                parent_executor=child_executor,
            )

        child_executor.register_handler("sub_workflow", child_sub_workflow_handler)

        # --- Execute ---
        start_ms = _now_ms()
        execution_result: ExecutionResult = child_executor.execute(
            sub_workflow, inputs=child_inputs
        )
        duration_ms = _now_ms() - start_ms

        # --- Build result ---
        steps_executed = sum(1 for s in execution_result.steps if s.status.value == "success")

        sub_result = SubWorkflowResult(
            workflow_name=sub_workflow.name,
            status=execution_result.status,
            outputs=execution_result.outputs,
            steps_executed=steps_executed,
            total_tokens=execution_result.total_tokens,
            total_duration_ms=duration_ms,
            depth=depth,
        )

        if execution_result.status == "failed":
            logger.error(
                "Sub-workflow '%s' failed at depth %d: %s",
                workflow_name,
                depth,
                execution_result.error,
            )
        else:
            logger.info(
                "Sub-workflow '%s' completed at depth %d (%d steps, %d tokens, %dms)",
                workflow_name,
                depth,
                steps_executed,
                execution_result.total_tokens,
                duration_ms,
            )

        # Merge outputs into a dict suitable for the parent context
        outputs: dict[str, Any] = dict(execution_result.outputs)
        outputs["_sub_workflow_result"] = sub_result
        return outputs

    def resolve_workflow_graph(self, workflow_name: str) -> list[str]:
        """Return the full dependency tree of workflows for a given root.

        Traverses all ``sub_workflow`` steps recursively, building a flat
        list of workflow names in depth-first order. Detects and raises
        on circular references.

        Args:
            workflow_name: The root workflow name (path) to start from.

        Returns:
            Ordered list of all workflow names reachable from the root,
            including the root itself.

        Raises:
            ValueError: If a circular reference is detected.
            FileNotFoundError: If a referenced workflow cannot be loaded.
        """
        result: list[str] = []
        visited: set[str] = set()
        ancestors: set[str] = set()  # current recursion stack for cycle detection

        self._walk_graph(workflow_name, result, visited, ancestors)
        return result

    def _walk_graph(
        self,
        workflow_name: str,
        result: list[str],
        visited: set[str],
        ancestors: set[str],
    ) -> None:
        """Depth-first walk of the workflow dependency graph.

        Args:
            workflow_name: Current workflow to inspect.
            result: Accumulator list for the traversal order.
            visited: Set of already-visited workflow names (avoids dupes).
            ancestors: Set of workflow names on the current recursion path
                (detects cycles).

        Raises:
            ValueError: On circular reference detection.
        """
        if workflow_name in ancestors:
            cycle_path = " -> ".join([*ancestors, workflow_name])
            raise ValueError(f"Circular workflow reference detected: {cycle_path}")

        if workflow_name in visited:
            return

        ancestors.add(workflow_name)
        visited.add(workflow_name)
        result.append(workflow_name)

        try:
            config = load_workflow(workflow_name)
        except FileNotFoundError:
            logger.warning(
                "Could not load workflow '%s' during graph resolution; "
                "it may be defined externally.",
                workflow_name,
            )
            ancestors.discard(workflow_name)
            return

        for step in config.steps:
            if step.type == "sub_workflow":
                child_name = step.params.get("workflow")
                if child_name:
                    self._walk_graph(child_name, result, visited, ancestors)

        ancestors.discard(workflow_name)

    def register_with_executor(self, executor: WorkflowExecutor) -> None:
        """Register the ``sub_workflow`` step handler with an executor.

        After registration the executor can process steps with
        ``type: sub_workflow`` transparently.

        Args:
            executor: The ``WorkflowExecutor`` to extend.
        """
        composer = self

        def sub_workflow_handler(step: StepConfig, context: dict) -> dict:
            return composer.execute_sub_workflow(
                step,
                context,
                depth=1,
                parent_executor=executor,
            )

        executor.register_handler("sub_workflow", sub_workflow_handler)
        logger.info(
            "Registered sub_workflow handler with executor (max_depth=%d)",
            self.max_depth,
        )


def _resolve_value(value: Any, context: dict) -> Any:
    """Resolve a value that may contain ``${var}`` references.

    Only string values with the exact pattern ``${key}`` are resolved.
    Other types (int, list, dict, etc.) are returned as-is.

    Args:
        value: The value to resolve.
        context: The context dictionary to look up references in.

    Returns:
        The resolved value, or the original if no substitution applies.
    """
    if not isinstance(value, str):
        return value

    # Exact match: entire value is a single reference
    if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
        key = value[2:-1]
        return context.get(key, value)

    # Inline substitution for strings with embedded references
    import re

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        resolved = context.get(key)
        return str(resolved) if resolved is not None else match.group(0)

    return re.sub(r"\$\{(\w+)}", _replace, value)


def _now_ms() -> int:
    """Return current monotonic time in milliseconds."""
    return int(time.monotonic() * 1000)
