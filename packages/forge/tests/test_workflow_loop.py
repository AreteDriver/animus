"""Tests for workflow loop step handler and approval gate resume."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from animus_forge.workflow.executor_core import WorkflowExecutor
from animus_forge.workflow.executor_loop import LoopHandlerMixin
from animus_forge.workflow.loader import StepConfig, WorkflowConfig


def _make_executor(**kwargs):
    """Create a WorkflowExecutor with mocks."""
    return WorkflowExecutor(**kwargs)


def _make_loop_step(step_id="loop_1", **params):
    """Create a loop StepConfig."""
    return StepConfig(id=step_id, type="loop", params=params, outputs=["results"])


def _shell_step_dict(step_id="body_step", prompt="echo hello"):
    """Create a shell sub-step dict for loop body."""
    return {
        "id": step_id,
        "type": "shell",
        "params": {"command": prompt},
        "outputs": ["stdout"],
    }


# ---------------------------------------------------------------------------
# Loop handler — basic iteration
# ---------------------------------------------------------------------------


class TestLoopBasic:
    def test_empty_steps_returns_zero(self):
        ex = _make_executor()
        step = _make_loop_step(steps=[])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 0
        assert result["break_reason"] == "no_steps"

    def test_counted_loop(self):
        """Loop runs max_iterations times with body steps."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 5}

        step = _make_loop_step(
            max_iterations=3,
            steps=[_shell_step_dict()],
        )
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 3
        assert result["tokens_used"] == 15
        assert result["break_reason"] == "max_iterations"

    def test_loop_sets_iteration_variable(self):
        """Each iteration gets _loop_iteration in context."""
        ex = _make_executor()
        seen_iterations = []

        def capture_handler(s, ctx):
            seen_iterations.append(ctx["_loop_iteration"])
            return {"stdout": "ok", "tokens_used": 0}

        ex._handlers["shell"] = capture_handler
        step = _make_loop_step(max_iterations=3, steps=[_shell_step_dict()])
        ex._execute_loop(step, {})
        assert seen_iterations == [0, 1, 2]

    def test_loop_cleans_up_variables(self):
        """Loop variables are removed from context after completion."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 0}
        context = {}
        step = _make_loop_step(max_iterations=1, steps=[_shell_step_dict()])
        ex._execute_loop(step, context)
        assert "_loop_iteration" not in context
        assert "_loop_index" not in context


# ---------------------------------------------------------------------------
# Loop handler — for-each mode
# ---------------------------------------------------------------------------


class TestLoopForEach:
    def test_foreach_items_list(self):
        """Loop iterates over literal items list."""
        ex = _make_executor()
        items_seen = []

        def capture_handler(s, ctx):
            items_seen.append(ctx.get("item"))
            return {"stdout": str(ctx.get("item")), "tokens_used": 1}

        ex._handlers["shell"] = capture_handler
        step = _make_loop_step(
            items=["a", "b", "c"],
            steps=[_shell_step_dict()],
        )
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 3
        assert items_seen == ["a", "b", "c"]
        assert result["break_reason"] == "items_exhausted"

    def test_foreach_items_from_context(self):
        """Loop resolves ${var} items from context."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 0}
        context = {"files": ["x.py", "y.py"]}
        step = _make_loop_step(
            items="${files}",
            steps=[_shell_step_dict()],
        )
        result = ex._execute_loop(step, context)
        assert result["iterations"] == 2

    def test_foreach_empty_items(self):
        """Empty items list returns immediately."""
        ex = _make_executor()
        step = _make_loop_step(items=[], steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 0
        assert result["break_reason"] == "no_items"

    def test_foreach_missing_context_var(self):
        """Missing context variable resolves to empty list."""
        ex = _make_executor()
        step = _make_loop_step(items="${nonexistent}", steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 0
        assert result["break_reason"] == "no_items"

    def test_custom_item_variable(self):
        """item_variable param changes the context key name."""
        ex = _make_executor()
        seen = []

        def capture_handler(s, ctx):
            seen.append(ctx.get("file"))
            return {"stdout": "ok", "tokens_used": 0}

        ex._handlers["shell"] = capture_handler
        step = _make_loop_step(
            items=["a.py", "b.py"],
            item_variable="file",
            steps=[_shell_step_dict()],
        )
        ex._execute_loop(step, {})
        assert seen == ["a.py", "b.py"]

    def test_max_iterations_caps_items(self):
        """max_iterations limits for-each even when items remain."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 0}
        step = _make_loop_step(
            max_iterations=2,
            items=["a", "b", "c", "d"],
            steps=[_shell_step_dict()],
        )
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 2


# ---------------------------------------------------------------------------
# Loop handler — break conditions
# ---------------------------------------------------------------------------


class TestLoopBreakCondition:
    def test_break_on_eq(self):
        """Loop breaks when field equals value."""
        ex = _make_executor()
        call_count = 0

        def handler(s, ctx):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                ctx["quality"] = "good"
            return {"stdout": "ok", "tokens_used": 0, "quality": ctx.get("quality", "bad")}

        ex._handlers["shell"] = handler
        step = _make_loop_step(
            max_iterations=10,
            steps=[
                {
                    "id": "check",
                    "type": "shell",
                    "params": {"command": "check"},
                    "outputs": ["quality"],
                }
            ],
            break_condition={"field": "quality", "operator": "eq", "value": "good"},
        )
        context = {}
        result = ex._execute_loop(step, context)
        assert result["break_reason"] == "break_condition"
        assert result["iterations"] == 2

    def test_break_on_gt(self):
        """Loop breaks when field > value."""
        ex = _make_executor()
        iteration = [0]

        def handler(s, ctx):
            iteration[0] += 1
            score = iteration[0] * 30
            return {"stdout": "ok", "tokens_used": 0, "score": score}

        ex._handlers["shell"] = handler
        step = _make_loop_step(
            max_iterations=10,
            steps=[
                {
                    "id": "score",
                    "type": "shell",
                    "params": {"command": "score"},
                    "outputs": ["score"],
                }
            ],
            break_condition={"field": "score", "operator": "gt", "value": 80},
        )
        result = ex._execute_loop(step, {})
        assert result["break_reason"] == "break_condition"
        assert result["iterations"] == 3  # 30, 60, 90 -> breaks after 90 > 80

    def test_break_on_contains(self):
        """Loop breaks when field contains value."""
        ex = _make_executor()

        def handler(s, ctx):
            return {"stdout": "ok", "tokens_used": 0, "status": "PASS: all tests"}

        ex._handlers["shell"] = handler
        step = _make_loop_step(
            max_iterations=5,
            steps=[
                {
                    "id": "test",
                    "type": "shell",
                    "params": {"command": "test"},
                    "outputs": ["status"],
                }
            ],
            break_condition={"field": "status", "operator": "contains", "value": "PASS"},
        )
        result = ex._execute_loop(step, {})
        assert result["break_reason"] == "break_condition"
        assert result["iterations"] == 1

    def test_break_condition_missing_field(self):
        """Break condition with missing field doesn't trigger."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 0}
        step = _make_loop_step(
            max_iterations=2,
            steps=[_shell_step_dict()],
            break_condition={"field": "nonexistent", "operator": "eq", "value": "x"},
        )
        result = ex._execute_loop(step, {})
        assert result["break_reason"] == "max_iterations"
        assert result["iterations"] == 2


# ---------------------------------------------------------------------------
# Loop handler — error handling
# ---------------------------------------------------------------------------


class TestLoopErrors:
    def test_unknown_handler_stops_loop(self):
        """Unknown step type in body stops the loop."""
        ex = _make_executor()
        step = _make_loop_step(
            max_iterations=3,
            steps=[{"id": "bad", "type": "nonexistent_type", "params": {}}],
        )
        # nonexistent_type won't be in _handlers
        # Need to remove it if somehow present
        ex._handlers.pop("nonexistent_type", None)
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 1
        assert result["break_reason"] == "step_failed"
        assert result["results"][0]["failed"] is True

    def test_handler_exception_stops_loop(self):
        """Exception in handler stops the loop."""
        ex = _make_executor()

        def failing_handler(s, ctx):
            raise RuntimeError("boom")

        ex._handlers["shell"] = failing_handler
        step = _make_loop_step(max_iterations=5, steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 1
        assert result["break_reason"] == "step_failed"

    def test_budget_exceeded_stops_loop(self):
        """Budget exhaustion stops the loop."""
        mock_budget = MagicMock()
        mock_budget.can_allocate.return_value = False
        ex = _make_executor(budget_manager=mock_budget)
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 10}

        step = _make_loop_step(max_iterations=5, steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 1
        assert result["break_reason"] == "step_failed"
        assert result["results"][0]["failed"] is True


# ---------------------------------------------------------------------------
# Loop handler — checkpoints and budget
# ---------------------------------------------------------------------------


class TestLoopCheckpointBudget:
    def test_checkpoint_per_iteration(self):
        """Checkpoint manager is called each iteration."""
        mock_cp = MagicMock()
        ex = _make_executor(checkpoint_manager=mock_cp)
        ex._current_workflow_id = "wf-123"
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 5}

        step = _make_loop_step(max_iterations=3, steps=[_shell_step_dict()])
        ex._execute_loop(step, {})
        assert mock_cp.checkpoint_now.call_count == 3

    def test_budget_recorded_per_iteration(self):
        """Budget manager records usage each iteration."""
        mock_budget = MagicMock()
        mock_budget.can_allocate.return_value = True
        ex = _make_executor(budget_manager=mock_budget)
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 10}

        step = _make_loop_step(max_iterations=2, steps=[_shell_step_dict()])
        ex._execute_loop(step, {})
        assert mock_budget.record_usage.call_count == 2


# ---------------------------------------------------------------------------
# Loop handler — output accumulation
# ---------------------------------------------------------------------------


class TestLoopOutputs:
    def test_outputs_accumulate_across_iterations(self):
        """Sub-step outputs are available in subsequent iterations."""
        ex = _make_executor()
        iteration_values = []

        def handler(s, ctx):
            prev = ctx.get("counter", 0)
            new_val = prev + 1
            iteration_values.append(new_val)
            return {"stdout": str(new_val), "tokens_used": 0, "counter": new_val}

        ex._handlers["shell"] = handler
        step = _make_loop_step(
            max_iterations=3,
            steps=[
                {
                    "id": "inc",
                    "type": "shell",
                    "params": {"command": "inc"},
                    "outputs": ["counter"],
                }
            ],
        )
        ex._execute_loop(step, {})
        assert iteration_values == [1, 2, 3]

    def test_result_contains_per_iteration_data(self):
        """Each iteration result has outputs, tokens, duration."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 7}

        step = _make_loop_step(max_iterations=2, steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})

        assert len(result["results"]) == 2
        for r in result["results"]:
            assert "iteration" in r
            assert "tokens_used" in r
            assert "duration_ms" in r
            assert r["tokens_used"] == 7


# ---------------------------------------------------------------------------
# Loop handler — edge cases for coverage
# ---------------------------------------------------------------------------


class TestLoopEdgeCases:
    def test_items_non_string_non_list_resolves_empty(self):
        """Non-string, non-list items_expr resolves to empty list."""
        ex = _make_executor()
        step = _make_loop_step(items=42, steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 0
        assert result["break_reason"] == "no_items"

    def test_checkpoint_exception_swallowed(self):
        """Checkpoint failure doesn't break the loop."""
        mock_cp = MagicMock()
        mock_cp.checkpoint_now.side_effect = RuntimeError("db error")
        ex = _make_executor(checkpoint_manager=mock_cp)
        ex._current_workflow_id = "wf-123"
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "ok", "tokens_used": 0}

        step = _make_loop_step(max_iterations=2, steps=[_shell_step_dict()])
        result = ex._execute_loop(step, {})
        assert result["iterations"] == 2  # Loop continued despite checkpoint errors

    def test_output_response_fallback(self):
        """When output key not in output, falls back to 'response' key."""
        ex = _make_executor()

        def handler(s, ctx):
            return {"response": "ai_response", "tokens_used": 0}

        ex._handlers["shell"] = handler
        step = _make_loop_step(
            max_iterations=1,
            steps=[
                {
                    "id": "ai_step",
                    "type": "shell",
                    "params": {},
                    "outputs": ["analysis"],
                }
            ],
        )
        context = {}
        ex._execute_loop(step, context)
        assert context["analysis"] == "ai_response"

    def test_output_stdout_fallback(self):
        """When output key and 'response' not in output, falls back to 'stdout'."""
        ex = _make_executor()

        def handler(s, ctx):
            return {"stdout": "shell_output", "tokens_used": 0}

        ex._handlers["shell"] = handler
        step = _make_loop_step(
            max_iterations=1,
            steps=[
                {
                    "id": "sh",
                    "type": "shell",
                    "params": {},
                    "outputs": ["result"],
                }
            ],
        )
        context = {}
        ex._execute_loop(step, context)
        assert context["result"] == "shell_output"


# ---------------------------------------------------------------------------
# Break condition evaluator
# ---------------------------------------------------------------------------


class TestBreakConditionEvaluator:
    def _make_mixin(self):
        mixin = LoopHandlerMixin()
        return mixin

    def test_eq(self):
        m = self._make_mixin()
        assert m._check_break_condition({"field": "x", "operator": "eq", "value": 5}, {"x": 5})
        assert not m._check_break_condition({"field": "x", "operator": "eq", "value": 5}, {"x": 3})

    def test_gt(self):
        m = self._make_mixin()
        assert m._check_break_condition({"field": "x", "operator": "gt", "value": 5}, {"x": 10})
        assert not m._check_break_condition({"field": "x", "operator": "gt", "value": 5}, {"x": 3})

    def test_lt(self):
        m = self._make_mixin()
        assert m._check_break_condition({"field": "x", "operator": "lt", "value": 5}, {"x": 3})

    def test_gte(self):
        m = self._make_mixin()
        assert m._check_break_condition({"field": "x", "operator": "gte", "value": 5}, {"x": 5})
        assert m._check_break_condition({"field": "x", "operator": "gte", "value": 5}, {"x": 6})

    def test_lte(self):
        m = self._make_mixin()
        assert m._check_break_condition({"field": "x", "operator": "lte", "value": 5}, {"x": 5})
        assert m._check_break_condition({"field": "x", "operator": "lte", "value": 5}, {"x": 4})

    def test_contains(self):
        m = self._make_mixin()
        assert m._check_break_condition(
            {"field": "x", "operator": "contains", "value": "PASS"},
            {"x": "PASS: all good"},
        )

    def test_unknown_operator(self):
        m = self._make_mixin()
        assert not m._check_break_condition(
            {"field": "x", "operator": "regex", "value": ".*"},
            {"x": "test"},
        )

    def test_missing_field(self):
        m = self._make_mixin()
        assert not m._check_break_condition(
            {"field": "missing", "operator": "eq", "value": 1},
            {},
        )

    def test_invalid_numeric_comparison(self):
        m = self._make_mixin()
        assert not m._check_break_condition(
            {"field": "x", "operator": "gt", "value": 5},
            {"x": "not_a_number"},
        )


# ---------------------------------------------------------------------------
# Integration: loop registered in executor
# ---------------------------------------------------------------------------


class TestLoopRegistered:
    def test_loop_handler_registered(self):
        """Loop handler is registered in WorkflowExecutor._handlers."""
        ex = _make_executor()
        assert "loop" in ex._handlers
        assert ex._handlers["loop"] == ex._execute_loop

    def test_loop_in_workflow(self):
        """Loop step works in a full workflow execution."""
        ex = _make_executor()
        ex._handlers["shell"] = lambda s, ctx: {"stdout": "done", "tokens_used": 2}

        workflow = WorkflowConfig(
            name="test_loop_workflow",
            version="1.0",
            description="test",
            steps=[
                StepConfig(
                    id="loop_step",
                    type="loop",
                    params={
                        "max_iterations": 2,
                        "steps": [_shell_step_dict()],
                    },
                    outputs=["results"],
                ),
            ],
        )
        with patch(
            "animus_forge.workflow.executor_core.WorkflowExecutor._execute_step"
        ) as mock_step:
            from animus_forge.workflow.executor_results import StepResult, StepStatus

            mock_step.return_value = StepResult(
                step_id="loop_step",
                status=StepStatus.SUCCESS,
                output={
                    "iterations": 2,
                    "results": [],
                    "tokens_used": 4,
                    "break_reason": "max_iterations",
                },
            )
            result = ex.execute(workflow, inputs={})
            assert result.status == "success"


# ---------------------------------------------------------------------------
# Approval gate resume verification
# ---------------------------------------------------------------------------


class TestApprovalGateResume:
    def test_resume_from_skips_completed_steps(self):
        """resume_from parameter skips steps before the resume point."""
        ex = _make_executor()
        call_log = []

        # Track which steps actually execute
        def tracking_step(step, workflow_id=None):
            call_log.append(step.id)
            from animus_forge.workflow.executor_results import StepResult, StepStatus

            return StepResult(
                step_id=step.id,
                status=StepStatus.SUCCESS,
                output={"stdout": "ok"},
            )

        ex._execute_step = tracking_step

        workflow = WorkflowConfig(
            name="resume_test",
            version="1.0",
            description="test",
            steps=[
                StepConfig(id="step_1", type="shell", params={"command": "echo 1"}),
                StepConfig(id="step_2", type="shell", params={"command": "echo 2"}),
                StepConfig(id="step_3", type="shell", params={"command": "echo 3"}),
            ],
        )
        ex.execute(workflow, inputs={}, resume_from="step_2")
        # Should skip step_1, execute step_2 and step_3
        assert "step_1" not in call_log
        assert "step_2" in call_log
        assert "step_3" in call_log

    def test_approval_halt_sets_awaiting_status(self):
        """Approval gate sets result status to awaiting_approval."""
        ex = _make_executor()

        def approval_handler(step, workflow_id=None):
            from animus_forge.workflow.executor_results import StepResult, StepStatus

            return StepResult(
                step_id=step.id,
                status=StepStatus.SUCCESS,
                output={
                    "status": "awaiting_approval",
                    "token": "tok-123",
                    "prompt": "Approve?",
                    "preview": {},
                },
            )

        ex._execute_step = approval_handler

        workflow = WorkflowConfig(
            name="approval_test",
            version="1.0",
            description="test",
            steps=[
                StepConfig(id="approve_deploy", type="approval", params={"prompt": "Deploy?"}),
                StepConfig(id="deploy", type="shell", params={"command": "deploy"}),
            ],
        )

        with patch("animus_forge.workflow.executor_core.get_approval_store", create=True):
            result = ex.execute(workflow, inputs={})

        assert result.status == "awaiting_approval"
        assert result.outputs.get("__approval_token") == "tok-123"
