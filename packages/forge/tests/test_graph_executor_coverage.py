"""Targeted coverage for graph_executor.py gaps on origin/main.

Closes blocks flagged by CI on 2c8cde2:
- Lines 486-512: `_execute_parallel` body (run_step + gather + combine)
- Lines 545-546: condition attachment in `_execute_step`
- Line 569: token count propagation in `_execute_step`
- Lines 610-615: `resume` happy path after execution_manager checks pass
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "src")

from animus_forge.workflow.graph_executor import ReactFlowExecutor
from animus_forge.workflow.graph_models import GraphNode, WorkflowGraph


class TestExecuteParallelBody:
    """Covers lines 486-512 — the run_step closure, gather, and combine loop."""

    @pytest.mark.asyncio
    async def test_parallel_steps_combine_outputs(self, monkeypatch):
        executor = ReactFlowExecutor()

        # Stub _execute_step so we don't descend into real step execution.
        async def fake_execute_step(self, node, context):
            return {f"out_{node.id}": True}

        monkeypatch.setattr(
            ReactFlowExecutor, "_execute_step", fake_execute_step, raising=True
        )

        node = GraphNode(
            id="parallel-1",
            type="parallel",
            data={
                "steps": [
                    {"id": "a", "type": "agent"},
                    {"id": "b", "type": "agent"},
                ]
            },
        )

        combined = await executor._execute_parallel(node, {}, "exec-1")

        # Both branches contributed to the combined dict.
        assert "out_parallel-1-a" in combined
        assert "out_parallel-1-b" in combined

    @pytest.mark.asyncio
    async def test_parallel_exception_is_logged_and_skipped(self, monkeypatch):
        """Lines 506-508: an Exception result from gather is logged + skipped,
        non-exception dict results still contribute."""
        executor = ReactFlowExecutor()

        call_count = {"n": 0}

        async def fake_execute_step(self, node, context):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("branch boom")
            return {"ok": 1}

        monkeypatch.setattr(
            ReactFlowExecutor, "_execute_step", fake_execute_step, raising=True
        )

        node = GraphNode(
            id="par",
            type="parallel",
            data={
                "steps": [
                    {"id": "bad", "type": "agent"},
                    {"id": "good", "type": "agent"},
                ]
            },
        )
        combined = await executor._execute_parallel(node, {}, "exec-1")
        # Good branch still contributes; bad branch silently dropped.
        assert combined == {"ok": 1}

    @pytest.mark.asyncio
    async def test_parallel_non_dict_result_is_skipped(self, monkeypatch):
        """Line 509-510: `if isinstance(result, dict): combined.update(result)`
        — non-dict, non-exception results fall through."""
        executor = ReactFlowExecutor()

        async def fake_execute_step(self, node, context):
            # Non-dict, non-exception result — line 509 gates out.
            return "string-result"

        monkeypatch.setattr(
            ReactFlowExecutor, "_execute_step", fake_execute_step, raising=True
        )

        node = GraphNode(
            id="par",
            type="parallel",
            data={"steps": [{"id": "x", "type": "agent"}]},
        )
        combined = await executor._execute_parallel(node, {}, "exec-1")
        assert combined == {}


class TestExecuteStepConditionAndTokens:
    """Covers lines 545-546 (ConditionConfig attach) + 569 (token count)."""

    @pytest.mark.asyncio
    async def test_step_condition_is_attached(self, monkeypatch):
        executor = ReactFlowExecutor()

        # Capture the StepConfig passed to the workflow executor.
        captured = {}

        class _FakeResult:
            outputs = {"value": 42}
            total_tokens = 0

        class _FakeExecutor:
            def execute(self, workflow, inputs=None):
                captured["workflow"] = workflow
                captured["inputs"] = inputs
                return _FakeResult()

        monkeypatch.setattr(
            ReactFlowExecutor,
            "_get_workflow_executor",
            lambda self: _FakeExecutor(),
            raising=True,
        )

        node = GraphNode(
            id="step-with-cond",
            type="shell",
            data={
                "params": {"cmd": "echo hi"},
                "timeout": 30,
                "condition": {
                    "field": "input.enabled",
                    "operator": "equals",
                    "value": True,
                },
            },
        )
        outputs = await executor._execute_step(node, {"input": {"enabled": True}})
        assert outputs == {"value": 42}
        # The StepConfig attached to the workflow has the ConditionConfig.
        step = captured["workflow"].steps[0]
        assert step.condition is not None
        assert step.condition.field == "input.enabled"
        assert step.condition.operator == "equals"
        assert step.condition.value is True

    @pytest.mark.asyncio
    async def test_token_count_is_propagated(self, monkeypatch):
        executor = ReactFlowExecutor()

        class _FakeResult:
            outputs = {"response": "ok"}
            total_tokens = 1337

        class _FakeExecutor:
            def execute(self, workflow, inputs=None):
                return _FakeResult()

        monkeypatch.setattr(
            ReactFlowExecutor,
            "_get_workflow_executor",
            lambda self: _FakeExecutor(),
            raising=True,
        )

        node = GraphNode(id="s1", type="shell", data={"params": {}})
        outputs = await executor._execute_step(node, {})
        assert outputs["_tokens"] == 1337
        assert outputs["response"] == "ok"

    @pytest.mark.asyncio
    async def test_agent_node_with_anthropic_provider_maps_to_claude_code(
        self, monkeypatch
    ):
        """Line 533: provider == 'anthropic' → step_type = claude_code."""
        executor = ReactFlowExecutor()
        captured = {}

        class _FakeResult:
            outputs = {}
            total_tokens = 0

        class _FakeExecutor:
            def execute(self, workflow, inputs=None):
                captured["step_type"] = workflow.steps[0].type
                return _FakeResult()

        monkeypatch.setattr(
            ReactFlowExecutor,
            "_get_workflow_executor",
            lambda self: _FakeExecutor(),
            raising=True,
        )

        node = GraphNode(
            id="agent-1",
            type="agent",
            data={"provider": "anthropic", "params": {}},
        )
        await executor._execute_step(node, {})
        assert captured["step_type"] == "claude_code"


class TestResumeHappyPath:
    """Covers lines 610-615 — resume past the manager / execution-exists
    guards all the way through to self.execute(...)."""

    def test_resume_continues_from_stored_variables(self, monkeypatch):
        manager = MagicMock()
        execution = MagicMock()
        execution.variables = {"resumed_key": "resumed_value"}
        manager.get_execution.return_value = execution
        manager.resume_execution = MagicMock()

        executor = ReactFlowExecutor(execution_manager=manager)

        # Stub the recursive execute(...) so resume hands off cleanly.
        captured = {}

        def fake_execute(self, graph, variables=None, execution_id=None):
            captured["variables"] = variables
            captured["execution_id"] = execution_id
            return MagicMock(status="resumed")

        monkeypatch.setattr(ReactFlowExecutor, "execute", fake_execute, raising=True)

        graph = WorkflowGraph(
            id="g",
            name="G",
            nodes=[GraphNode(id="start", type="start")],
            edges=[],
        )
        result = executor.resume("exec-42", graph)

        manager.resume_execution.assert_called_once_with("exec-42")
        assert captured["variables"] == {"resumed_key": "resumed_value"}
        assert captured["execution_id"] == "exec-42"
        assert result.status == "resumed"
