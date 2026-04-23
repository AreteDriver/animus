"""Coverage for executor_step.py fallback branches on 2c8cde2.

Targets:
- Line 219: sync alternate_step fallback returns None when alt step fails
- Line 234: trailing `return None` when no fallback branch matched
- Line 266: async alternate_step fallback returns None on alt step failure
- Lines 282-289: async callback fallback — not registered + trailing None
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "src")

from animus_forge.workflow.executor_results import StepResult, StepStatus
from animus_forge.workflow.executor_step import StepExecutionMixin
from animus_forge.workflow.loader import FallbackConfig, StepConfig


class _HostSync(StepExecutionMixin):
    """Minimal host for the sync fallback path."""

    def __init__(self, fallback_callbacks=None, alt_result_status=StepStatus.FAILED):
        self.fallback_callbacks = fallback_callbacks or {}
        self._context: dict = {}
        self.contract_validator = None
        self.checkpoint_manager = None
        self._handlers: dict = {}
        self._alt_result_status = alt_result_status

    # Stub _execute_step so _execute_fallback alternate_step call doesn't
    # recurse into real step execution.
    def _execute_step(self, alt_step, workflow_id=None):  # type: ignore[override]
        return StepResult(
            step_id=alt_step.id,
            status=self._alt_result_status,
            output={"ok": True} if self._alt_result_status == StepStatus.SUCCESS else None,
        )


class _HostAsync(StepExecutionMixin):
    """Minimal host for the async fallback path."""

    def __init__(self, fallback_callbacks=None, alt_result_status=StepStatus.FAILED):
        self.fallback_callbacks = fallback_callbacks or {}
        self._context: dict = {}
        self.contract_validator = None
        self.checkpoint_manager = None
        self._handlers: dict = {}
        self._alt_result_status = alt_result_status

    async def _execute_step_async(self, alt_step, workflow_id=None):  # type: ignore[override]
        return StepResult(
            step_id=alt_step.id,
            status=self._alt_result_status,
            output={"ok": True} if self._alt_result_status == StepStatus.SUCCESS else None,
        )


# ---------------------------------------------------------------------------
# Sync fallback
# ---------------------------------------------------------------------------


class TestSyncFallbackAlternateStepFails:
    """Line 219: alt step returns non-SUCCESS → fallback returns None."""

    def test_alt_step_failure_returns_none(self):
        host = _HostSync(alt_result_status=StepStatus.FAILED)
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(
                type="alternate_step",
                step={"id": "alt", "type": "shell", "params": {}},
            ),
        )
        assert host._execute_fallback(step, "boom", "wf-1") is None

    def test_alt_step_success_returns_output(self):
        host = _HostSync(alt_result_status=StepStatus.SUCCESS)
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(
                type="alternate_step",
                step={"id": "alt", "type": "shell", "params": {}},
            ),
        )
        assert host._execute_fallback(step, "boom", "wf-1") == {"ok": True}


class TestSyncFallbackCallbackNotRegistered:
    """Callback fallback where the named callback isn't registered → None."""

    def test_callback_not_registered_returns_none(self):
        host = _HostSync(fallback_callbacks={})  # empty registry
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(type="callback", callback="nope"),
        )
        assert host._execute_fallback(step, "boom", "wf-1") is None


class TestSyncFallbackUnknownType:
    """Line 234: trailing `return None` when no branch matched.

    `alternate_step` with step=None falls through every elif so the final
    `return None` at line 234 fires.
    """

    def test_alternate_step_without_step_falls_through(self):
        host = _HostSync()
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(type="alternate_step", step=None),
        )
        assert host._execute_fallback(step, "boom", "wf-1") is None


# ---------------------------------------------------------------------------
# Async fallback
# ---------------------------------------------------------------------------


class TestAsyncFallbackAlternateStepFails:
    """Line 266: async alt-step fallback returns None on non-success."""

    @pytest.mark.asyncio
    async def test_alt_step_failure_returns_none(self):
        host = _HostAsync(alt_result_status=StepStatus.FAILED)
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(
                type="alternate_step",
                step={"id": "alt", "type": "shell", "params": {}},
            ),
        )
        assert await host._execute_fallback_async(step, "boom", "wf-1") is None


class TestAsyncFallbackCallback:
    """Lines 282-283: async callback fallback — not registered → warning + None.
    Also covers 268-280 happy path for a registered callback."""

    @pytest.mark.asyncio
    async def test_callback_not_registered_returns_none(self):
        host = _HostAsync(fallback_callbacks={})
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(type="callback", callback="missing"),
        )
        assert await host._execute_fallback_async(step, "boom", "wf-1") is None

    @pytest.mark.asyncio
    async def test_callback_registered_is_invoked_in_executor(self):
        captured = {}

        def my_cb(step, context, error):
            captured["step_id"] = step.id
            captured["error_str"] = str(error)
            return {"from_callback": True}

        host = _HostAsync(fallback_callbacks={"cb": my_cb})
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(type="callback", callback="cb"),
        )
        out = await host._execute_fallback_async(step, "boom", "wf-1")
        assert out == {"from_callback": True}
        assert captured["step_id"] == "primary"
        assert "boom" in captured["error_str"]


class TestAsyncFallbackUnknownType:
    """Line 289: async trailing `return None` when no branch matches."""

    @pytest.mark.asyncio
    async def test_callback_without_name_falls_through(self):
        host = _HostAsync(fallback_callbacks={})
        step = StepConfig(
            id="primary",
            type="shell",
            fallback=FallbackConfig(type="callback", callback=None),
        )
        assert await host._execute_fallback_async(step, "boom", "wf-1") is None
