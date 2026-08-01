"""Tests for AgentRuntime and LocalRuntime."""

from __future__ import annotations

from animus.mission.order import MissionOrder
from animus.mission.runtime import LocalRuntime, RuntimeCapabilities


class TestRuntimeCapabilities:
    def test_default_can_handle(self):
        caps = RuntimeCapabilities()
        assert caps.can_handle("anything") is True

    def test_max_concurrent(self):
        caps = RuntimeCapabilities(max_concurrent_missions=3)
        assert caps.max_concurrent_missions == 3


class TestLocalRuntime:
    def test_spawn(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        assert handle.startswith("local-")
        assert len(handle) > len("local-")

    def test_message_status(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        resp = rt.message(handle, {"type": "status"})
        assert resp["type"] == "status"
        assert resp["state"] == "spawned"

    def test_message_unknown_handle(self):
        rt = LocalRuntime()
        resp = rt.message("invalid-handle", {"type": "status"})
        assert "error" in resp

    def test_checkpoint(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        state = rt.checkpoint(handle)
        assert state["handle"] == handle
        assert "message_count" in state
        assert "checkpointed_at" in state

    def test_terminate(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        final = rt.terminate(handle, reason="complete")
        assert final["reason"] == "complete"
        assert final["handle"] == handle
        assert "terminated_at" in final

    def test_tool_call(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        result = rt.tool_call(handle, "read_file", {"path": "/tmp/test.py"})
        assert result["tool"] == "read_file"
        assert "/tmp/test.py" in result["path"]

    def test_schedule(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        task_id = rt.schedule(handle, {"action": "poll"}, delay_seconds=10.0)
        assert task_id.startswith("task-")

    def test_message_objective_progress(self):
        rt = LocalRuntime()
        order = MissionOrder(citizen_id="c1", mission_type="scan")
        handle = rt.spawn(order)
        resp = rt.message(handle, {"type": "objective_progress", "objective_id": "o1"})
        assert resp["type"] == "progress"
        assert resp["objective_id"] == "o1"

    def test_capabilities(self):
        rt = LocalRuntime()
        assert rt.name == "local"
        assert rt.capabilities.supports_async is True
        assert rt.capabilities.supports_checkpointing is True
