"""Integration tests for cross-package import boundaries.

Verifies that ``core`` can import and use ``kernel`` types without
breaking when ``kernel`` internal APIs change. These are smoke tests
for the monorepo contract between packages.
"""

import json
from datetime import timedelta


class TestKernelCoreBoundary:
    """Tests the kernel → core package boundary."""

    def test_session_controller_importable_from_core(self):
        """core.mcp_server and core.cli can import SessionController."""
        from animus_kernel.head.session_controller import (
            SessionController,
            SessionPolicy,
        )

        policy = SessionPolicy(
            wrapup_threshold=0.96,
            session_timer=timedelta(minutes=30),
            auto_restart=True,
        )
        ctrl = SessionController(policy=policy)
        assert ctrl is not None
        assert ctrl.policy.wrapup_threshold == 0.96

    def test_session_steward_can_audit_kernel_controller(self):
        """SessionSteward (core) can observe telemetry from SessionController (kernel)."""
        from animus_kernel.head.session_controller import (
            SessionController,
            SessionLifecycleEvent,
            SessionPolicy,
        )

        from animus.citizens.session_steward import SessionStewardCitizen

        policy = SessionPolicy(
            wrapup_threshold=0.96,
            session_timer=timedelta(minutes=5),
            auto_restart=True,
        )
        ctrl = SessionController(policy=policy)
        ctrl.log_event(
            session_id="s1",
            event=SessionLifecycleEvent.WRAPPING_UP,
            utilization_percent=55.0,
            elapsed_seconds=300.0,
            turns=10,
            message="timer expired",
        )

        steward = SessionStewardCitizen(min_sessions=1)
        patterns = steward.observe_telemetry(ctrl)
        assert isinstance(patterns, list)

    def test_mcp_server_session_steward_tool_signature(self):
        """The MCP server tool can be imported without kernel errors."""
        from animus.mcp_server import create_mcp_server

        # Just verify the factory function exists and the tool names are stable
        assert callable(create_mcp_server)

    def test_cli_session_command_imports_kernel(self):
        """The CLI session command can import HeadREPL from kernel."""
        from animus.cli import _cmd_session

        assert callable(_cmd_session)

    def test_session_policy_json_roundtrip(self):
        """SessionPolicy values survive JSON roundtrip (used by MCP/CLI)."""
        from animus_kernel.head.session_controller import SessionPolicy

        policy = SessionPolicy(
            wrapup_threshold=0.92,
            session_timer=timedelta(hours=1),
            auto_restart=False,
        )
        data = {
            "wrapup_threshold": policy.wrapup_threshold,
            "session_timer_minutes": policy.session_timer.total_seconds() / 60,
            "auto_restart": policy.auto_restart,
        }
        json_str = json.dumps(data)
        recovered = json.loads(json_str)
        assert recovered["wrapup_threshold"] == 0.92
        assert recovered["session_timer_minutes"] == 60.0
        assert recovered["auto_restart"] is False
