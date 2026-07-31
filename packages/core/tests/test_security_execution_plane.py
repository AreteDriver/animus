"""SEC-00 — execution-plane security regression tests for animus core.

These tests reproduce the pre-fix behavior described in
``security/SEC-00-threat-model.md``. They are designed to fail on the
SEC-00 baseline commit and pass once SEC-01..SEC-07 fixes land.

All proofs use temporary directories, mock HTTP servers, inert shell
commands, and fake secrets. No destructive commands or live credentials
are used.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.config import ToolsSecurityConfig
from animus.memory import MemoryLayer
from animus.memory.types import Sensitivity
from animus.network import is_egress_allowed
from animus.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    _set_security_config,
    _validate_command,
    _validate_path,
    _tool_http_request,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


class _MockHTTPHandler(BaseHTTPRequestHandler):
    """Tiny handler that returns a fixed body and never touches disk."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"mock-server-ok")

    def log_message(self, _fmt, *_args):
        # suppress default stderr logging
        pass


def _start_mock_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _MockHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def _stop_mock_server(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()


# ═══════════════════════════════════════════════════════════════════
# SEC-01 — missing security config means unrestricted
# ═══════════════════════════════════════════════════════════════════


class TestMissingConfigFailOpen:
    def test_validate_path_allows_blocked_path_when_config_is_none(self):
        """When _security_config is None, _validate_path returns True for /etc/shadow."""
        _set_security_config(None)
        is_valid, error = _validate_path("/etc/shadow")
        assert is_valid is True, f"expected fail-open, got error={error!r}"
        assert error is None

    def test_validate_command_allows_any_command_when_config_is_none(self):
        """When _security_config is None, _validate_command allows arbitrary input."""
        _set_security_config(None)
        is_valid, error = _validate_command("rm -rf /")
        assert is_valid is True, f"expected fail-open, got error={error!r}"
        assert error is None


# ═══════════════════════════════════════════════════════════════════
# SEC-02 — /build finally block clears security config globally
# ═══════════════════════════════════════════════════════════════════


class TestBuildFinallyClearsPolicy:
    def test_build_clears_security_config_in_finally(self, tmp_path: Path):
        """Mimics /build's try/finally that sets _set_security_config(None).

        After the block, even if a previous policy was active, the global
        config is gone and validation is unrestricted.
        """
        restricted = ToolsSecurityConfig(
            allowed_paths=[str(tmp_path)],
            blocked_paths=["/etc/shadow"],
            command_enabled=True,
            command_blocklist=["rm -rf /"],
        )
        _set_security_config(restricted)

        # Simulate /build sandbox body
        build_workspace = tmp_path / "build"
        build_workspace.mkdir()
        sandbox_config = ToolsSecurityConfig(
            allowed_paths=[str(build_workspace)],
            write_roots=[str(build_workspace)],
            command_enabled=True,
        )
        _set_security_config(sandbox_config)

        # /build finally resets global config to None
        _set_security_config(None)

        # Post-build: the previously-restricted policy is gone
        is_valid, error = _validate_path("/etc/shadow")
        assert is_valid is True, f"expected cleared policy, got error={error!r}"


# ═══════════════════════════════════════════════════════════════════
# SEC-03 — MCP server creates default registry with no security policy
# ═══════════════════════════════════════════════════════════════════


class TestMCPServerRegistryHasNoPolicy:
    def test_mcp_server_run_workflow_uses_default_registry_without_security_config(self, tmp_path: Path):
        """Inside the MCP ``animus_run_workflow`` tool handler, ``create_default_registry()``
        is called with no security_config argument. We exercise the tool handler with the
        MCP SDK stubbed so no real LLM or Forge run happens."""

        # Stub FastMCP and record every tool function registered by the server.
        registered_tools: dict[str, Any] = {}

        class _StubFastMCP:
            def __init__(self, *args, **kwargs):
                self._tool_manager = MagicMock()
                self._tool_manager.list_tools.return_value = []

            def tool(self):
                def _decorator(fn):
                    registered_tools[fn.__name__] = fn
                    return fn
                return _decorator

        stub_module = MagicMock()
        stub_module.FastMCP = _StubFastMCP

        with patch.dict("sys.modules", {"mcp.server.fastmcp": stub_module}):
            import importlib
            import animus.mcp_server as _mcp_server_module
            importlib.reload(_mcp_server_module)

            with patch("animus.mcp_server.AnimusConfig") as mock_cfg_cls, \
                 patch("animus.mcp_server.MemoryLayer") as mock_mem_cls, \
                 patch("animus.mcp_server.TaskTracker") as mock_tasks_cls, \
                 patch("animus.mcp_server.EgressAuditLog") as mock_audit_cls:

                cfg = MagicMock()
                cfg.data_dir = tmp_path
                cfg.memory.backend = "json"
                mock_cfg_cls.load.return_value = cfg
                mock_mem_cls.return_value = MagicMock()
                mock_tasks_cls.return_value = MagicMock()
                mock_audit_cls.return_value = MagicMock()

                _mcp_server_module.create_mcp_server()

        assert "animus_run_workflow" in registered_tools, (
            f"animus_run_workflow tool not registered; got {list(registered_tools.keys())}"
        )
        animus_run_workflow = registered_tools["animus_run_workflow"]

        captured_calls: list = []
        from animus.tools import create_default_registry as original_create_registry

        def _capture_create_registry(security_config=None):
            captured_calls.append(security_config)
            return original_create_registry(security_config)

        # Stub everything the workflow handler needs so no real execution occurs.
        fake_state = MagicMock()
        fake_state.status = "completed"
        fake_state.results = []
        fake_state.total_tokens = 0
        fake_state.total_cost = 0.0

        fake_engine = MagicMock()
        fake_engine.run.return_value = fake_state

        fake_workflow = MagicMock()
        fake_workflow.agents = []
        fake_workflow.name = "fake"
        fake_workflow.max_cost_usd = 0.0

        fake_model_config = MagicMock()

        with patch("animus.tools.create_default_registry", side_effect=_capture_create_registry), \
             patch("animus.cognitive.CognitiveLayer") as mock_cognitive_cls, \
             patch("animus.cognitive.ModelConfig") as mock_model_cls, \
             patch("animus.forge.ForgeEngine", return_value=fake_engine), \
             patch("animus.forge.loader.load_workflow", return_value=fake_workflow):

            mock_model_cls.ollama.return_value = fake_model_config
            mock_cognitive_cls.return_value = MagicMock()

            workflow_file = tmp_path / "workflow.yaml"
            workflow_file.write_text("name: fake\n")

            animus_run_workflow(
                workflow_path=str(workflow_file),
                task_description="",
                api_key="",
            )

        # The handler invoked create_default_registry() with no explicit policy.
        assert any(c is None for c in captured_calls), (
            "Expected create_default_registry to be called with security_config=None; "
            f"captured={captured_calls}"
        )


# ═══════════════════════════════════════════════════════════════════
# SEC-06 — requires_approval is metadata, not enforced by registry
# ═══════════════════════════════════════════════════════════════════


class TestRequiresApprovalNotEnforced:
    def test_registry_executes_approval_required_tool_without_callback(self):
        """ToolRegistry.execute() runs a requires_approval=True tool even though
        no approval callback is registered."""
        registry = ToolRegistry()

        def handler(params: dict) -> ToolResult:
            return ToolResult(tool_name="dangerous", success=True, output="ran")

        tool = Tool(
            name="dangerous",
            description="requires human approval",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
            requires_approval=True,
        )
        registry.register(tool)

        result = registry.execute("dangerous", {})
        assert result.success is True
        assert result.output == "ran"


# ═══════════════════════════════════════════════════════════════════
# SEC-07 — generic HTTP tool bypasses centralized egress policy
# ═══════════════════════════════════════════════════════════════════


class TestHttpRequestBypassesEgressPolicy:
    def test_http_request_ignores_egress_policy_when_blocked(self, monkeypatch):
        """_tool_http_request reaches the network even when is_egress_allowed says
        the destination should be blocked."""
        server, url = _start_mock_server()
        try:
            # Centralized egress policy says "deny everything"
            monkeypatch.setattr(
                "animus.network.egress.is_egress_allowed", lambda *args, **kwargs: False
            )

            result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
            # Pre-fix: the request ignores the policy and succeeds.
            assert result.success is True, f"expected success (egress ignored), got {result.error}"
            assert "mock-server-ok" in result.output
        finally:
            _stop_mock_server(server)

    def test_http_request_no_egress_call_for_private_target(self, monkeypatch):
        """_tool_http_request does not consult is_egress_allowed at all."""
        call_log: list[tuple] = []

        def _log_and_allow(destination, tier=None, content=None):
            call_log.append((destination, tier, content))
            return True

        monkeypatch.setattr("animus.network.egress.is_egress_allowed", _log_and_allow)

        server, url = _start_mock_server()
        try:
            result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
            assert result.success is True
            # Pre-fix: no egress check was performed.
            assert call_log == [], (
                "Expected no egress-policy call before fix; calls were logged: "
                f"{call_log}"
            )
        finally:
            _stop_mock_server(server)


# ═══════════════════════════════════════════════════════════════════
# SEC-08 — MemoryLayer logs raw unredacted content at INFO level
# ═══════════════════════════════════════════════════════════════════


class TestMemoryLayerLogsRawSecrets:
    def test_remember_logs_original_secret_before_redaction(self, tmp_path: Path, caplog):
        """MemoryLayer.remember() stores the redacted copy but logs the raw original
        content in its INFO-level preview."""
        fake_secret = "sk-ant-api03-fakefakefakefakefakefake"
        content = f"API key is {fake_secret}"

        data_dir = tmp_path / "memory"
        memory = MemoryLayer(data_dir, backend="json")

        with caplog.at_level(logging.INFO, logger="animus.memory"):
            memory.remember(content, sensitivity=Sensitivity.PUBLIC)

        # The stored memory must be redacted.
        stored = memory.store.list_all()
        assert len(stored) == 1
        assert fake_secret not in stored[0].content

        # Pre-fix: the INFO preview contains the raw secret.
        raw_in_log = any(fake_secret in record.message for record in caplog.records)
        assert raw_in_log, (
            "Expected raw secret in INFO log preview before fix; "
            f"logs: {[r.message for r in caplog.records]}"
        )
