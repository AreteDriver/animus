"""SEC-00 — execution-plane security regression tests for animus core.

These tests reproduce the pre-fix behavior described in
``security/SEC-00-threat-model.md``. They are designed to fail on the
SEC-00 baseline commit and pass once SEC-01..SEC-07 fixes land.

All proofs use temporary directories, mock HTTP servers, inert shell
commands, and fake secrets. No destructive commands or live credentials
are used.
"""

from __future__ import annotations

import importlib
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from animus_types.secrets import redact

from animus.memory import MemoryLayer
from animus.memory.types import Sensitivity
from animus.network.client import EgressDeniedError, GovernedClient, SSRFBlockedError
from animus.tools import (
    DenyAllToolPolicy,
    Tool,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    WorkspaceToolPolicy,
    _tool_http_request,
    _tool_run_command,
    _validate_command,
    _validate_path,
    create_default_registry,
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


class TestMissingPolicyFailsClosed:
    def test_validate_path_denies_blocked_path_when_no_policy(self):
        """When no policy is supplied, _validate_path fails closed."""
        is_valid, error = _validate_path("/etc/shadow")
        assert is_valid is False, f"expected fail-closed, got is_valid={is_valid}"
        assert error is not None
        assert "no tool policy" in error.lower()

    def test_validate_command_denies_any_command_when_no_policy(self):
        """When no policy is supplied, _validate_command fails closed."""
        is_valid, error = _validate_command("rm -rf /")
        assert is_valid is False, f"expected fail-closed, got is_valid={is_valid}"
        assert error is not None
        assert "no tool policy" in error.lower()

    def test_registry_without_policy_uses_deny_all(self, tmp_path: Path):
        """A ToolRegistry created without an explicit policy defaults to DenyAllToolPolicy."""
        registry = create_default_registry()
        assert isinstance(registry.policy, DenyAllToolPolicy)

        # read_file through the registry should be denied.
        result = registry.execute("read_file", {"path": str(tmp_path / "file.txt")})
        assert result.success is False
        assert result.error is not None
        assert "no tool policy" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════
# SEC-01A — registry-owned policies are isolated; no global mutable state
# ═══════════════════════════════════════════════════════════════════


class TestRegistryPoliciesAreIsolated:
    def test_two_registries_can_use_different_policies(self, tmp_path: Path):
        """Multiple registries in one process must not share mutable policy state."""
        restricted_dir = tmp_path / "restricted"
        build_dir = tmp_path / "build"
        restricted_dir.mkdir()
        build_dir.mkdir()

        restricted = WorkspaceToolPolicy(
            allowed_paths=[str(restricted_dir)],
            blocked_paths=["/etc/shadow"],
            command_enabled=True,
            command_blocklist=["rm -rf /"],
        )
        build = WorkspaceToolPolicy(
            allowed_paths=[str(build_dir)],
            write_roots=[str(build_dir)],
            command_enabled=True,
        )

        restricted_registry = create_default_registry(policy=restricted)
        build_registry = create_default_registry(policy=build)

        # The build registry can write inside its workspace once approved.
        build_write_params = {"path": str(build_dir / "file.py"), "content": "x = 1\n"}
        build_write_params["_approval_id"] = build_registry.request_approval(
            "write_file", build_write_params
        )
        write_result = build_registry.execute("write_file", build_write_params)
        assert write_result.success is True, write_result.error

        # The restricted registry still denies writes outside its scope.
        restricted_write_params = {
            "path": str(build_dir / "file.py"),
            "content": "x = 1\n",
        }
        restricted_write_params["_approval_id"] = restricted_registry.request_approval(
            "write_file", restricted_write_params
        )
        denied = restricted_registry.execute("write_file", restricted_write_params)
        assert denied.success is False
        assert "denied" in denied.error.lower()

        # The restricted registry still blocks /etc/shadow.
        shadow = restricted_registry.execute("read_file", {"path": "/etc/shadow"})
        assert shadow.success is False
        assert "blocked" in shadow.error.lower() or "denied" in shadow.error.lower()

    def test_build_does_not_clear_global_state(self, tmp_path: Path):
        """After /build constructs a local registry, the main registry's policy is unchanged."""
        # Simulating the new /build behavior: create a local registry with a workspace policy.
        build_workspace = tmp_path / "build"
        build_workspace.mkdir()
        build_policy = WorkspaceToolPolicy(
            allowed_paths=[str(build_workspace)],
            write_roots=[str(build_workspace)],
            command_enabled=True,
        )
        build_registry = create_default_registry(policy=build_policy)

        # Meanwhile, a main registry keeps its own policy.
        main_dir = tmp_path / "main"
        main_dir.mkdir()
        main_policy = WorkspaceToolPolicy(
            allowed_paths=[str(main_dir)],
            write_roots=[str(main_dir)],
            command_enabled=True,
        )
        main_registry = create_default_registry(policy=main_policy)

        # Execute in build registry after requesting approval.
        build_write_params = {
            "path": str(build_workspace / "file.py"),
            "content": "x = 1\n",
        }
        build_write_params["_approval_id"] = build_registry.request_approval(
            "write_file", build_write_params
        )
        build_registry.execute("write_file", build_write_params)

        # Main registry policy is unaffected; write to its own workspace still works.
        main_write_params = {
            "path": str(main_dir / "file.py"),
            "content": "y = 2\n",
        }
        main_write_params["_approval_id"] = main_registry.request_approval(
            "write_file", main_write_params
        )
        main_result = main_registry.execute("write_file", main_write_params)
        assert main_result.success is True, main_result.error


# ═══════════════════════════════════════════════════════════════════
# SEC-03 — MCP server creates default registry with a restrictive policy
# ═══════════════════════════════════════════════════════════════════


class TestMCPServerRegistryUsesRestrictivePolicy:
    def test_mcp_server_run_workflow_uses_restrictive_policy(self, tmp_path: Path):
        """Inside the MCP ``animus_run_workflow`` tool handler, ``create_default_registry()``
        is called with an explicit ``WorkspaceToolPolicy``. We exercise the tool handler with
        the MCP SDK stubbed so no real LLM or Forge run happens."""

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

        captured_policies: list[ToolPolicy] = []
        from animus.tools import create_default_registry as original_create_registry

        def _capture_create_registry(policy=None, security_config=None):
            captured_policies.append(policy)
            return original_create_registry(policy=policy, security_config=security_config)

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

        # The handler invoked create_default_registry() with an explicit restrictive policy.
        assert captured_policies, "Expected create_default_registry to be called"
        assert all(p is not None for p in captured_policies), (
            "Expected create_default_registry to be called with a non-None policy; "
            f"captured={captured_policies}"
        )
        assert any(isinstance(p, WorkspaceToolPolicy) for p in captured_policies), (
            "Expected a WorkspaceToolPolicy; " f"captured={captured_policies}"
        )


# ═══════════════════════════════════════════════════════════════════
# SEC-02 — approval is enforced at the registry execution boundary
# ═══════════════════════════════════════════════════════════════════


class TestRequiresApprovalEnforced:
    def _register_dangerous_tool(self, registry: ToolRegistry):
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
        return handler

    def test_registry_rejects_approval_required_tool_without_approval_id(self):
        """ToolRegistry.execute() denies a requires_approval=True tool when no
        approval_id is supplied."""
        registry = ToolRegistry()
        self._register_dangerous_tool(registry)

        result = registry.execute("dangerous", {})
        assert result.success is False
        assert "approval" in result.error.lower()

    def test_registry_executes_approved_tool(self):
        """A valid approval_id lets an approved tool run."""
        registry = ToolRegistry()
        self._register_dangerous_tool(registry)

        params = {"target": "production"}
        approval_id = registry.request_approval(
            "dangerous", params, approver="test", reason="test approval"
        )
        params["_approval_id"] = approval_id

        result = registry.execute("dangerous", params)
        assert result.success is True
        assert result.output == "ran"

    def test_registry_rejects_unknown_approval_id(self):
        """An approval_id that does not exist in the store is rejected."""
        registry = ToolRegistry()
        self._register_dangerous_tool(registry)

        params = {"_approval_id": "does-not-exist"}
        result = registry.execute("dangerous", params)
        assert result.success is False
        assert "approval" in result.error.lower()

    def test_registry_rejects_approval_with_mismatched_params(self):
        """Reusing an approval for different parameters is rejected."""
        registry = ToolRegistry()
        self._register_dangerous_tool(registry)

        approval_id = registry.request_approval(
            "dangerous", {"target": "production"}, approver="test", reason="test approval"
        )

        result = registry.execute(
            "dangerous", {"target": "production", "_approval_id": approval_id}
        )
        assert result.success is True, result.error

        # Reuse the same approval_id with a different logical parameter.
        result2 = registry.execute(
            "dangerous", {"target": "staging", "_approval_id": approval_id}
        )
        assert result2.success is False
        assert "mismatch" in result2.error.lower()

    def test_registry_rejects_approval_for_different_tool(self):
        """An approval granted for one tool cannot authorize another."""
        registry = ToolRegistry()
        self._register_dangerous_tool(registry)
        registry.register(
            Tool(
                name="other",
                description="another tool",
                parameters={},
                handler=lambda p: ToolResult(tool_name="other", success=True, output="ok"),
                requires_approval=True,
            )
        )

        approval_id = registry.request_approval(
            "dangerous", {}, approver="test", reason="test approval"
        )
        result = registry.execute("other", {"_approval_id": approval_id})
        assert result.success is False
        assert "mismatch" in result.error.lower()

    def test_registry_rejects_expired_approval(self):
        """An expired approval cannot authorize execution."""
        registry = ToolRegistry()
        self._register_dangerous_tool(registry)

        params = {}
        approval_id = registry.request_approval(
            "dangerous",
            params,
            approver="test",
            reason="short-lived approval",
            expiry_seconds=1,
        )
        params["_approval_id"] = approval_id

        import time

        time.sleep(1.1)
        result = registry.execute("dangerous", params)
        assert result.success is False
        assert "expired" in result.error.lower()

    def test_sensitive_values_not_leaked_in_audit_hash(self):
        """The canonical parameter hash binds to a secret without exposing it."""
        from animus.tools import _canonical_params_hash

        secret_a = {"auth_value": "super-secret-token-a"}
        secret_b = {"auth_value": "super-secret-token-b"}

        hash_a = _canonical_params_hash(secret_a)
        hash_b = _canonical_params_hash(secret_b)

        # Different secrets must produce different hashes (binding).
        assert hash_a != hash_b
        # The secret value itself does not appear in the hash output.
        assert "super-secret-token" not in hash_a

    def test_canonical_hash_stable_for_same_logical_request(self):
        """The same logical parameters must always hash to the same value."""
        from animus.tools import _canonical_params_hash

        params = {"b": 2, "a": 1, "nested": {"c": True, "d": "x"}}
        assert _canonical_params_hash(params) == _canonical_params_hash(
            {"a": 1, "b": 2, "nested": {"d": "x", "c": True}}
        )


# ═══════════════════════════════════════════════════════════════════
# SEC-05 — generic HTTP tool enforces SSRF guards and centralized egress
# ═══════════════════════════════════════════════════════════════════


class TestHttpRequestEgressAndSSRFGuards:
    """Post-SEC-05 behavior: the generic HTTP tool is funneled through the
    governed client, which validates every destination and outbound payload.
    """

    def test_http_request_blocked_for_loopback_127(self):
        server, url = _start_mock_server()
        try:
            result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
            assert result.success is False, "expected loopback to be blocked"
            assert "SSRF" in result.error or "loopback" in result.error.lower()
        finally:
            _stop_mock_server(server)

    def test_http_request_blocked_for_localhost_hostname(self):
        server, url = _start_mock_server()
        try:
            host, port = server.server_address
            url = f"http://localhost:{port}/"
            result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
            assert result.success is False
        finally:
            _stop_mock_server(server)

    def test_http_request_egress_policy_denies_blocked_destination(self, monkeypatch):
        server, url = _start_mock_server()
        try:
            monkeypatch.setattr(
                "animus.network.client.is_egress_allowed", lambda *args, **kwargs: False
            )

            with pytest.raises(EgressDeniedError):
                GovernedClient.request(
                    url, timeout=5, allow_loopback=True, sensitivity="PUBLIC"
                )
        finally:
            _stop_mock_server(server)

    def test_egress_policy_is_consulted_for_allowed_request(self, monkeypatch):
        """The governed client calls the centralized egress policy before sending."""
        call_log: list[tuple] = []

        def _log_and_allow(destination, tier=None, *, sensitivity=None, content=None):
            call_log.append((destination, tier, sensitivity, content))
            return True

        monkeypatch.setattr("animus.network.client.is_egress_allowed", _log_and_allow)

        server, url = _start_mock_server()
        try:
            result = GovernedClient.request(
                url, timeout=5, allow_loopback=True, sensitivity="PUBLIC"
            )
            assert result.status == 200
            assert call_log != [], (
                "Expected egress-policy call after fix; calls: " f"{call_log}"
            )
        finally:
            _stop_mock_server(server)


class TestGovernedClientSSRFBlocks:
    """Direct SSRF regression tests for ``animus.network.client.GovernedClient``."""

    def test_blocks_plain_ipv4_loopback(self):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request("http://127.0.0.1:8000/", timeout=2)

    def test_blocks_ipv6_loopback(self):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request("http://[::1]:8000/", timeout=2)

    def test_blocks_unspecified_address(self):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request("http://0.0.0.0:8000/", timeout=2)

    @pytest.mark.parametrize(
        "host",
        [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
        ],
    )
    def test_blocks_rfc1918_addresses(self, host):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request(f"http://{host}:8000/", timeout=2)

    @pytest.mark.parametrize(
        "host",
        [
            "169.254.169.254",
            "100.100.100.200",
            "fd00:ec2::254",
        ],
    )
    def test_blocks_metadata_and_link_local_addresses(self, host):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request(f"http://{host}/", timeout=2)

    @pytest.mark.parametrize(
        "host",
        [
            "2130706433",
            "0x7f000001",
            "0177.0.0.1",
        ],
    )
    def test_blocks_encoded_ipv4_loopback(self, host):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request(f"http://{host}:8000/", timeout=2)

    def test_blocks_metadata_hostnames(self):
        for host in [
            "metadata.google.internal",
            "metadata.platform.instance.net",
            "metadata.oraclecloud.com",
            "169.254.169.254.nip.io",
        ]:
            with pytest.raises(SSRFBlockedError, match=host):
                GovernedClient.request(f"http://{host}/", timeout=2)

    def test_allows_loopback_when_explicitly_allowed(self):
        server, url = _start_mock_server()
        try:
            result = GovernedClient.request(url, timeout=5, allow_loopback=True)
            assert result.status == 200
            assert "mock-server-ok" in result.body
        finally:
            _stop_mock_server(server)

    def test_allows_localhost_when_explicitly_allowed(self):
        server, _ = _start_mock_server()
        try:
            host, port = server.server_address
            url = f"http://localhost:{port}/"
            result = GovernedClient.request(url, timeout=5, allow_loopback=True)
            assert result.status == 200
            assert "mock-server-ok" in result.body
        finally:
            _stop_mock_server(server)

    def test_redirect_to_private_host_is_blocked(self):
        class _RedirectHandler(_MockHTTPHandler):
            def do_GET(self):
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://169.254.169.254/")
                    self.end_headers()
                else:
                    super().do_GET()

        server = HTTPServer(("127.0.0.1", 0), _RedirectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            url = f"http://{host}:{port}/redirect"
            with pytest.raises(SSRFBlockedError):
                GovernedClient.request(
                    url, timeout=5, allow_loopback=True, max_redirects=3
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_outbound_secret_blocked_by_egress_dlp(self):
        with pytest.raises(EgressDeniedError):
            GovernedClient.request(
                "http://example.com/api",
                method="POST",
                body='{"token": "ghp_fake1234567890abcdef"}',
                timeout=2,
            )


# ═══════════════════════════════════════════════════════════════════
# SEC-06 — Secret-safe logging and telemetry
# ═══════════════════════════════════════════════════════════════════


class _PrivateKeyParts:
    """Assemble a fake private key block at runtime so the source file never
    contains the contiguous marker lines that secret scanners flag.
    """

    BEGIN = "-----BEGIN "
    END = "-----END "
    RSA = "RSA PRIVATE KEY"
    PAD = "-----"
    BODY = "MIIEpAIBAAKCAQEAx..."

    @classmethod
    def block(cls) -> str:
        return "\n".join(
            [
                f"{cls.BEGIN}{cls.RSA}{cls.PAD}",
                cls.BODY,
                f"{cls.END}{cls.RSA}{cls.PAD}",
            ]
        )


class TestRedactionCorpus:
    """Sanity-check the shared fake-secret corpus used by SEC-06 tests."""

    def test_fake_secrets_are_detected_and_redacted(self, fake_secret_corpus):
        for name, secret in fake_secret_corpus.items():
            if name == "correlation_id":
                continue
            redacted = redact(secret)
            assert secret not in redacted, (
                f"secret type {name!r} was not redacted: {redacted!r}"
            )

    def test_correlation_id_survives_redaction(self, fake_secret_corpus):
        cid = fake_secret_corpus["correlation_id"]
        assert redact(f"request_id={cid}") == f"request_id={cid}"

    def test_multiline_private_key_redacted(self):
        block = _PrivateKeyParts.block()
        redacted = redact(block)
        assert _PrivateKeyParts.BODY not in redacted
        assert "[REDACTED:" in redacted

    def test_base64_encoded_token_redacted(self, fake_secret_corpus):
        token = fake_secret_corpus["encoded_token"]
        redacted = redact(token)
        assert token not in redacted


class TestMemoryLayerLogsNoSecrets:
    def test_remember_log_does_not_contain_raw_secret(
        self, tmp_path: Path, caplog, fake_secret_corpus
    ):
        """MemoryLayer.remember() must log a redacted preview, never the raw secret."""
        fake_secret = fake_secret_corpus["anthropic_key"]
        content = f"API key is {fake_secret}"

        data_dir = tmp_path / "memory"
        memory = MemoryLayer(data_dir, backend="json")

        # Ensure log records reach pytest's root capture handler even if a
        # previous test configured the animus logger with propagate=False.
        animus_logger = logging.getLogger("animus")
        old_propagate = animus_logger.propagate
        animus_logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger="animus.memory"):
                memory.remember(content, sensitivity=Sensitivity.PUBLIC)
        finally:
            animus_logger.propagate = old_propagate

        # The stored memory must be redacted.
        stored = memory.store.list_all()
        assert len(stored) == 1
        assert fake_secret not in stored[0].content

        # Post-fix: the INFO preview must not contain the raw secret.
        raw_in_log = any(fake_secret in record.message for record in caplog.records)
        assert raw_in_log is False, (
            "Raw secret leaked into INFO log preview; "
            f"logs: {[r.message for r in caplog.records]}"
        )
        # Operational metadata (memory id) still present.
        assert any(stored[0].id[:8] in record.message for record in caplog.records)


class TestToolCommandLogsNoSecrets:
    def test_command_validation_log_does_not_emit_secret(
        self, caplog, fake_secret_corpus
    ):
        """A rejected command containing a secret must not leak the secret in logs."""
        secret = fake_secret_corpus["password_labeled"]
        # command_enabled=False by default, so any command is denied.
        command = f"echo {secret}"

        animus_logger = logging.getLogger("animus")
        old_propagate = animus_logger.propagate
        animus_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="animus.tools"):
                result = _tool_run_command({"command": command})
        finally:
            animus_logger.propagate = old_propagate

        assert result.success is False
        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, (
            f"Raw secret leaked into command validation log: {logged}"
        )
        # Operational signal: command intent still present.
        assert "echo" in logged


class TestHttpToolLogsNoSecrets:
    def test_http_request_failure_log_does_not_emit_auth_secret(
        self, caplog, fake_secret_corpus
    ):
        """An HTTP request failure must not echo the bearer/api-key in logs."""
        secret = fake_secret_corpus["bearer_token"]

        animus_logger = logging.getLogger("animus")
        old_propagate = animus_logger.propagate
        animus_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger="animus.tools"):
                result = _tool_http_request(
                    {
                        "url": "http://127.0.0.1:1/",
                        "method": "GET",
                        "timeout": 1,
                        "auth_type": "bearer",
                        "auth_value": secret,
                    }
                )
        finally:
            animus_logger.propagate = old_propagate

        assert result.success is False
        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, (
            f"Bearer secret leaked into HTTP tool log: {logged}"
        )
        assert "http_request failed" in logged


class TestMCPAuthFailureLogsNoSecrets:
    def test_mcp_auth_failure_log_does_not_echo_key(self, monkeypatch, caplog):
        """A failed MCP API-key check logs the event but not the supplied key."""
        monkeypatch.setenv("ANIMUS_MCP_API_KEY", "real-key-never-logged-fake")
        # Force re-import of module-level key by reloading is not required:
        # _check_auth reads the module variable, which was set at import time.
        # We patch it directly for a deterministic test.
        import animus.mcp_server as _mcp_server

        _mcp_server._MCP_API_KEY = "real-key-never-logged-fake"
        try:
            animus_logger = logging.getLogger("animus")
            old_propagate = animus_logger.propagate
            animus_logger.propagate = True
            try:
                with caplog.at_level(logging.WARNING, logger="animus.mcp_server"):
                    error = _mcp_server._check_auth("wrong-key-value")
            finally:
                animus_logger.propagate = old_propagate

            assert error is not None
            logged = "\n".join(record.message for record in caplog.records)
            assert "wrong-key-value" not in logged
            assert "real-key-never-logged-fake" not in logged
            assert "API key mismatch" in logged
        finally:
            _mcp_server._MCP_API_KEY = None


class TestApprovalDecisionLogsNoSecrets:
    def test_approval_reason_is_redacted_in_log(self, caplog, fake_secret_corpus):
        """Approval decision logs keep metadata but redact the reason string."""
        secret = fake_secret_corpus["github_token"]
        registry = ToolRegistry()

        def handler(params: dict) -> ToolResult:
            return ToolResult(tool_name="dangerous", success=True, output="ran")

        registry.register(
            Tool(
                name="dangerous",
                description="requires human approval",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=handler,
                requires_approval=True,
            )
        )

        params = {"target": "production"}
        approval_id = registry.request_approval(
            "dangerous",
            params,
            approver="test",
            reason=f"approved because {secret}",
        )
        params["_approval_id"] = approval_id

        animus_logger = logging.getLogger("animus")
        old_propagate = animus_logger.propagate
        animus_logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger="animus.tools"):
                result = registry.execute("dangerous", params)
        finally:
            animus_logger.propagate = old_propagate

        assert result.success is True
        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, (
            f"Approval secret leaked into INFO log: {logged}"
        )
        # Correlation/request metadata preserved.
        assert approval_id in logged
