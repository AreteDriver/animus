"""SEC-05/SEC-06 — execution-plane security regression tests for animus kernel.

Covers SSRF/egress (SEC-05) and secret-safe logging/telemetry (SEC-06).
All proofs use temporary mock HTTP servers bound to loopback and fake
secrets.  No live credentials or destructive actions are used.
"""

from __future__ import annotations

import base64
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from animus.config import ToolsSecurityConfig
from animus.network.client import EgressDeniedError, GovernedClient, SSRFBlockedError

from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.tools.registry import ForgeToolRegistry
from animus_kernel.tools_core import (
    _set_security_config,
    _tool_http_request,
    _tool_run_command,
)
from animus_types.secrets import redact


class _MockHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"mock-server-ok")

    def log_message(self, _fmt, *_args):
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


class TestKernelHttpRequestSSRF:
    def test_tool_blocks_loopback_127(self):
        server, url = _start_mock_server()
        try:
            result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
            assert result.success is False
            assert "SSRF" in result.error or "loopback" in result.error.lower()
        finally:
            _stop_mock_server(server)

    def test_tool_blocks_localhost_hostname(self):
        server, url = _start_mock_server()
        try:
            host, port = server.server_address
            url = f"http://localhost:{port}/"
            result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
            assert result.success is False
        finally:
            _stop_mock_server(server)


class TestKernelGovernedClientSSRF:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "[::1]",
            "0.0.0.0",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.100.100.200",
        ],
    )
    def test_blocks_disallowed_addresses(self, host):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request(f"http://{host}/", timeout=2)

    def test_blocks_encoded_ipv4(self):
        with pytest.raises(SSRFBlockedError):
            GovernedClient.request("http://2130706433/", timeout=2)

    def test_blocks_metadata_hostnames(self):
        for host in [
            "metadata.google.internal",
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

    def test_outbound_secret_blocked_by_egress_dlp(self):
        with pytest.raises(EgressDeniedError):
            GovernedClient.request(
                "http://example.com/api",
                method="POST",
                body='{"token": "ghp_fake1234567890abcdef"}',
                timeout=2,
            )


# ═══════════════════════════════════════════════════════════════════════════
# SEC-06 — Secret-safe logging and telemetry
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fake_secret_corpus() -> dict[str, str]:
    """Inert fake secrets that exercise every credential pattern."""
    encoded_token_input = b"sk-ant-api03-encodedfakefakefakefakefakefake"
    return {
        "anthropic_key": "sk-ant-api03-fakefakefakefakefakefakefakefakefake",
        "openai_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890abcdef",
        "github_token": "ghp_fake1234567890abcdef",
        "github_pat": "github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdef",
        "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
        "stripe_key": "sk_test_fakefakefakefakefakefakefakefake",  # synthetic; not a real Stripe key
        "slack_token": "xoxb-fakefakefakefakefakefakefakefake",
        "bearer_token": "Bearer fakefakefakefakefakefakefakefakefakefakefakefake",
        "api_key_label": "my_secret_key_is: fakefakefakefakefakefakefake",
        "password_labeled": "password=Sup3rS3cr3tP@ssw0rd!12345",
        "encoded_token": base64.b64encode(encoded_token_input).decode(),
        "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    }


@pytest.fixture(autouse=True)
def _reset_kernel_security_config():
    """Isolate tests that mutate the module-level security config."""
    yield
    _set_security_config(None)


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


class TestToolCommandLogsNoSecrets:
    def test_command_validation_log_does_not_emit_secret(
        self, caplog, fake_secret_corpus
    ):
        """A rejected shell command containing a secret must not leak the secret."""
        secret = fake_secret_corpus["password_labeled"]
        _set_security_config(ToolsSecurityConfig(command_enabled=False))
        command = f"echo {secret}"

        logger = logging.getLogger("animus")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="animus.tools"):
                result = _tool_run_command({"command": command})
        finally:
            logger.propagate = old_propagate

        assert result.success is False
        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, (
            f"Raw secret leaked into command validation log: {logged}"
        )
        # Command intent remains visible.
        assert "echo" in logged


class TestHttpToolLogsNoSecrets:
    def test_http_request_failure_log_does_not_emit_auth_secret(
        self, caplog, fake_secret_corpus
    ):
        """An HTTP request failure must not echo the bearer token in logs."""
        secret = fake_secret_corpus["bearer_token"]

        logger = logging.getLogger("animus")
        old_propagate = logger.propagate
        logger.propagate = True
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
            logger.propagate = old_propagate

        assert result.success is False
        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, (
            f"Bearer secret leaked into HTTP tool log: {logged}"
        )
        assert "http_request failed" in logged


class TestForgeRegistryAuditLogsNoSecrets:
    def test_audit_log_redacts_string_arguments(
        self, tmp_path, caplog, fake_secret_corpus
    ):
        """Structured forge.tool_audit entries must redact string args."""
        secret = fake_secret_corpus["github_pat"]
        nested_secret = fake_secret_corpus["password_labeled"]
        registry = ForgeToolRegistry(project_root=tmp_path, enable_shell=False)

        logger = logging.getLogger("forge.tool_audit")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger="forge.tool_audit"):
                registry.execute(
                    "nonexistent_tool",
                    {
                        "token": secret,
                        "correlation_id": "cid-123",
                        "nested": {"password": nested_secret},
                    },
                    agent_id="test-agent",
                )
        finally:
            logger.propagate = old_propagate

        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, f"Secret leaked into audit log: {logged}"
        assert nested_secret not in logged, (
            f"Nested secret leaked into audit log: {logged}"
        )
        # Operational fields survive.
        assert "nonexistent_tool" in logged
        assert "test-agent" in logged
        assert "cid-123" in logged


class TestHeadToolOrchestratorLogsNoSecrets:
    def test_tool_call_info_log_redacts_string_args(
        self, tmp_path, caplog, fake_secret_corpus
    ):
        """The Head tool-call INFO log keeps tool names but redacts string args."""
        secret = fake_secret_corpus["anthropic_key"]
        nested_secret = fake_secret_corpus["password_labeled"]
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_mcp=False,
        )

        logger = logging.getLogger("animus_kernel.head.tool_orchestrator")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger=logger.name):
                result = orchestrator.execute(
                    "nonexistent_tool",
                    {"key": secret, "nested": {"password": nested_secret}},
                )
        finally:
            logger.propagate = old_propagate

        assert "nonexistent_tool" in result
        logged = "\n".join(record.message for record in caplog.records)
        assert secret not in logged, (
            f"Secret leaked into Head tool-call log: {logged}"
        )
        assert nested_secret not in logged, (
            f"Nested secret leaked into Head tool-call log: {logged}"
        )
        assert "nonexistent_tool" in logged
