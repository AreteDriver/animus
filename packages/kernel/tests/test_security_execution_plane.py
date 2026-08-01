"""SEC-05 — execution-plane SSRF / egress regression tests for animus kernel.

All proofs use temporary mock HTTP servers bound to loopback and fake
secrets.  No live credentials or destructive actions are used.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from animus.network.client import EgressDeniedError, GovernedClient, SSRFBlockedError

from animus_kernel.tools_core import _tool_http_request


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
