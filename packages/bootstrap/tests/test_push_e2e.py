"""End-to-end push test: backend → mock Web Push protocol server.

Since Playwright is not installed, this test uses an HTTP mock server as the
push endpoint.  It mocks ``pywebpush`` to send real HTTP requests to the mock
server, validating that ``POST /api/push/send-test`` produces a well-formed
Web Push request (VAPID auth header + JSON payload) without requiring the
``pywebpush`` package itself.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.config.schema import AnimusConfig
from animus_bootstrap.dashboard.app import app
from animus_bootstrap.intelligence.push_sender import generate_vapid_keys
from animus_bootstrap.intelligence.push_store import PushSubscriptionStore

# ------------------------------------------------------------------
# Mock push endpoint
# ------------------------------------------------------------------


class _Captured:
    """Thread-safe request capture."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def add(self, req: dict) -> None:
        with self._lock:
            self.requests.append(req)


CAPTURED = _Captured()


class _MockPushHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records incoming push requests."""

    def log_message(self, format: str, *args: object) -> None:
        pass  # suppress noise

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        CAPTURED.add(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": body.decode("utf-8") if body else None,
            }
        )

        self.send_response(201)
        self.end_headers()


class _GoneHandler(BaseHTTPRequestHandler):
    """Returns 410 Gone to simulate expired subscription."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.send_error(410)
        self.end_headers()


def _make_fake_pywebpush(*, auth_header: bool = True) -> types.ModuleType:
    """Build a fake ``pywebpush`` module that forwards to a local HTTP server.

    When ``auth_header`` is True, a synthetic ``vapid`` Authorization header
    is included so the assertion chain can verify VAPID signing.
    """
    fake = types.ModuleType("pywebpush")

    class WebPushException(Exception):
        def __init__(self, message: str, status: int) -> None:
            super().__init__(message)
            self.response = types.SimpleNamespace(status_code=status)

    def webpush(*, subscription_info, data, vapid_private_key, vapid_claims):  # type: ignore[no-untyped-def]
        headers: dict[str, str] = {"Content-Type": "application/json", "TTL": "60"}
        if auth_header:
            headers["Authorization"] = f"vapid t=FAKEJWT.{vapid_private_key[:8]}"
        req = urllib.request.Request(
            subscription_info["endpoint"],
            data=data.encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 201:
                    return types.SimpleNamespace(status_code=201)
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                raise WebPushException("gone", 410)
            raise WebPushException(str(exc), exc.code)
        return types.SimpleNamespace(status_code=201)

    fake.WebPushException = WebPushException  # type: ignore[attr-defined]
    fake.webpush = webpush  # type: ignore[attr-defined]
    return fake


def _wait_for_requests(captured: _Captured, expected: int, timeout: float = 3.0) -> list[dict]:
    """Poll until ``expected`` requests arrive or ``timeout`` expires.

    Returns the captured requests. Raises ``AssertionError`` on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        with captured._lock:
            reqs = list(captured.requests)
        if len(reqs) >= expected:
            return reqs
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Expected {expected} request(s), got {len(reqs)} after {timeout}s"
            )
        time.sleep(0.05)


def _csrf_headers(client: TestClient) -> dict[str, str]:
    """Extract CSRF token from client cookies for POST requests."""
    token = client.cookies.get("animus_csrf")
    return {"x-csrf-token": token} if token else {}


@pytest.fixture()
def restore_push_store() -> Iterator[None]:
    had = hasattr(app.state, "push_store")
    original = getattr(app.state, "push_store", None)
    try:
        yield
    finally:
        if had:
            app.state.push_store = original
        elif hasattr(app.state, "push_store"):
            delattr(app.state, "push_store")


# ------------------------------------------------------------------
# E2E tests
# ------------------------------------------------------------------


class TestPushE2E:
    def test_send_test_hits_mock_endpoint(
        self, tmp_path, monkeypatch, restore_push_store: None
    ) -> None:  # type: ignore[no-untyped-def]
        """Subscribe a fake endpoint on a local HTTP server, send a test push,
        and verify the mock server receives a well-formed Web Push request.
        """
        # Start a mock push server on an ephemeral port.
        server = HTTPServer(("127.0.0.1", 0), _MockPushHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        endpoint = f"http://127.0.0.1:{port}/push/test-device"

        fake = _make_fake_pywebpush(auth_header=True)
        monkeypatch.setitem(sys.modules, "pywebpush", fake)

        # Create a store with one subscription pointing at our mock server.
        store = PushSubscriptionStore(tmp_path / "push_e2e.db")
        store.add(
            {
                "endpoint": endpoint,
                "keys": {
                    "p256dh": "BOrvLCh7aN4U8bZ3vVjQv8X1s2t3u4v5w6x7y8z9a0b1c2d3e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3y4z5a6b7c8d9e0f1",
                    "auth": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                },
            }
        )
        app.state.push_store = store

        # Generate real VAPID keys so the JWT signature is valid.
        original_config = getattr(app.state, "config", None)
        cfg = AnimusConfig()
        priv, pub = generate_vapid_keys()
        cfg.services.vapid_private_key = priv
        cfg.services.vapid_public_key = pub
        app.state.config = cfg

        client = TestClient(app)
        client.get("/health")  # Prime CSRF cookie

        try:
            CAPTURED.requests.clear()

            resp = client.post(
                "/api/push/send-test",
                json={"title": "E2E Test", "body": "Hello from e2e", "url": "/test"},
                headers=_csrf_headers(client),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["sent"] == 1
            assert data["pruned"] == 0

            reqs = _wait_for_requests(CAPTURED, expected=1)
            req = reqs[0]
            assert req["path"] == "/push/test-device"

            # Verify payload structure.
            payload = json.loads(req["body"])
            assert payload["title"] == "E2E Test"
            assert payload["body"] == "Hello from e2e"
            assert payload["url"] == "/test"

            # Verify VAPID Authorization header is present.
            auth = req["headers"].get("Authorization", "")
            assert auth.startswith("vapid "), f"Expected vapid auth, got: {auth}"

            # Verify TTL header (Web Push requirement).  BaseHTTPRequestHandler
            # lower-cases header keys, so check case-insensitively.
            assert any(k.lower() == "ttl" for k in req["headers"]), f"Headers: {req['headers']}"

        finally:
            server.shutdown()
            store.close()
            if original_config is not None:
                app.state.config = original_config

    def test_send_test_with_prune(self, tmp_path, monkeypatch, restore_push_store: None) -> None:  # type: ignore[no-untyped-def]
        """A 410 response from the push server should prune the subscription."""
        server = HTTPServer(("127.0.0.1", 0), _GoneHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        endpoint = f"http://127.0.0.1:{port}/push/gone"

        fake = _make_fake_pywebpush(auth_header=False)
        monkeypatch.setitem(sys.modules, "pywebpush", fake)

        store = PushSubscriptionStore(tmp_path / "push_e2e_prune.db")
        store.add(
            {
                "endpoint": endpoint,
                "keys": {"p256dh": "x", "auth": "y"},
            }
        )
        app.state.push_store = store

        original_config = getattr(app.state, "config", None)
        cfg = AnimusConfig()
        priv, pub = generate_vapid_keys()
        cfg.services.vapid_private_key = priv
        cfg.services.vapid_public_key = pub
        app.state.config = cfg

        client = TestClient(app)
        client.get("/health")  # Prime CSRF cookie

        try:
            resp = client.post(
                "/api/push/send-test",
                json={"title": "Prune Test", "body": "B"},
                headers=_csrf_headers(client),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["sent"] == 0
            assert data["pruned"] == 1
            assert store.count() == 0
        finally:
            server.shutdown()
            store.close()
            if original_config is not None:
                app.state.config = original_config
