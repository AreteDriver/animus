"""Tests for security patterns (request limits and brute force protection)."""

import sys

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

sys.path.insert(0, "src")

from animus_forge.security.brute_force import (
    AttemptRecord,
    BruteForceConfig,
    BruteForceMiddleware,
    BruteForceProtection,
)
from animus_forge.security.request_limits import (
    RequestLimitConfig,
    RequestSizeLimitMiddleware,
    create_size_limit_middleware,
)


class TestRequestLimitConfig:
    """Tests for RequestLimitConfig."""

    def test_default_values(self):
        """Has sensible defaults."""
        config = RequestLimitConfig()
        assert config.max_body_size == 10 * 1024 * 1024  # 10 MB
        assert config.max_json_size == 1 * 1024 * 1024  # 1 MB
        assert config.max_form_size == 50 * 1024 * 1024  # 50 MB

    def test_custom_values(self):
        """Accepts custom configuration."""
        config = RequestLimitConfig(
            max_body_size=5 * 1024 * 1024,
            max_json_size=500 * 1024,
            large_upload_paths=("/api/upload",),
        )
        assert config.max_body_size == 5 * 1024 * 1024
        assert config.max_json_size == 500 * 1024
        assert "/api/upload" in config.large_upload_paths


class TestRequestSizeLimitMiddleware:
    """Tests for RequestSizeLimitMiddleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create test app with size limit middleware."""
        app = FastAPI()
        config = RequestLimitConfig(
            max_body_size=1000,
            max_json_size=500,
            large_upload_paths=("/upload",),
            large_upload_max_size=5000,
        )
        app.add_middleware(RequestSizeLimitMiddleware, config=config)

        @app.post("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.post("/upload")
        async def upload_endpoint():
            return {"status": "uploaded"}

        @app.get("/get")
        async def get_endpoint():
            return {"status": "ok"}

        return app

    def test_allows_small_requests(self, app_with_middleware):
        """Allows requests within size limit."""
        client = TestClient(app_with_middleware)
        response = client.post(
            "/test",
            json={"data": "small"},
        )
        assert response.status_code == 200

    def test_rejects_oversized_requests(self, app_with_middleware):
        """Rejects requests exceeding size limit."""
        client = TestClient(app_with_middleware)
        # Create payload larger than 500 bytes (JSON limit)
        large_data = "x" * 1000
        response = client.post(
            "/test",
            json={"data": large_data},
            headers={"Content-Length": "1500"},
        )
        assert response.status_code == 413
        assert "exceeds maximum size" in response.json()["message"]

    def test_allows_get_requests(self, app_with_middleware):
        """GET requests are not size-limited."""
        client = TestClient(app_with_middleware)
        response = client.get("/get")
        assert response.status_code == 200

    def test_large_upload_paths(self, app_with_middleware):
        """Allows larger payloads on upload paths."""
        client = TestClient(app_with_middleware)
        # 2000 bytes would be rejected on /test but allowed on /upload
        response = client.post(
            "/upload",
            content=b"x" * 2000,
            headers={"Content-Length": "2000"},
        )
        assert response.status_code == 200


class TestCreateSizeLimitMiddleware:
    """Tests for create_size_limit_middleware factory."""

    def test_creates_configured_middleware(self):
        """Factory creates properly configured middleware."""
        middleware_class = create_size_limit_middleware(
            max_body_size=5000,
            max_json_size=1000,
            large_upload_paths=("/api/files",),
        )

        app = FastAPI()

        @app.post("/test")
        async def test():
            return {}

        app.add_middleware(middleware_class)
        client = TestClient(app)

        # Small request should work
        response = client.post("/test", json={"a": "b"})
        assert response.status_code == 200


class TestBruteForceConfig:
    """Tests for BruteForceConfig."""

    def test_default_values(self):
        """Has sensible defaults."""
        config = BruteForceConfig()
        assert config.max_attempts_per_minute == 60
        assert config.max_auth_attempts_per_minute == 5
        assert config.initial_block_seconds == 60.0

    def test_auth_paths(self):
        """Default auth paths are configured."""
        config = BruteForceConfig()
        assert "/auth/" in config.auth_paths
        assert "/token" in config.auth_paths


class TestBruteForceProtection:
    """Tests for BruteForceProtection."""

    def test_allows_normal_requests(self):
        """Allows requests within limits."""
        protection = BruteForceProtection()
        allowed, retry_after = protection.check_allowed("192.168.1.1")
        assert allowed is True
        assert retry_after == 0.0

    def test_blocks_after_limit(self):
        """Blocks after exceeding rate limit."""
        config = BruteForceConfig(
            max_attempts_per_minute=3,
            max_attempts_per_hour=100,  # High enough to not interfere
        )
        protection = BruteForceProtection(config=config)

        # Make allowed requests up to limit
        for i in range(3):
            allowed, _ = protection.check_allowed("192.168.1.1")
            # First 3 should be allowed
            if i < 3:
                assert allowed is True, f"Request {i + 1} should be allowed"

        # Fourth should be blocked
        allowed, retry_after = protection.check_allowed("192.168.1.1")
        assert allowed is False
        assert retry_after > 0

    def test_auth_has_stricter_limits(self):
        """Authentication endpoints have stricter limits."""
        config = BruteForceConfig(
            max_attempts_per_minute=100,
            max_auth_attempts_per_minute=2,
        )
        protection = BruteForceProtection(config=config)

        # Auth requests should be blocked sooner
        for _ in range(2):
            allowed, _ = protection.check_allowed("192.168.1.1", is_auth=True)
            assert allowed is True

        allowed, retry_after = protection.check_allowed("192.168.1.1", is_auth=True)
        assert allowed is False
        assert retry_after > 0

    def test_different_ips_tracked_separately(self):
        """Different IPs have separate limits."""
        config = BruteForceConfig(max_attempts_per_minute=2)
        protection = BruteForceProtection(config=config)

        # IP 1 hits limit
        for _ in range(3):
            protection.check_allowed("192.168.1.1")

        allowed1, _ = protection.check_allowed("192.168.1.1")
        assert allowed1 is False

        # IP 2 should still be allowed
        allowed2, _ = protection.check_allowed("192.168.1.2")
        assert allowed2 is True

    def test_failed_attempts_increase_block(self):
        """Failed attempts lead to longer blocks."""
        config = BruteForceConfig(
            max_auth_attempts_per_minute=5,
            initial_block_seconds=10,
            block_multiplier=2.0,
        )
        protection = BruteForceProtection(config=config)

        # Record some failed attempts
        for _ in range(5):
            protection.record_failed_attempt("192.168.1.1")

        # Should be blocked
        allowed, retry_after = protection.check_allowed("192.168.1.1", is_auth=True)
        assert allowed is False
        assert retry_after >= 10  # At least initial block

    def test_success_reduces_failed_count(self):
        """Successful auth reduces failed attempt counter."""
        protection = BruteForceProtection()

        protection.record_failed_attempt("192.168.1.1")
        protection.record_failed_attempt("192.168.1.1")

        record = protection._attempts["192.168.1.1"]
        initial_failed = record.failed_attempts

        protection.record_success("192.168.1.1")

        assert record.failed_attempts < initial_failed

    def test_exponential_backoff(self):
        """Block duration increases exponentially."""
        config = BruteForceConfig(
            max_auth_attempts_per_minute=1,
            initial_block_seconds=10,
            block_multiplier=2.0,
            max_block_seconds=1000,
        )
        protection = BruteForceProtection(config=config)

        # First block
        protection.check_allowed("192.168.1.1", is_auth=True)
        _, retry1 = protection.check_allowed("192.168.1.1", is_auth=True)

        # Verify block duration calculation
        record = protection._attempts["192.168.1.1"]
        duration = protection._calculate_block_duration(record)
        assert duration >= config.initial_block_seconds

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        protection = BruteForceProtection()

        protection.check_allowed("192.168.1.1")
        protection.check_allowed("192.168.1.2")

        stats = protection.get_stats()
        assert stats["total_allowed"] == 2
        assert stats["tracked_identifiers"] == 2


class TestBruteForceMiddleware:
    """Tests for BruteForceMiddleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create test app with brute force middleware."""
        config = BruteForceConfig(
            max_attempts_per_minute=5,
            max_auth_attempts_per_minute=2,
            auth_paths=("/auth/",),
            initial_block_seconds=60,
        )
        protection = BruteForceProtection(config=config)

        app = FastAPI()
        app.add_middleware(BruteForceMiddleware, protection=protection)

        @app.post("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.post("/auth/login")
        async def auth_endpoint():
            return {"status": "ok"}

        return app, protection

    def test_allows_normal_requests(self, app_with_middleware):
        """Allows requests within limits."""
        app, _ = app_with_middleware
        client = TestClient(app)

        response = client.post("/test")
        assert response.status_code == 200

    def test_blocks_excessive_requests(self, app_with_middleware):
        """Blocks requests exceeding rate limit."""
        app, _ = app_with_middleware
        client = TestClient(app)

        # Make multiple requests to trigger block
        for _ in range(6):
            client.post("/test")

        # Should be blocked
        response = client.post("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_auth_paths_have_stricter_limits(self, app_with_middleware):
        """Auth endpoints are blocked sooner."""
        app, _ = app_with_middleware
        client = TestClient(app)

        # Auth should be blocked after 2 requests
        for _ in range(2):
            response = client.post("/auth/login")
            assert response.status_code == 200

        response = client.post("/auth/login")
        assert response.status_code == 429


class TestAttemptRecord:
    """Tests for AttemptRecord dataclass."""

    def test_default_values(self):
        """Has correct defaults."""
        record = AttemptRecord()
        assert record.attempts == 0
        assert record.blocked_until == 0.0
        assert record.failed_attempts == 0

    def test_mutable_fields(self):
        """Fields can be updated."""
        record = AttemptRecord()
        record.attempts = 5
        record.failed_attempts = 2
        assert record.attempts == 5
        assert record.failed_attempts == 2
