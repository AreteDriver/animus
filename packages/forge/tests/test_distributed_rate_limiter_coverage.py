"""Coverage for distributed_rate_limiter.py gaps on 2c8cde2.

Targets:
- Line 112: lazy-load of async Redis client via `aioredis.from_url`
- Lines 158-166: RedisRateLimiter.reset() scan + delete loop
- Lines 226-228: SQLiteRateLimiter._get_connection except/rollback/raise
"""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.workflow.distributed_rate_limiter import (
    RedisRateLimiter,
    SQLiteRateLimiter,
)


class TestRedisRateLimiterAsyncClientLazy:
    """Line 112: first call to _get_async_client actually builds a client
    via redis.asyncio.from_url — previous tests pre-seed _async_client."""

    @pytest.mark.asyncio
    async def test_lazy_builds_client_on_first_use(self):
        """The forge test env doesn't have redis installed, so stub the
        `redis.asyncio` module via sys.modules before the lazy import runs."""
        import sys as _sys
        import types as _types

        fake_client = MagicMock()
        fake_aioredis = _types.SimpleNamespace(
            from_url=AsyncMock(return_value=fake_client)
        )
        fake_redis_pkg = _types.ModuleType("redis")
        fake_redis_pkg.asyncio = fake_aioredis  # type: ignore[attr-defined]

        with patch.dict(
            _sys.modules,
            {"redis": fake_redis_pkg, "redis.asyncio": fake_aioredis},
        ):
            limiter = RedisRateLimiter(url="redis://fake-host:6379")
            out = await limiter._get_async_client()
            assert out is fake_client
            fake_aioredis.from_url.assert_awaited_once_with("redis://fake-host:6379")
            # Second call returns cached — no re-import, no re-from_url.
            out2 = await limiter._get_async_client()
            assert out2 is fake_client
            assert fake_aioredis.from_url.await_count == 1

    @pytest.mark.asyncio
    async def test_import_error_when_redis_missing(self):
        """Lines 113-114: ImportError fallback when redis package absent."""
        import sys as _sys

        # Stash any installed redis modules and force ImportError on lookup.
        saved = {k: _sys.modules.pop(k, None) for k in ("redis", "redis.asyncio")}
        try:
            with patch.dict(_sys.modules, {"redis.asyncio": None}):
                limiter = RedisRateLimiter(url="redis://x")
                with pytest.raises(ImportError, match="Redis package not installed"):
                    await limiter._get_async_client()
        finally:
            for k, v in saved.items():
                if v is not None:
                    _sys.modules[k] = v


class TestRedisRateLimiterReset:
    """Lines 158-166: reset() iterates scan + delete until cursor=0."""

    @pytest.mark.asyncio
    async def test_reset_walks_scan_and_deletes_keys(self):
        limiter = RedisRateLimiter(url="redis://fake:6379", prefix="rl:")

        # First scan returns some keys with nonzero cursor; second returns []
        # with cursor=0 to break the loop.
        client = MagicMock()
        client.scan = AsyncMock(
            side_effect=[
                (42, [b"rl:key1:123", b"rl:key1:456"]),
                (0, []),
            ]
        )
        client.delete = AsyncMock(return_value=2)
        limiter._async_client = client

        await limiter.reset("key1")

        assert client.scan.await_count == 2
        client.delete.assert_awaited_once()
        # The delete call unpacks the keys from the first scan batch.
        delete_args = client.delete.await_args.args
        assert b"rl:key1:123" in delete_args
        assert b"rl:key1:456" in delete_args

    @pytest.mark.asyncio
    async def test_reset_handles_empty_scan(self):
        """When scan yields no keys, delete is never called."""
        limiter = RedisRateLimiter(url="redis://fake:6379")
        client = MagicMock()
        client.scan = AsyncMock(return_value=(0, []))
        client.delete = AsyncMock()
        limiter._async_client = client

        await limiter.reset("nonexistent")
        client.delete.assert_not_called()


class TestSQLiteGetConnectionRollback:
    """Lines 226-228: when the context body raises, the connection rolls
    back and the exception propagates."""

    def test_rollback_on_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rl.db")
            limiter = SQLiteRateLimiter(db_path=db_path)

            captured_conn: dict = {}

            class _Sentinel(Exception):
                pass

            # Trigger _get_connection with a failing body.
            with pytest.raises(_Sentinel):
                with limiter._get_connection() as conn:
                    captured_conn["conn"] = conn
                    # Verify we have a real sqlite3 connection.
                    assert isinstance(conn, sqlite3.Connection)
                    raise _Sentinel("body boom")

            # Conn was closed in finally; attempting to use it raises.
            with pytest.raises(sqlite3.ProgrammingError):
                captured_conn["conn"].execute("SELECT 1")
