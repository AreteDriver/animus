"""Minimal adversarial tests for the browser bridge.

These tests verify contract invariants without requiring a live Chrome
instance.  Integration tests that launch nodriver belong in a separate
CI-gated suite.
"""

from __future__ import annotations

import pytest

from animus.browser.bridge import BrowserConfig, BrowserResult, ExtractionFormat
from animus.browser.cache import BrowserCache, _cache_key
from animus.browser.emulation.scroll import ScrollEmulator
from animus.browser.emulation.timing import TimingEmulator
from animus.browser.rate_limit import RateLimitedDomain


class TestBrowserConfig:
    def test_defaults(self) -> None:
        cfg = BrowserConfig()
        assert cfg.format == ExtractionFormat.TEXT
        assert cfg.timeout_ms == 30_000
        assert cfg.human_mode is False

    def test_format_override(self) -> None:
        cfg = BrowserConfig(format=ExtractionFormat.HTML)
        assert cfg.format == ExtractionFormat.HTML


class TestBrowserResult:
    def test_ok_when_no_error_and_2xx(self) -> None:
        r = BrowserResult(
            url="https://example.com",
            final_url="https://example.com",
            title="OK",
            content="hello",
            status_code=200,
            format=ExtractionFormat.TEXT,
        )
        assert r.ok is True

    def test_not_ok_on_404(self) -> None:
        r = BrowserResult(
            url="https://example.com",
            final_url="https://example.com",
            title="Not Found",
            content="",
            status_code=404,
            format=ExtractionFormat.TEXT,
        )
        assert r.ok is False

    def test_not_ok_on_error(self) -> None:
        r = BrowserResult(
            url="https://example.com",
            final_url="https://example.com",
            title="",
            content="",
            status_code=0,
            format=ExtractionFormat.TEXT,
            error="nodriver not installed",
        )
        assert r.ok is False


class TestBrowserCache:
    def test_cache_key_determinism(self) -> None:
        cfg = BrowserConfig(format=ExtractionFormat.TEXT)
        k1 = _cache_key("https://example.com", cfg.__dict__)
        k2 = _cache_key("https://example.com", cfg.__dict__)
        assert k1 == k2

    def test_cache_key_differs_by_url(self) -> None:
        cfg = BrowserConfig()
        k1 = _cache_key("https://a.com", cfg.__dict__)
        k2 = _cache_key("https://b.com", cfg.__dict__)
        assert k1 != k2

    @pytest.mark.asyncio
    async def test_local_cache_roundtrip(self) -> None:
        cache = BrowserCache(store=None, ttl_sec=3600)
        cfg = BrowserConfig()
        result = {"content": "hello", "title": "Hi"}
        await cache.put("https://example.com", cfg, result)
        got = await cache.get("https://example.com", cfg)
        assert got == result

    @pytest.mark.asyncio
    async def test_local_cache_respects_ttl(self) -> None:
        cache = BrowserCache(store=None, ttl_sec=0)
        cfg = BrowserConfig()
        await cache.put("https://example.com", cfg, {"content": "x"})
        got = await cache.get("https://example.com", cfg)
        assert got is None  # expired immediately


class TestRateLimitedDomain:
    @pytest.mark.asyncio
    async def test_acquire_permits_under_limit(self) -> None:
        rl = RateLimitedDomain(requests_per_minute=60, burst=10)
        # Should not block for first burst requests
        for _ in range(5):
            await rl.acquire("https://example.com/page")

    @pytest.mark.asyncio
    async def test_different_domains_independent(self) -> None:
        rl = RateLimitedDomain(requests_per_minute=60, burst=1)
        # Two different domains should not interfere
        await rl.acquire("https://a.com/1")
        await rl.acquire("https://b.com/1")


class TestEmulation:
    def test_scroll_seed_reproducibility(self) -> None:
        s1 = ScrollEmulator(seed=42)
        s2 = ScrollEmulator(seed=42)
        # Steps should be identical given same seed
        steps1 = [s1._gaussian_step() for _ in range(20)]
        steps2 = [s2._gaussian_step() for _ in range(20)]
        assert steps1 == steps2

    def test_timing_jitter_bounds(self) -> None:
        t = TimingEmulator(seed=42)
        base = 1.0
        for _ in range(50):
            jittered = t._jitter(base)
            assert 0.8 <= jittered <= 1.2  # ±10%

    def test_timing_truncated_gaussian_respects_bounds(self) -> None:
        t = TimingEmulator(seed=42)
        for _ in range(100):
            val = t._truncated_gaussian(1.0, 0.5, 0.5, 2.0)
            assert 0.5 <= val <= 2.0
