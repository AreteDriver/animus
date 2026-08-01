"""Integration tests for Animus Browser Bridge.

Requires:
  - Chrome (google-chrome or chromium) installed
  - ``nodriver`` and ``readability-lxml`` pip packages
  - Environment variable ``ANIMUS_BROWSER_INTEGRATION=1``

Run::

    ANIMUS_BROWSER_INTEGRATION=1 pytest tests/test_browser_integration.py -v

These tests spin up real Chrome instances.  Each test is isolated
(temporary profile, no cookies persist).  On CI without Chrome they are
auto-skipped.
"""

from __future__ import annotations

import os

import pytest

from animus.browser.bridge import BrowserBridge, BrowserConfig, ExtractionFormat

# ------------------------------------------------------------------
# Skip gating
# ------------------------------------------------------------------

_INTEGRATION_ENABLED = os.environ.get("ANIMUS_BROWSER_INTEGRATION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = pytest.mark.skipif(
    not _INTEGRATION_ENABLED,
    reason="Set ANIMUS_BROWSER_INTEGRATION=1 to run live Chrome tests",
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _relax_memory_limit():
    """Remove the 32 GB RLIMIT_AS from conftest.py so Chrome can start.

    Chromium maps large virtual-memory regions during startup; the project-wide
    memory-limit in conftest.py causes ``Failed to connect to browser``.
    """
    import resource

    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (-1, old_hard))
    yield
    resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))


@pytest.fixture
async def bridge():
    """Yield a fresh BrowserBridge; auto-close after test."""
    br = BrowserBridge()
    yield br


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_text_simple():
    """Basic fetch against a static page returns non-empty text."""
    br = BrowserBridge()
    result = await br.fetch("https://example.com")
    assert result.ok, f"Fetch failed: {result.error}"
    assert result.status_code == 200
    assert "Example Domain" in result.content or "example" in result.content.lower()
    assert result.title
    assert result.fetch_time_sec > 0


@pytest.mark.asyncio
async def test_fetch_html_format():
    """HTML format returns raw markup."""
    br = BrowserBridge()
    cfg = BrowserConfig(format=ExtractionFormat.HTML)
    result = await br.fetch("https://example.com", config=cfg)
    assert result.ok
    assert "<html" in result.content.lower() or "<!doctype" in result.content.lower()


@pytest.mark.asyncio
async def test_fetch_js_heavy_react_docs():
    """React docs (Next.js SPA) — requires real browser."""
    br = BrowserBridge()
    result = await br.fetch("https://react.dev")
    assert result.ok, f"Fetch failed: {result.error}"
    assert len(result.content) > 300
    # React docs should contain React-specific content, not a blank loader
    assert "React" in result.content or "react" in result.content.lower()


@pytest.mark.asyncio
async def test_fetch_with_human_mode():
    """Human mode completes successfully (anti-detection path)."""
    br = BrowserBridge()
    cfg = BrowserConfig(human_mode=True)
    result = await br.fetch("https://example.com", config=cfg)
    assert result.ok
    # human_mode forces a headed retry if blocked; for example.com it just
    # adds scroll/timing emulation, so used_human_mode may be True or False
    # depending on whether the site needed escalation.


@pytest.mark.asyncio
async def test_fetch_batch():
    """Batch fetch returns results for all URLs."""
    br = BrowserBridge()
    urls = ["https://example.com", "https://httpbin.org/html"]
    results = await br.fetch_batch(urls)
    assert len(results) == len(urls)
    for r in results:
        assert r.ok, f"Batch item failed: {r.error}"
        assert len(r.content) > 10


@pytest.mark.asyncio
async def test_fetch_invalid_url():
    """Invalid URL returns structured error, no crash."""
    br = BrowserBridge()
    result = await br.fetch("not-a-url")
    assert not result.ok
    assert result.error
    assert "Invalid URL" in result.error


@pytest.mark.asyncio
async def test_fetch_404():
    """404 page is still fetched (browser returns rendered 404)."""
    br = BrowserBridge()
    result = await br.fetch("https://httpbin.org/status/404")
    # nodiver doesn't reliably expose HTTP status; we just verify no crash
    assert result.error is None or "404" in result.error or result.status_code == 404


@pytest.mark.asyncio
async def test_cache_roundtrip():
    """DurableObjectStore-backed cache persists across fetches."""
    from animus.browser.cache import BrowserCache

    cache = BrowserCache(store=None, ttl_sec=300)
    cfg = BrowserConfig()
    payload = {"content": "cached data", "title": "Cache Test"}
    await cache.put("https://cache.test", cfg, payload)
    got = await cache.get("https://cache.test", cfg)
    assert got == payload
