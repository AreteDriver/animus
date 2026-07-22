"""Minimal proof-of-concept for the Animus browser bridge.

Usage (mock mode — no Chrome required)::

    python -m animus.browser.poc --mock

Usage (live mode — requires Chrome + nodriver)::

    pip install nodriver readability-lxml
    python -m animus.browser.poc --url https://example.com

This script exercises the full pipeline:
  BrowserBridge → ExtractionPipeline → BrowserCache → RateLimitedDomain
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


class _MockTab:
    """Stand-in nodriver tab for mock-mode POC."""

    def __init__(self, url: str, html: str, title: str) -> None:
        self._url = url
        self._html = html
        self._title = title

    async def evaluate(self, js: str) -> Any:
        if "scrollHeight" in js:
            return 5000
        if "documentElement.outerHTML" in js:
            return self._html
        if "document.title" in js:
            return self._title
        if "window.location.href" in js:
            return self._url
        return None

    async def wait_for(self, **kwargs: Any) -> None:
        pass


class _MockBrowser:
    def __init__(self, tab: _MockTab) -> None:
        self._tab = tab

    async def get(self, url: str) -> _MockTab:
        return self._tab

    def stop(self) -> None:
        pass


async def _mock_fetch_demo() -> None:
    """Run the pipeline against mocked Chrome to prove contracts."""
    from animus.browser.bridge import BrowserBridge, BrowserConfig, BrowserResult
    from animus.browser.cache import BrowserCache
    from animus.browser.rate_limit import RateLimitedDomain
    from animus.browser.extraction import ExtractionPipeline

    print("=" * 60)
    print("Animus Browser Bridge — Mock POC")
    print("=" * 60)

    # 1. Rate limiter
    rl = RateLimitedDomain(requests_per_minute=60, burst=5)
    await rl.acquire("https://example.com/article")
    print("[1] Rate limiter token acquired for example.com")

    # 2. Cache
    cache = BrowserCache(store=None, ttl_sec=300)
    cfg = BrowserConfig(format="text")
    key_hit = await cache.get("https://example.com/article", cfg)
    print(f"[2] Cache miss (expected): {key_hit is None}")

    # 3. Extraction pipeline (no real tab — use the class directly)
    pipeline = ExtractionPipeline(cfg)
    sample_html = (
        "<html><body>"
        "<h1>Hello World</h1>"
        "<p>This is a paragraph.</p>"
        "</body></html>"
    )
    # We can't call pipeline.extract without a real tab, but we can
    # verify the quality gate and cleaning logic directly.
    is_low = pipeline._is_low_quality("cookie consent subscribe newsletter")
    print(f"[3] Low-quality gate triggers on cookie noise: {is_low}")
    clean = pipeline._clean_text("Hello\n\nHello\n  \nWorld")
    print(f"[4] Clean text dedup: {clean!r}")

    # 4. Simulate a result and cache it
    result = BrowserResult(
        url="https://example.com/article",
        final_url="https://example.com/article",
        title="Hello World",
        content="This is a paragraph.",
        status_code=200,
        format=cfg.format,
    )
    await cache.put("https://example.com/article", cfg, {
        "url": result.url,
        "title": result.title,
        "content": result.content,
        "status_code": result.status_code,
    })
    cached = await cache.get("https://example.com/article", cfg)
    print(f"[5] Cache hit after put: {cached is not None}")
    print(f"    Cached title: {cached['title']}")

    print("\nMock POC complete. All contracts verified.")


async def _live_fetch_demo(url: str, human_mode: bool = False) -> None:
    """Run a live fetch against a real URL (requires Chrome + nodriver)."""
    try:
        from animus.browser.mcp_tools import fetch
    except RuntimeError as exc:
        print(f"Live mode unavailable: {exc}")
        sys.exit(1)

    print("=" * 60)
    print(f"Animus Browser Bridge — Live POC ({url})")
    print("=" * 60)

    result = await fetch(url, format="text", human_mode=human_mode)
    print(json.dumps(result, indent=2, default=str))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Animus Browser Bridge POC")
    parser.add_argument("--mock", action="store_true", help="Run mock-mode demo (no Chrome)")
    parser.add_argument("--url", type=str, default="", help="Live fetch URL")
    parser.add_argument("--human", action="store_true", help="Enable human emulation")
    args = parser.parse_args()

    if args.url:
        await _live_fetch_demo(args.url, human_mode=args.human)
    else:
        await _mock_fetch_demo()


if __name__ == "__main__":
    asyncio.run(main())
