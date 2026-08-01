"""MCP tool definitions for browser fetch.

Exposes ``fetch`` and ``fetch_batch`` to the LLM toolchain.
Tools are registered via the existing animus MCP server plumbing.
"""

from __future__ import annotations

from typing import Any

from animus.browser.bridge import BrowserBridge, BrowserConfig, BrowserResult
from animus.browser.cache import BrowserCache
from animus.browser.rate_limit import RateLimitedDomain
from animus.logging import get_logger

logger = get_logger("browser.mcp_tools")

# Global singletons (lazy-init per process)
_bridge: BrowserBridge | None = None
_cache: BrowserCache | None = None
_rate_limiter: RateLimitedDomain | None = None


def _lazy_init() -> tuple[BrowserBridge, BrowserCache, RateLimitedDomain]:
    """Initialise shared bridge, cache, and rate limiter."""
    global _bridge, _cache, _rate_limiter
    if _bridge is None:
        _bridge = BrowserBridge()
    if _cache is None:
        # Attempt to wire DurableObjectStore if available
        store = None
        try:
            from animus.durability.postgres_store import DurableObjectStore

            if hasattr(DurableObjectStore, "__init__"):
                store = DurableObjectStore()  # type: ignore[misc]
        except Exception:
            pass
        _cache = BrowserCache(store=store)
    if _rate_limiter is None:
        _rate_limiter = RateLimitedDomain()
    return _bridge, _cache, _rate_limiter


async def fetch(
    url: str,
    format: str = "text",
    wait_for: str | None = None,
    timeout: int = 30_000,
    human_mode: bool = False,
) -> dict[str, Any]:
    """Fetch a single URL via real browser.

    Args:
        url: Target URL to fetch.
        format: One of ``text``, ``markdown``, ``html``. Defaults to ``text``.
        wait_for: Optional CSS selector to wait for before extraction.
        timeout: Maximum time in milliseconds (default 30_000).
        human_mode: Enable anti-detection emulation (slower, more robust).

    Returns:
        JSON object with ``url``, ``title``, ``content``, ``status_code``,
        ``ok``, ``error``, ``fetch_time_sec``.
    """
    bridge, cache, rl = _lazy_init()
    await rl.acquire(url)

    cfg = BrowserConfig(
        format=format,  # type: ignore[arg-type]
        timeout_ms=timeout,
        wait_for=wait_for,
        human_mode=human_mode,
    )

    # Cache check
    cached = await cache.get(url, cfg)
    if cached:
        cached["cache_hit"] = True
        return cached

    result: BrowserResult = await bridge.fetch(url, config=cfg)
    payload = {
        "url": result.url,
        "final_url": result.final_url,
        "title": result.title,
        "content": result.content,
        "status_code": result.status_code,
        "ok": result.ok,
        "error": result.error,
        "fetch_time_sec": result.fetch_time_sec,
        "used_human_mode": result.used_human_mode,
        "cache_hit": False,
    }
    await cache.put(url, cfg, payload)
    return payload


async def fetch_batch(urls: list[str], format: str = "text") -> list[dict[str, Any]]:
    """Fetch multiple URLs in parallel.

    Args:
        urls: List of target URLs (max 14).
        format: Shared output format (``text``, ``markdown``, ``html``).

    Returns:
        List of result objects, same schema as ``fetch``.
    """
    if len(urls) > 14:
        raise ValueError("fetch_batch supports at most 14 URLs")

    import asyncio

    tasks = [fetch(u, format=format) for u in urls]
    return await asyncio.gather(*tasks)


# JSON Schema surfaces for LLM tool registration
FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "format": {"type": "string", "enum": ["text", "markdown", "html"]},
        "wait_for": {"type": "string"},
        "timeout": {"type": "integer", "default": 30000},
        "human_mode": {"type": "boolean", "default": False},
    },
    "required": ["url"],
}

FETCH_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 14,
        },
        "format": {"type": "string", "enum": ["text", "markdown", "html"]},
    },
    "required": ["urls"],
}
