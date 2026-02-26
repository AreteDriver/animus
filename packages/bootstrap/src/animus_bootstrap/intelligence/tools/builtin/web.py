"""Web tools — search and fetch."""

from __future__ import annotations

import logging
import re

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS  # type: ignore[import-untyped]

    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False


async def _web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    if not HAS_DUCKDUCKGO:
        return (
            f"[web_search] Would search for: {query}\n"
            "Install duckduckgo-search for live results: pip install duckduckgo-search"
        )
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for: {query}"
        lines: list[str] = []
        for r in results:
            lines.append(f"- {r.get('title', 'No title')}: {r.get('href', '')}")
            body = r.get("body", "")
            if body:
                lines.append(f"  {body[:200]}")
        return "\n".join(lines)
    except (ConnectionError, TimeoutError, ValueError, RuntimeError) as exc:
        logger.warning("web_search failed: %s", exc)
        return f"Search failed: {exc}"


async def _web_fetch(url: str) -> str:
    """Fetch a URL and return stripped text content."""
    import httpx

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Fetch failed: {exc}"

    # Strip HTML tags with a simple regex
    text = re.sub(r"<[^>]+>", " ", resp.text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]


def get_web_tools() -> list[ToolDefinition]:
    """Return web-related tool definitions."""
    return [
        ToolDefinition(
            name="web_search",
            description="Search the web for information using DuckDuckGo.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
            handler=_web_search,
            category="web",
        ),
        ToolDefinition(
            name="web_fetch",
            description="Fetch a URL and return its text content (HTML stripped).",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                },
                "required": ["url"],
            },
            handler=_web_fetch,
            category="web",
        ),
    ]
