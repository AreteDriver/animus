"""
Live web-search source — SearXNG JSON API.

Closes the gap Ted's stack surfaced: Lugh harvests arxiv/hn/podcasts/
youtube/rss but had no live web-search source. This adds one, using a
self-hosted SearXNG instance (sovereign, no vendor key) rather than
Tavily/Brave — consistent with the vendor-neutral stack.

SearXNG exposes a JSON search API:

    GET {base}/search?q=<query>&format=json&time_range=<>&categories=news

Set the instance base URL via ANIMUS_SEARCH_URL (e.g. http://localhost:8888).
If unset, the source fails open (yields nothing) — same fail-open contract
as the other adapters, so the harvest never raises on a missing endpoint.

The stable item_id is the result URL (SearXNG has no stable result id),
so the same link surfacing across runs dedups via the SourceCache.

Stdlib-only. Fail-open on I/O and on missing config.

To wire a Tavily/Brave backend instead, swap _fetch_searxng + _to_item for
that API's shape — the SourceItem contract downstream is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from animus.lugh.sources.base import SourceItem

logger = logging.getLogger(__name__)

_ENV_BASE_URL = "ANIMUS_SEARCH_URL"
HTTP_TIMEOUT_SECONDS = 20
USER_AGENT = "animus-lugh/1.0 (+https://github.com/AreteDriver/animus)"


def _resolve_base_url() -> str | None:
    """SearXNG base URL from env, trailing slash stripped. None if unset."""
    raw = os.environ.get(_ENV_BASE_URL, "").strip()
    return raw.rstrip("/") or None


@dataclass
class WebSearchSource:
    """One standing web-search query against a SearXNG instance.

    Examples:
        WebSearchSource(query_id="agent_runtime",
                        query='"agent runtime" OR "MCP server"')
        WebSearchSource(query_id="ai_news", query="AI agents",
                        categories="news", time_range="week")
    """

    query_id: str  # unique key among configured search sources
    query: str  # the search string (required)
    categories: str = ""  # SearXNG categories, e.g. "news", "science"
    time_range: str = ""  # "", "day", "week", "month", "year"
    language: str = "en"
    max_results: int = 20
    tags: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        return f"search:{self.query_id}"

    def request_url(self, base_url: str) -> str:
        params: dict[str, str] = {
            "q": self.query,
            "format": "json",
            "language": self.language,
        }
        if self.categories:
            params["categories"] = self.categories
        if self.time_range:
            params["time_range"] = self.time_range
        return f"{base_url}/search?{urllib.parse.urlencode(params)}"

    def fetch(self, limit: int | None = None) -> Iterable[SourceItem]:
        base_url = _resolve_base_url()
        if base_url is None:
            logger.warning(
                "%s unset — web-search source %s skipped (fail-open)",
                _ENV_BASE_URL,
                self.source_id,
            )
            return
        results = _fetch_searxng(self.request_url(base_url))
        cap = limit if limit is not None else self.max_results
        for hit in results[:cap]:
            item = self._to_item(hit)
            if item is not None:
                yield item

    def _to_item(self, hit: dict) -> SourceItem | None:
        url = hit.get("url")
        title = hit.get("title") or ""
        if not url or not title:
            return None

        content = hit.get("content") or ""
        summary = _truncate(_strip_html(content), 400)
        published = _parse_date(hit.get("publishedDate"))
        engine = hit.get("engine") or ""

        metadata = {
            "engine": engine,
            "engines": list(hit.get("engines") or []),
            "score": hit.get("score"),
            "query": self.query,
            "category": hit.get("category") or self.categories,
        }

        return SourceItem(
            source_id=self.source_id,
            item_id=url,  # URL is the stable dedup key for web results
            title=title,
            url=url,
            published=published,
            summary=summary,
            author=None,
            tags=[*self.tags, *([engine] if engine else [])],
            raw_text=content,
            metadata=metadata,
        )


def default_sources() -> list[WebSearchSource]:
    """Minimal seed catalog. User extends via the registry.

    Intentionally small — search queries are personal. Seeded only when an
    ANIMUS_SEARCH_URL endpoint is configured; otherwise they fail open.
    """
    return [
        WebSearchSource(
            query_id="agent_infra",
            query='"agent runtime" OR "MCP server" OR "tool calling"',
            categories="news",
            time_range="week",
            tags=["ai-infra", "core"],
        ),
        WebSearchSource(
            query_id="local_llm",
            query="local LLM inference OR open weights OR quantization",
            time_range="week",
            tags=["local-llm"],
        ),
    ]


def _fetch_searxng(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.warning("SearXNG HTTP %s on %s: %s", e.code, url, e.reason)
        return []
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.warning("SearXNG error on %s: %s", url, e)
        return []
    return list(payload.get("results") or [])


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _strip_html(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"
