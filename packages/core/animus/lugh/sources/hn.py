"""
Hacker News source — Algolia API queries.

HN's Algolia API is public, fast, and needs no auth:

    https://hn.algolia.com/api/v1/search_by_date?tags=front_page&hitsPerPage=50
    https://hn.algolia.com/api/v1/search?query=AI+memory&tags=story&hitsPerPage=20

Docs: https://hn.algolia.com/api

We expose one `HackerNewsSource` per configured query. The stable `item_id`
is the Algolia `objectID` (same as HN story id). Fail-open on network errors.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from animus.lugh.sources.base import SourceItem

logger = logging.getLogger(__name__)

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HTTP_TIMEOUT_SECONDS = 15
USER_AGENT = "animus-lugh/1.0 (+https://github.com/AreteDriver/animus)"


@dataclass
class HackerNewsSource:
    """One HN Algolia query.

    Examples:
        HackerNewsSource(query_id="front_page", tags="front_page")
        HackerNewsSource(query_id="ai_memory", query="AI memory", tags="story")
        HackerNewsSource(query_id="show_hn", tags="show_hn")
    """

    query_id: str  # unique key among configured HN sources, e.g. "front_page"
    query: str = ""  # empty = no keyword filter (just tags)
    tags: str = "story"  # "front_page", "story", "show_hn", "ask_hn", "poll"
    min_points: int = 0  # minimum HN score to keep
    hits_per_page: int = 30
    numeric_filters: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        return f"hn:{self.query_id}"

    @property
    def request_url(self) -> str:
        params: dict[str, str] = {
            "tags": self.tags,
            "hitsPerPage": str(self.hits_per_page),
        }
        if self.query:
            params["query"] = self.query
        if self.min_points > 0:
            filters = [*self.numeric_filters, f"points>={self.min_points}"]
        else:
            filters = self.numeric_filters
        if filters:
            params["numericFilters"] = ",".join(filters)
        return f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"

    def fetch(self, limit: int | None = None) -> Iterable[SourceItem]:
        hits = _fetch_algolia(self.request_url)
        if limit is not None:
            hits = hits[:limit]
        for hit in hits:
            item = self._to_item(hit)
            if item is not None:
                yield item

    def _to_item(self, hit: dict) -> SourceItem | None:
        object_id = hit.get("objectID")
        if not object_id:
            return None

        title = hit.get("title") or hit.get("story_title") or ""
        if not title:
            return None

        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        hn_url = f"https://news.ycombinator.com/item?id={object_id}"
        created_at = hit.get("created_at")
        published = _parse_algolia_date(created_at)

        author = hit.get("author")
        points = int(hit.get("points") or 0)
        num_comments = int(hit.get("num_comments") or 0)
        tags = list(hit.get("_tags") or [])

        story_text = hit.get("story_text") or ""
        summary_source = story_text or title
        summary = _truncate(_strip_html(summary_source), 400)

        metadata = {
            "hn_id": object_id,
            "hn_url": hn_url,
            "external_url": url,
            "points": points,
            "num_comments": num_comments,
        }

        return SourceItem(
            source_id=self.source_id,
            item_id=str(object_id),
            title=title,
            url=url,
            published=published,
            summary=summary,
            author=author,
            tags=tags,
            raw_text=story_text,
            metadata=metadata,
        )


def default_sources() -> list[HackerNewsSource]:
    """A minimal default catalog — user can extend via registry."""
    return [
        HackerNewsSource(query_id="front_page", tags="front_page", hits_per_page=30),
        HackerNewsSource(query_id="show_hn", tags="show_hn", min_points=10, hits_per_page=30),
        HackerNewsSource(
            query_id="ai_research",
            query='LLM OR transformer OR "AI agent"',
            tags="story",
            min_points=20,
            hits_per_page=30,
        ),
    ]


def _fetch_algolia(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.warning("HN Algolia HTTP %s on %s: %s", e.code, url, e.reason)
        return []
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.warning("HN Algolia error on %s: %s", url, e)
        return []
    return list(payload.get("hits") or [])


def _parse_algolia_date(raw: str | None) -> datetime | None:
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
