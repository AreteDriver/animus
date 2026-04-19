"""
arXiv source — category RSS feeds.

arXiv publishes RSS 2.0 feeds per category at
`http://export.arxiv.org/rss/<category>` (e.g., cs.LG, cs.CL, cs.AI, cs.IR).
Each feed carries the day's new submissions. We map each RSS item into a
`SourceItem` with the paper's arXiv id as the stable `item_id`.

No auth, no paging; the feed returns ~N papers/day per category.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from animus.lugh.sources.base import SourceItem
from animus.lugh.sources.rss import fetch_feed

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("cs.CL", "cs.LG", "cs.AI", "cs.IR")
FEED_URL_TEMPLATE = "http://export.arxiv.org/rss/{category}"

# Matches "arXiv:2604.14176v1" or "arxiv.org/abs/2604.14176" (with optional version).
_ARXIV_ID_RE = re.compile(r"(?:arXiv:|/abs/)([0-9]{4}\.[0-9]{4,6})(v\d+)?", re.IGNORECASE)


@dataclass
class ArxivSource:
    """One arXiv category feed."""

    category: str

    @property
    def source_id(self) -> str:
        return f"arxiv:{self.category}"

    @property
    def feed_url(self) -> str:
        return FEED_URL_TEMPLATE.format(category=self.category)

    def fetch(self, limit: int | None = None) -> Iterable[SourceItem]:
        entries = fetch_feed(self.feed_url)
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            item = self._to_item(entry)
            if item is not None:
                yield item

    def _to_item(self, entry) -> SourceItem | None:
        arxiv_id = (
            _extract_arxiv_id(entry.link)
            or _extract_arxiv_id(entry.id)
            or _extract_arxiv_id(entry.summary)
            or _extract_arxiv_id(entry.content)
        )
        if not arxiv_id:
            logger.debug("arxiv: no id in entry %r", entry.link)
            return None

        abstract = entry.content or entry.summary
        abstract = _strip_announce_prefix(abstract)
        pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

        metadata = {
            "arxiv_id": arxiv_id,
            "primary_category": self.category,
            "pdf_url": pdf_url,
        }

        return SourceItem(
            source_id=self.source_id,
            item_id=arxiv_id,
            title=entry.title.strip().rstrip("."),
            url=entry.link,
            published=entry.published,
            summary=_first_paragraph(abstract, limit=500),
            author=entry.author,
            tags=[self.category, *entry.tags],
            raw_text=abstract,
            metadata=metadata,
        )


def default_sources(categories: Iterable[str] = DEFAULT_CATEGORIES) -> list[ArxivSource]:
    """Instantiate ArxivSource for each requested category."""
    return [ArxivSource(category=c) for c in categories]


def _extract_arxiv_id(text: str) -> str | None:
    if not text:
        return None
    m = _ARXIV_ID_RE.search(text)
    return m.group(1) if m else None


def _strip_announce_prefix(text: str) -> str:
    """arXiv RSS descriptions start with 'arXiv:<id>v1 Announce Type: ...\\nAbstract:'."""
    if not text:
        return text
    m = re.search(r"Abstract:\s*", text)
    return text[m.end() :] if m else text


def _first_paragraph(text: str, limit: int = 500) -> str:
    if not text:
        return ""
    # Strip trivial HTML tags that arXiv occasionally includes (<p>, <abstract>).
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "…"
