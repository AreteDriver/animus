"""
Stdlib RSS/Atom feed reader.

Parses RSS 2.0 and Atom 1.0 into a uniform list of `FeedEntry` dicts.
No feedparser, no BeautifulSoup — `xml.etree.ElementTree` + urllib.request.

Timeouts and malformed feeds are handled fail-open: the reader returns
an empty list and logs a warning. Individual entries that fail to parse
are skipped; the rest of the feed still comes through.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "animus-lugh/1.0 (+https://github.com/AreteDriver/animus)"
HTTP_TIMEOUT_SECONDS = 15

# Namespaces we encounter in the wild.
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "media": "http://search.yahoo.com/mrss/",
}


@dataclass
class FeedEntry:
    """One entry from an RSS or Atom feed, normalized."""

    id: str  # GUID for RSS, <id> for Atom; falls back to link
    title: str
    link: str
    published: datetime | None
    author: str | None = None
    summary: str = ""  # <description> / <summary>
    content: str = ""  # <content:encoded> / Atom <content> — richer body
    tags: list[str] = field(default_factory=list)  # <category>
    enclosure_url: str | None = None  # podcast audio / video
    enclosure_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # source-specific fields


def fetch_feed(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> list[FeedEntry]:
    """Fetch and parse an RSS or Atom feed. Returns [] on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        logger.warning("feed HTTP %s on %s: %s", e.code, url, e.reason)
        return []
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("feed transport error on %s: %s", url, e)
        return []

    return parse_feed(body, source_url=url)


def parse_feed(body: bytes, source_url: str = "") -> list[FeedEntry]:
    """Parse feed bytes into FeedEntry list. Returns [] on unrecognized root."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        logger.warning("feed parse error on %s: %s", source_url, e)
        return []

    tag = _localname(root.tag)
    if tag == "rss":
        channel = root.find("channel")
        if channel is None:
            return []
        return [_parse_rss_item(it) for it in channel.findall("item")]
    if tag == "feed":  # Atom
        return [_parse_atom_entry(en) for en in root.findall(f"{{{_NS['atom']}}}entry")]

    logger.warning("unrecognized feed root <%s> on %s", tag, source_url)
    return []


# --------------------------------------------------------------------------- #
# RSS 2.0
# --------------------------------------------------------------------------- #


def _parse_rss_item(item: ET.Element) -> FeedEntry:
    title = _text(item.find("title"))
    link = _text(item.find("link"))
    guid = _text(item.find("guid")) or link
    pub_raw = _text(item.find("pubDate"))
    published = _parse_rfc2822(pub_raw)
    author = _text(item.find("author")) or _text(item.find(f"{{{_NS['dc']}}}creator"))
    summary = _text(item.find("description"))
    content = _text(item.find(f"{{{_NS['content']}}}encoded")) or summary
    tags = [c.text.strip() for c in item.findall("category") if c.text]

    enclosure = item.find("enclosure")
    enc_url = enclosure.get("url") if enclosure is not None else None
    enc_type = enclosure.get("type") if enclosure is not None else None

    extra: dict[str, Any] = {}
    dur = item.find(f"{{{_NS['itunes']}}}duration")
    if dur is not None and dur.text:
        extra["itunes_duration"] = dur.text.strip()
    ep = item.find(f"{{{_NS['itunes']}}}episode")
    if ep is not None and ep.text:
        extra["itunes_episode"] = ep.text.strip()
    transcript = item.find(f"{{{_NS['itunes']}}}transcript")
    if transcript is not None:
        extra["itunes_transcript_url"] = transcript.get("url")
        extra["itunes_transcript_type"] = transcript.get("type")

    return FeedEntry(
        id=guid,
        title=title,
        link=link,
        published=published,
        author=author or None,
        summary=summary,
        content=content,
        tags=tags,
        enclosure_url=enc_url,
        enclosure_type=enc_type,
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Atom 1.0
# --------------------------------------------------------------------------- #


def _parse_atom_entry(entry: ET.Element) -> FeedEntry:
    ns = _NS["atom"]
    atom_id = _text(entry.find(f"{{{ns}}}id"))
    title = _text(entry.find(f"{{{ns}}}title"))

    link_el = entry.find(f"{{{ns}}}link[@rel='alternate']")
    if link_el is None:
        link_el = entry.find(f"{{{ns}}}link")
    link = link_el.get("href") if link_el is not None else ""

    published = _parse_iso(_text(entry.find(f"{{{ns}}}published"))) or _parse_iso(
        _text(entry.find(f"{{{ns}}}updated"))
    )

    author_el = entry.find(f"{{{ns}}}author/{{{ns}}}name")
    author = _text(author_el)

    summary = _text(entry.find(f"{{{ns}}}summary"))
    content = _text(entry.find(f"{{{ns}}}content")) or summary

    tags = [c.get("term", "") for c in entry.findall(f"{{{ns}}}category") if c.get("term")]

    return FeedEntry(
        id=atom_id or link,
        title=title,
        link=link,
        published=published,
        author=author or None,
        summary=summary,
        content=content,
        tags=tags,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(el: ET.Element | None) -> str:
    """Return the full text content of an element, including children's tails."""
    if el is None:
        return ""
    # Mixed content: "foo<b>bar</b>baz" has el.text="foo", one child with
    # text="bar" and tail="baz". itertext() walks the lot.
    return "".join(el.itertext()).strip()


def _parse_rfc2822(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso(raw: str) -> datetime | None:
    if not raw:
        return None
    # Python's fromisoformat() handles "2026-04-18T10:00:00Z" poorly pre-3.11;
    # strip trailing Z and add +00:00 to normalize.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
