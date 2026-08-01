"""
Podcast source — config-driven RSS subscriber.

One `PodcastSource` per configured show. The user points each entry at a
public RSS/Atom feed URL (no guessing — the registry is explicit). We
extract episode title, show notes, publish date, audio URL, and any
iTunes transcript link.

Transcript retrieval is opportunistic: if the feed exposes an
`<itunes:transcript url="...">` link in a format we can handle (srt,
vtt, json), we fetch and attach the plaintext to `raw_text`. Otherwise
`raw_text` falls back to the episode's show notes, which is usually
enough signal for relevance scoring.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

from animus.lugh.sources.base import SourceItem
from animus.lugh.sources.rss import FeedEntry, fetch_feed

logger = logging.getLogger(__name__)

USER_AGENT = "animus-lugh/1.0 (+https://github.com/your-org/animus)"
HTTP_TIMEOUT_SECONDS = 15
TRANSCRIPT_MAX_BYTES = 2_000_000  # 2 MB ceiling on transcript fetches


@dataclass
class PodcastSource:
    """One configured podcast feed.

    Attributes:
        show_id:   short slug, e.g. "all-in", "moonshots", "ai-daily-brief"
        feed_url:  public RSS/Atom URL for the show
        fetch_transcripts: if True, try the itunes:transcript URL when present
    """

    show_id: str
    feed_url: str
    show_name: str = ""
    fetch_transcripts: bool = False

    @property
    def source_id(self) -> str:
        return f"podcast:{self.show_id}"

    def fetch(self, limit: int | None = None) -> Iterable[SourceItem]:
        entries = fetch_feed(self.feed_url)
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            item = self._to_item(entry)
            if item is not None:
                yield item

    def _to_item(self, entry: FeedEntry) -> SourceItem | None:
        if not entry.title:
            return None

        episode_id = entry.id or entry.link
        if not episode_id:
            return None

        show_notes = entry.content or entry.summary
        show_notes_text = _strip_html(show_notes)

        raw_text = show_notes_text
        transcript_text = ""
        transcript_url = entry.extra.get("itunes_transcript_url") if entry.extra else None
        transcript_type = entry.extra.get("itunes_transcript_type") if entry.extra else None

        if self.fetch_transcripts and transcript_url:
            transcript_text = _fetch_transcript(transcript_url, transcript_type or "")
            if transcript_text:
                raw_text = transcript_text

        metadata: dict = {
            "show_id": self.show_id,
            "show_name": self.show_name or self.show_id,
            "audio_url": entry.enclosure_url,
            "audio_type": entry.enclosure_type,
            "has_transcript": bool(transcript_text),
            "transcript_source_url": transcript_url,
        }
        if entry.extra.get("itunes_duration"):
            metadata["duration"] = entry.extra["itunes_duration"]
        if entry.extra.get("itunes_episode"):
            metadata["episode_number"] = entry.extra["itunes_episode"]

        return SourceItem(
            source_id=self.source_id,
            item_id=episode_id,
            title=entry.title,
            url=entry.link,
            published=entry.published,
            summary=_truncate(show_notes_text, 500),
            author=entry.author or self.show_name or None,
            tags=entry.tags,
            raw_text=raw_text,
            metadata=metadata,
        )


def probe_feed(feed_url: str) -> dict:
    """Quick introspection of a feed URL — for registry/CLI use.

    Returns {ok: bool, entry_count: int, sample_titles: [...], error: str?}.
    Does not raise. Intended for `lugh sources probe <url>` before adding.
    """
    try:
        entries = fetch_feed(feed_url)
    except Exception as e:  # pragma: no cover — fetch_feed is already fail-open
        return {"ok": False, "entry_count": 0, "sample_titles": [], "error": str(e)}
    return {
        "ok": bool(entries),
        "entry_count": len(entries),
        "sample_titles": [e.title for e in entries[:3]],
        "error": "" if entries else "empty or unrecognized feed",
    }


def _fetch_transcript(url: str, mime_type: str) -> str:
    """Fetch and normalize a podcast transcript. Returns plaintext or ''."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            body = resp.read(TRANSCRIPT_MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        logger.warning("transcript HTTP %s on %s: %s", e.code, url, e.reason)
        return ""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("transcript transport error on %s: %s", url, e)
        return ""

    if len(body) > TRANSCRIPT_MAX_BYTES:
        logger.warning("transcript too large (%d bytes) on %s — skipping", len(body), url)
        return ""

    text = body.decode("utf-8", errors="replace")
    lowered_type = (mime_type or "").lower()
    if "srt" in lowered_type or "vtt" in lowered_type or url.endswith((".srt", ".vtt")):
        return _strip_cue_lines(text)
    if "json" in lowered_type or url.endswith(".json"):
        return _flatten_json_transcript(text)
    # Default: treat as plaintext.
    return text.strip()


def _strip_cue_lines(text: str) -> str:
    """Remove SRT/VTT cue numbers, timestamps, and metadata; keep spoken text."""
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.isdigit():  # cue index
            continue
        if "-->" in s:  # timestamp
            continue
        if s.startswith(("WEBVTT", "NOTE ", "STYLE ", "REGION ")):
            continue
        out.append(s)
    return " ".join(out)


def _flatten_json_transcript(text: str) -> str:
    """Best-effort plaintext extraction from podcast-index transcript JSON."""
    import json

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()
    if isinstance(data, dict):
        segments = data.get("segments") or data.get("transcripts") or []
    elif isinstance(data, list):
        segments = data
    else:
        return text.strip()
    parts: list[str] = []
    for seg in segments:
        if isinstance(seg, dict):
            body = seg.get("body") or seg.get("text") or seg.get("utterance") or ""
            if body:
                parts.append(str(body).strip())
    return " ".join(parts).strip()


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"
