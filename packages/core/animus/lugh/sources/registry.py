"""
Source registry — declarative catalog of configured lugh sources.

Stored as JSON at ~/.animus/lugh_sources.json. Each entry is one of:

    {"kind": "arxiv", "category": "cs.LG"}
    {"kind": "hn", "query_id": "front_page", "tags": "front_page", "min_points": 0}
    {"kind": "podcast", "show_id": "all-in",
     "feed_url": "https://...", "show_name": "All-In",
     "fetch_transcripts": false}
    {"kind": "youtube", "channel": "@AIDailyBrief",
     "show_name": "The AI Daily Brief", "fetch_captions": true,
     "tags": ["ai-news", "core"]}

`load_registry()` materializes these into Source instances. On first run
we seed the default arXiv categories + HN queries + the curated YouTube
channel list — podcasts start empty (the user adds real RSS feed URLs via
`lugh sources add-podcast` rather than us guessing), but YouTube handles
are stable and curated, so they seed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus.lugh.sources.arxiv import DEFAULT_CATEGORIES, ArxivSource
from animus.lugh.sources.base import Source
from animus.lugh.sources.hn import HackerNewsSource
from animus.lugh.sources.hn import default_sources as _default_hn
from animus.lugh.sources.mer import MERSource
from animus.lugh.sources.podcasts import PodcastSource
from animus.lugh.sources.youtube import DEFAULT_CHANNELS, YouTubeSource

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("~/.animus/lugh_sources.json").expanduser()


def default_registry() -> list[dict]:
    """Seed catalog for first-time users. Podcasts intentionally empty."""
    entries: list[dict] = []
    for cat in DEFAULT_CATEGORIES:
        entries.append({"kind": "arxiv", "category": cat})
    for hn in _default_hn():
        entries.append(
            {
                "kind": "hn",
                "query_id": hn.query_id,
                "query": hn.query,
                "tags": hn.tags,
                "min_points": hn.min_points,
                "hits_per_page": hn.hits_per_page,
            }
        )
    for channel, show_name, fetch_captions, tags in DEFAULT_CHANNELS:
        entries.append(
            {
                "kind": "youtube",
                "channel": channel,
                "show_name": show_name,
                "fetch_captions": fetch_captions,
                "tags": list(tags),
            }
        )
    # One MER tracker — there's a single Monthly Economic Report series.
    entries.append({"kind": "mer", "lookback_months": 3, "tags": ["eve-economy", "eve"]})
    return entries


def load_registry(path: Path | None = None) -> list[dict]:
    """Read the JSON registry, seeding defaults if the file doesn't exist."""
    p = path or DEFAULT_REGISTRY_PATH
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        entries = default_registry()
        p.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        logger.info("seeded lugh sources registry at %s (%d entries)", p, len(entries))
        return entries
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning("registry %s invalid JSON (%s); using empty list", p, e)
        return []
    if not isinstance(data, list):
        logger.warning("registry %s must be a JSON array; using empty list", p)
        return []
    return data


def save_registry(entries: list[dict], path: Path | None = None) -> Path:
    p = path or DEFAULT_REGISTRY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return p


def instantiate(entries: list[dict]) -> list[Source]:
    """Convert registry dicts into concrete Source instances. Bad rows skipped."""
    out: list[Source] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("kind")
        try:
            if kind == "arxiv":
                out.append(ArxivSource(category=raw["category"]))
            elif kind == "hn":
                out.append(
                    HackerNewsSource(
                        query_id=raw["query_id"],
                        query=raw.get("query", ""),
                        tags=raw.get("tags", "story"),
                        min_points=int(raw.get("min_points", 0)),
                        hits_per_page=int(raw.get("hits_per_page", 30)),
                    )
                )
            elif kind == "podcast":
                out.append(
                    PodcastSource(
                        show_id=raw["show_id"],
                        feed_url=raw["feed_url"],
                        show_name=raw.get("show_name", ""),
                        fetch_transcripts=bool(raw.get("fetch_transcripts", False)),
                    )
                )
            elif kind == "youtube":
                out.append(
                    YouTubeSource(
                        channel=raw["channel"],
                        show_name=raw.get("show_name", ""),
                        fetch_captions=bool(raw.get("fetch_captions", True)),
                        tags=_as_tag_list(raw.get("tags")),
                    )
                )
            elif kind == "mer":
                out.append(
                    MERSource(
                        lookback_months=int(raw.get("lookback_months", 3)),
                        tags=_as_tag_list(raw.get("tags")) or None,
                    )
                )
            else:
                logger.warning("registry: unknown kind %r — skipping", kind)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("registry: bad entry %r: %s", raw, e)
    return out


def add_podcast(
    show_id: str,
    feed_url: str,
    show_name: str = "",
    fetch_transcripts: bool = False,
    path: Path | None = None,
) -> bool:
    """Append a podcast entry. Returns False if show_id already exists."""
    entries = load_registry(path=path)
    if any(e.get("kind") == "podcast" and e.get("show_id") == show_id for e in entries):
        return False
    entries.append(
        {
            "kind": "podcast",
            "show_id": show_id,
            "feed_url": feed_url,
            "show_name": show_name,
            "fetch_transcripts": fetch_transcripts,
        }
    )
    save_registry(entries, path=path)
    return True


def add_youtube(
    channel: str,
    show_name: str = "",
    fetch_captions: bool = True,
    tags: list[str] | None = None,
    path: Path | None = None,
) -> bool:
    """Append a YouTube channel entry. Returns False if the channel already exists."""
    entries = load_registry(path=path)
    if any(e.get("kind") == "youtube" and e.get("channel") == channel for e in entries):
        return False
    entries.append(
        {
            "kind": "youtube",
            "channel": channel,
            "show_name": show_name,
            "fetch_captions": fetch_captions,
            "tags": list(tags) if tags else [],
        }
    )
    save_registry(entries, path=path)
    return True


def remove_source(source_id: str, path: Path | None = None) -> bool:
    """Remove an entry by its fully-qualified source_id. Returns False if absent."""
    entries = load_registry(path=path)
    before = len(entries)
    remaining: list[dict] = []
    for e in entries:
        if _entry_source_id(e) == source_id:
            continue
        remaining.append(e)
    if len(remaining) == before:
        return False
    save_registry(remaining, path=path)
    return True


def _entry_source_id(entry: dict) -> str:
    kind = entry.get("kind")
    if kind == "arxiv":
        return f"arxiv:{entry.get('category', '')}"
    if kind == "hn":
        return f"hn:{entry.get('query_id', '')}"
    if kind == "podcast":
        return f"podcast:{entry.get('show_id', '')}"
    if kind == "youtube":
        return f"youtube:{entry.get('channel', '')}"
    if kind == "mer":
        return "mer"
    return ""


def _as_tag_list(value: object) -> list[str]:
    """Accept a JSON list or a comma-separated string; return a clean tag list."""
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    return []
