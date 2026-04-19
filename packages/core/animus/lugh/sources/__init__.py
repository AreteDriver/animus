"""
Lugh external-source adapters.

This subpackage extends Lugh beyond GitHub repos + Claude Code transcripts
to ingest research papers, podcasts, HN, and AI-news feeds. Each adapter
produces `SourceItem`s which are deduped into `SourceCache` and optionally
scored for relevance against the user's portfolio.

v1 scope:
    arxiv    — per-category RSS feeds
    hn       — HN Algolia API (planned)
    podcasts — config-driven RSS subscriber (planned)

v2 (not yet wired):
    ChromaDB write path, MCP tool surface, cron integration.
"""

from __future__ import annotations

from animus.lugh.sources.arxiv import ArxivSource
from animus.lugh.sources.arxiv import default_sources as default_arxiv_sources
from animus.lugh.sources.base import Scorer, Source, SourceCache, SourceItem
from animus.lugh.sources.hn import HackerNewsSource
from animus.lugh.sources.hn import default_sources as default_hn_sources
from animus.lugh.sources.podcasts import PodcastSource, probe_feed
from animus.lugh.sources.registry import (
    add_podcast,
    default_registry,
    instantiate,
    load_registry,
    remove_source,
    save_registry,
)
from animus.lugh.sources.relevance import DEFAULT_KEYWORDS, KeywordScorer, default_scorer
from animus.lugh.sources.rss import FeedEntry, fetch_feed, parse_feed

__all__ = [
    # sources
    "ArxivSource",
    "HackerNewsSource",
    "PodcastSource",
    "default_arxiv_sources",
    "default_hn_sources",
    # base
    "FeedEntry",
    "Scorer",
    "Source",
    "SourceCache",
    "SourceItem",
    # RSS
    "fetch_feed",
    "parse_feed",
    # registry
    "add_podcast",
    "default_registry",
    "instantiate",
    "load_registry",
    "probe_feed",
    "remove_source",
    "save_registry",
    # relevance
    "DEFAULT_KEYWORDS",
    "KeywordScorer",
    "default_scorer",
]
