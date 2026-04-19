"""Tests for animus.lugh.sources.arxiv."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from animus.lugh.sources.arxiv import ArxivSource, default_sources
from animus.lugh.sources.rss import FeedEntry


def _entry(**kw) -> FeedEntry:
    defaults = dict(
        id="http://arxiv.org/abs/2604.14176",
        title="A Paper About Learned Memory",
        link="https://arxiv.org/abs/2604.14176",
        published=datetime(2026, 4, 18, tzinfo=timezone.utc),
        author="A. Researcher",
        summary="arXiv:2604.14176v1 Announce Type: new \nAbstract: We propose... more text here.",
        content="",
        tags=["cs.LG", "cs.AI"],
    )
    defaults.update(kw)
    return FeedEntry(**defaults)


class TestArxivSource:
    def test_source_id(self):
        assert ArxivSource(category="cs.LG").source_id == "arxiv:cs.LG"

    def test_feed_url(self):
        src = ArxivSource(category="cs.AI")
        assert src.feed_url == "http://export.arxiv.org/rss/cs.AI"

    def test_to_item_extracts_id_from_link(self):
        src = ArxivSource(category="cs.LG")
        item = src._to_item(_entry())
        assert item is not None
        assert item.item_id == "2604.14176"
        assert item.source_id == "arxiv:cs.LG"
        assert item.metadata["arxiv_id"] == "2604.14176"
        assert item.metadata["pdf_url"] == "http://arxiv.org/pdf/2604.14176"

    def test_to_item_extracts_id_from_summary_when_other_fields_bare(self):
        src = ArxivSource(category="cs.LG")
        item = src._to_item(
            _entry(link="", id="", summary="arXiv:2604.99999v2 Abstract: Some text.")
        )
        assert item is not None
        assert item.item_id == "2604.99999"

    def test_to_item_strips_announce_prefix_from_summary(self):
        src = ArxivSource(category="cs.LG")
        item = src._to_item(_entry())
        assert item.summary.startswith("We propose")

    def test_to_item_no_id_returns_none(self):
        src = ArxivSource(category="cs.LG")
        assert src._to_item(_entry(link="", id="", summary="no id here")) is None

    def test_fetch_calls_fetch_feed_and_maps(self):
        src = ArxivSource(category="cs.LG")
        entries = [_entry(), _entry(link="https://arxiv.org/abs/2604.00002")]
        with patch("animus.lugh.sources.arxiv.fetch_feed", return_value=entries):
            items = list(src.fetch())
        assert len(items) == 2
        assert {i.item_id for i in items} == {"2604.14176", "2604.00002"}

    def test_fetch_respects_limit(self):
        src = ArxivSource(category="cs.LG")
        entries = [_entry(link=f"https://arxiv.org/abs/2604.{i:05d}") for i in range(10)]
        with patch("animus.lugh.sources.arxiv.fetch_feed", return_value=entries):
            items = list(src.fetch(limit=3))
        assert len(items) == 3


class TestDefaultSources:
    def test_default_sources_nonempty(self):
        srcs = default_sources()
        assert srcs
        assert all(s.source_id.startswith("arxiv:") for s in srcs)

    def test_default_sources_custom_categories(self):
        srcs = default_sources(categories=["cs.LG", "stat.ML"])
        assert [s.category for s in srcs] == ["cs.LG", "stat.ML"]
