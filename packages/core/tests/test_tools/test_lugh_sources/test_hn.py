"""Tests for animus.lugh.sources.hn — Algolia API client."""

from __future__ import annotations

import json
from unittest.mock import patch

from animus.lugh.sources.hn import HackerNewsSource, default_sources

ALGOLIA_FIXTURE = {
    "hits": [
        {
            "objectID": "40000001",
            "title": "Zero-Copy GPU Inference from WebAssembly",
            "url": "https://example.com/gpu-wasm",
            "author": "alice",
            "points": 150,
            "num_comments": 25,
            "created_at": "2026-04-18T10:00:00.000Z",
            "_tags": ["story", "front_page"],
            "story_text": "",
        },
        {
            "objectID": "40000002",
            "title": "Show HN: A Markdown superset",
            "url": "https://example.com/markdown-x",
            "author": "bob",
            "points": 42,
            "num_comments": 10,
            "created_at": "2026-04-18T09:00:00.000Z",
            "_tags": ["story", "show_hn"],
            "story_text": "<p>Hi HN.</p>",
        },
        {
            # Ask HN story with only story_title
            "objectID": "40000003",
            "story_title": "Ask HN: Best tools for agent memory?",
            "url": "",
            "author": "carol",
            "points": 10,
            "num_comments": 3,
            "created_at": "2026-04-18T08:00:00.000Z",
            "_tags": ["story", "ask_hn"],
        },
    ]
}


class TestRequestUrl:
    def test_basic(self):
        src = HackerNewsSource(query_id="front_page", tags="front_page", hits_per_page=5)
        url = src.request_url
        assert url.startswith("https://hn.algolia.com/api/v1/search_by_date?")
        assert "tags=front_page" in url
        assert "hitsPerPage=5" in url
        assert "query=" not in url

    def test_with_query_and_min_points(self):
        src = HackerNewsSource(
            query_id="ai_research",
            query="LLM agent",
            tags="story",
            min_points=20,
        )
        url = src.request_url
        assert "query=LLM+agent" in url
        assert "numericFilters=points%3E%3D20" in url


class TestToItem:
    def test_basic_hit(self):
        src = HackerNewsSource(query_id="front_page", tags="front_page")
        item = src._to_item(ALGOLIA_FIXTURE["hits"][0])
        assert item is not None
        assert item.item_id == "40000001"
        assert item.source_id == "hn:front_page"
        assert item.title == "Zero-Copy GPU Inference from WebAssembly"
        assert item.url == "https://example.com/gpu-wasm"
        assert item.metadata["points"] == 150
        assert item.metadata["num_comments"] == 25
        assert item.metadata["hn_url"] == "https://news.ycombinator.com/item?id=40000001"

    def test_show_hn_story_text_populates_summary(self):
        src = HackerNewsSource(query_id="show_hn", tags="show_hn")
        item = src._to_item(ALGOLIA_FIXTURE["hits"][1])
        assert "Hi HN" in item.summary

    def test_ask_hn_uses_story_title(self):
        src = HackerNewsSource(query_id="ask_hn", tags="ask_hn")
        item = src._to_item(ALGOLIA_FIXTURE["hits"][2])
        assert item is not None
        assert item.title == "Ask HN: Best tools for agent memory?"
        # No url → fallback to HN item URL
        assert item.url == "https://news.ycombinator.com/item?id=40000003"

    def test_missing_id_returns_none(self):
        src = HackerNewsSource(query_id="x")
        assert src._to_item({"title": "no id"}) is None

    def test_missing_title_returns_none(self):
        src = HackerNewsSource(query_id="x")
        assert src._to_item({"objectID": "1"}) is None


class TestFetch:
    def test_fetch_maps_hits(self):
        src = HackerNewsSource(query_id="front_page", tags="front_page")
        with patch("animus.lugh.sources.hn._fetch_algolia", return_value=ALGOLIA_FIXTURE["hits"]):
            items = list(src.fetch())
        assert len(items) == 3
        assert [i.item_id for i in items] == ["40000001", "40000002", "40000003"]

    def test_fetch_respects_limit(self):
        src = HackerNewsSource(query_id="front_page", tags="front_page")
        with patch("animus.lugh.sources.hn._fetch_algolia", return_value=ALGOLIA_FIXTURE["hits"]):
            items = list(src.fetch(limit=2))
        assert len(items) == 2

    def test_fetch_network_failure_returns_empty(self):
        src = HackerNewsSource(query_id="front_page", tags="front_page")
        import urllib.error

        def fail(*a, **kw):
            raise urllib.error.URLError("no network")

        with patch("urllib.request.urlopen", fail):
            assert list(src.fetch()) == []


class TestAlgoliaParse:
    def test_algolia_bad_json_returns_empty(self):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            from animus.lugh.sources.hn import _fetch_algolia

            assert _fetch_algolia("https://x") == []

    def test_algolia_success(self):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.read.return_value = json.dumps(ALGOLIA_FIXTURE).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            from animus.lugh.sources.hn import _fetch_algolia

            hits = _fetch_algolia("https://x")
            assert len(hits) == 3


class TestDefaults:
    def test_default_sources_nonempty(self):
        assert default_sources()
        assert all(s.source_id.startswith("hn:") for s in default_sources())
