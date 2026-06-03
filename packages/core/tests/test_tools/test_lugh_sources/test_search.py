"""Tests for animus.lugh.sources.search — SearXNG JSON web-search source."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from animus.lugh.sources.search import (
    WebSearchSource,
    _fetch_searxng,
    default_sources,
)

SEARXNG_FIXTURE = {
    "query": "agent runtime",
    "number_of_results": 2,
    "results": [
        {
            "url": "https://example.com/agent-runtime",
            "title": "Designing an Agent Runtime",
            "content": "<p>A deep dive into <b>MCP servers</b> and tool calling.</p>",
            "engine": "duckduckgo",
            "engines": ["duckduckgo", "brave"],
            "score": 1.5,
            "category": "general",
            "publishedDate": "2026-04-18T10:00:00Z",
        },
        {
            "url": "https://example.com/local-llm",
            "title": "Local LLM Inference",
            "content": "Quantization and open weights for sovereign inference.",
            "engine": "bing",
            "engines": ["bing"],
            "score": 0.9,
            "category": "news",
            "publishedDate": None,
        },
        {
            # missing url — must be dropped by _to_item
            "title": "No URL Result",
            "content": "orphan",
            "engine": "google",
        },
        {
            # missing title — must be dropped by _to_item
            "url": "https://example.com/no-title",
            "content": "orphan",
            "engine": "google",
        },
    ],
}


def _mock_response(payload: object) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestRequestUrl:
    def test_basic(self):
        src = WebSearchSource(query_id="agent", query="agent runtime")
        url = src.request_url("http://localhost:8888")
        assert url.startswith("http://localhost:8888/search?")
        assert "q=agent+runtime" in url
        assert "format=json" in url
        assert "language=en" in url
        # unset optional params must not appear
        assert "categories=" not in url
        assert "time_range=" not in url

    def test_with_categories_and_time_range(self):
        src = WebSearchSource(
            query_id="ai_news",
            query="AI agents",
            categories="news",
            time_range="week",
            language="fr",
        )
        url = src.request_url("http://localhost:8888/")
        # base passed with trailing slash is used verbatim by request_url
        assert "categories=news" in url
        assert "time_range=week" in url
        assert "language=fr" in url
        assert "q=AI+agents" in url


class TestSourceId:
    def test_source_id(self):
        src = WebSearchSource(query_id="agent_infra", query="x")
        assert src.source_id == "search:agent_infra"


class TestToItem:
    def test_basic_hit(self):
        src = WebSearchSource(query_id="agent", query="agent runtime", tags=["core"])
        item = src._to_item(SEARXNG_FIXTURE["results"][0])
        assert item is not None
        assert item.source_id == "search:agent"
        # item_id is the result URL (stable dedup key for web results)
        assert item.item_id == "https://example.com/agent-runtime"
        assert item.url == "https://example.com/agent-runtime"
        assert item.title == "Designing an Agent Runtime"
        # summary has HTML stripped + whitespace collapsed
        assert "<" not in item.summary
        assert "MCP servers" in item.summary
        assert "tool calling" in item.summary
        # published parsed from Z-suffixed ISO timestamp → tz-aware
        assert item.published is not None
        assert item.published.year == 2026
        assert item.published.tzinfo is not None
        # tags merge configured tags + engine
        assert "core" in item.tags
        assert "duckduckgo" in item.tags
        # metadata mapping
        assert item.metadata["engine"] == "duckduckgo"
        assert item.metadata["engines"] == ["duckduckgo", "brave"]
        assert item.metadata["score"] == 1.5
        assert item.metadata["query"] == "agent runtime"
        assert item.metadata["category"] == "general"

    def test_missing_published_date(self):
        src = WebSearchSource(query_id="agent", query="x")
        item = src._to_item(SEARXNG_FIXTURE["results"][1])
        assert item is not None
        assert item.published is None
        # category falls through to the hit's value
        assert item.metadata["category"] == "news"

    def test_missing_url_returns_none(self):
        src = WebSearchSource(query_id="x", query="y")
        assert src._to_item(SEARXNG_FIXTURE["results"][2]) is None

    def test_missing_title_returns_none(self):
        src = WebSearchSource(query_id="x", query="y")
        assert src._to_item(SEARXNG_FIXTURE["results"][3]) is None


class TestFetch:
    def test_fail_open_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_SEARCH_URL", raising=False)
        src = WebSearchSource(query_id="agent", query="agent runtime")
        # urlopen must never be called when the endpoint is unconfigured
        with patch("urllib.request.urlopen", side_effect=AssertionError("urlopen called")):
            assert list(src.fetch()) == []

    def test_fail_open_when_env_blank(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_SEARCH_URL", "   ")
        src = WebSearchSource(query_id="agent", query="agent runtime")
        assert list(src.fetch()) == []

    def test_fetch_maps_results(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_SEARCH_URL", "http://localhost:8888")
        src = WebSearchSource(query_id="agent", query="agent runtime")
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(SEARXNG_FIXTURE),
        ):
            items = list(src.fetch())
        # two valid results; the url-less and title-less hits are dropped
        assert len(items) == 2
        assert [i.item_id for i in items] == [
            "https://example.com/agent-runtime",
            "https://example.com/local-llm",
        ]

    def test_fetch_respects_limit(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_SEARCH_URL", "http://localhost:8888")
        src = WebSearchSource(query_id="agent", query="agent runtime")
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(SEARXNG_FIXTURE),
        ):
            items = list(src.fetch(limit=1))
        assert len(items) == 1
        assert items[0].item_id == "https://example.com/agent-runtime"

    def test_fetch_strips_trailing_slash_in_base_url(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_SEARCH_URL", "http://localhost:8888/")
        src = WebSearchSource(query_id="agent", query="agent runtime")
        captured: dict[str, str] = {}

        def fake_urlopen(req, *a, **kw):
            captured["url"] = req.full_url
            return _mock_response(SEARXNG_FIXTURE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            list(src.fetch())
        # no double slash before /search — trailing slash was stripped
        assert "8888/search?" in captured["url"]
        assert "8888//search" not in captured["url"]

    def test_fetch_network_failure_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_SEARCH_URL", "http://localhost:8888")
        import urllib.error

        src = WebSearchSource(query_id="agent", query="agent runtime")

        def fail(*a, **kw):
            raise urllib.error.URLError("no network")

        with patch("urllib.request.urlopen", fail):
            assert list(src.fetch()) == []


class TestFetchSearxng:
    def test_success(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(SEARXNG_FIXTURE),
        ):
            results = _fetch_searxng("http://localhost:8888/search?q=x")
        assert len(results) == 4

    def test_bad_json_returns_empty(self):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            assert _fetch_searxng("http://localhost:8888/search?q=x") == []

    def test_http_error_returns_empty(self):
        import urllib.error

        def raise_http(*a, **kw):
            raise urllib.error.HTTPError(url="http://x", code=503, msg="down", hdrs=None, fp=None)

        with patch("urllib.request.urlopen", side_effect=raise_http):
            assert _fetch_searxng("http://localhost:8888/search?q=x") == []

    def test_missing_results_key_returns_empty(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response({"query": "x"}),
        ):
            assert _fetch_searxng("http://localhost:8888/search?q=x") == []


class TestDefaults:
    def test_default_sources_nonempty(self):
        srcs = default_sources()
        assert srcs
        assert all(s.source_id.startswith("search:") for s in srcs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
