"""Tests for animus.lugh.sources.mer — Monthly Economic Report availability tracker."""

from __future__ import annotations

import urllib.error
from datetime import date

import pytest

from animus.lugh.sources import mer
from animus.lugh.sources.mer import MERSource, _recent_year_months, _zip_exists, default_mer_sources


class TestRecentYearMonths:
    def test_simple_within_year(self):
        assert _recent_year_months(3, today=date(2026, 5, 12)) == [(2026, 3), (2026, 4), (2026, 5)]

    def test_crosses_year_boundary(self):
        assert _recent_year_months(3, today=date(2026, 1, 15)) == [
            (2025, 11),
            (2025, 12),
            (2026, 1),
        ]

    def test_one_month(self):
        assert _recent_year_months(1, today=date(2026, 7, 1)) == [(2026, 7)]

    def test_floor_of_one(self):
        assert _recent_year_months(0, today=date(2026, 7, 1)) == [(2026, 7)]


class TestZipExists:
    def test_200_is_true(self, monkeypatch):
        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(mer.urllib.request, "urlopen", lambda *a, **k: _Resp())
        assert _zip_exists("https://example.com/x.zip") is True

    def test_404_is_false(self, monkeypatch):
        def _raise(*a, **k):
            raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

        monkeypatch.setattr(mer.urllib.request, "urlopen", _raise)
        assert _zip_exists("https://example.com/x.zip") is False

    def test_transport_error_is_false(self, monkeypatch):
        def _raise(*a, **k):
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(mer.urllib.request, "urlopen", _raise)
        assert _zip_exists("https://example.com/x.zip") is False


class TestMERSource:
    def test_source_id(self):
        assert MERSource().source_id == "mer"

    def test_default_mer_sources(self):
        srcs = default_mer_sources()
        assert len(srcs) == 1 and isinstance(srcs[0], MERSource)

    def test_fetch_emits_only_available_months(self, monkeypatch):
        # pretend "now" is mid-May 2026 and only April's zip exists
        monkeypatch.setattr(
            mer, "_recent_year_months", lambda n, today=None: [(2026, 3), (2026, 4), (2026, 5)]
        )
        available = {"https://web.ccpgamescdn.com/aws/community/EVEOnline_MER_202604.zip"}
        monkeypatch.setattr(mer, "_zip_exists", lambda url, timeout=15: url in available)

        items = list(MERSource().fetch())
        assert len(items) == 1
        it = items[0]
        assert it.source_id == "mer"
        assert it.item_id == "202604"
        assert "April 2026" in it.title
        assert it.metadata["zip_url"].endswith("EVEOnline_MER_202604.zip")
        assert it.metadata["news_url"].endswith("monthly-economic-report-april-2026")
        assert it.published is not None and it.published.year == 2026 and it.published.month == 4
        assert "eve-economy" in it.tags

    def test_fetch_respects_limit(self, monkeypatch):
        monkeypatch.setattr(
            mer, "_recent_year_months", lambda n, today=None: [(2026, 2), (2026, 3), (2026, 4)]
        )
        monkeypatch.setattr(mer, "_zip_exists", lambda url, timeout=15: True)
        # limit=1 keeps only the most recent of the probed months
        items = list(MERSource(lookback_months=3).fetch(limit=1))
        assert [i.item_id for i in items] == ["202604"]

    def test_fetch_none_available_yields_nothing(self, monkeypatch):
        monkeypatch.setattr(mer, "_zip_exists", lambda url, timeout=15: False)
        assert list(MERSource().fetch()) == []

    def test_custom_tags(self, monkeypatch):
        monkeypatch.setattr(mer, "_recent_year_months", lambda n, today=None: [(2026, 4)])
        monkeypatch.setattr(mer, "_zip_exists", lambda url, timeout=15: True)
        items = list(MERSource(tags=["custom"]).fetch())
        assert items[0].tags == ["custom"]


@pytest.mark.parametrize(
    "kind_entry", [{"kind": "mer"}, {"kind": "mer", "lookback_months": 2, "tags": ["x"]}]
)
def test_registry_instantiates_mer(kind_entry):
    from animus.lugh.sources.registry import instantiate

    out = instantiate([kind_entry])
    assert len(out) == 1 and out[0].source_id == "mer"


def test_default_registry_includes_mer():
    from animus.lugh.sources.registry import default_registry

    assert any(e.get("kind") == "mer" for e in default_registry())
