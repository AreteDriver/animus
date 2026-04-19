"""Targeted tests to close coverage gaps in animus.lugh.sources.

Kept in one file so the coverage-floor work is visible and easy to prune
when the modules it backstops get their own thorough tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus.lugh.sources.arxiv import _first_paragraph
from animus.lugh.sources.arxiv import default_sources as arxiv_defaults
from animus.lugh.sources.base import SourceCache, SourceItem
from animus.lugh.sources.cli import main
from animus.lugh.sources.hn import _fetch_algolia, _strip_html, _truncate
from animus.lugh.sources.podcasts import (
    PodcastSource,
    _fetch_transcript,
    _flatten_json_transcript,
)
from animus.lugh.sources.rss import _parse_iso, _parse_rfc2822, fetch_feed, parse_feed


def _item(**kw) -> SourceItem:
    d = dict(
        source_id="arxiv:cs.LG",
        item_id="2604.00001",
        title="T",
        url="https://x",
        published=datetime(2026, 4, 18, tzinfo=timezone.utc),
        summary="s",
        raw_text="r",
    )
    d.update(kw)
    return SourceItem(**d)


# --------------------------------------------------------------------------- #
# rss.py
# --------------------------------------------------------------------------- #


class TestRSSGaps:
    def test_parse_rfc2822_bad_input(self):
        assert _parse_rfc2822("") is None
        assert _parse_rfc2822("garbage") is None

    def test_parse_iso_none_and_bad(self):
        assert _parse_iso("") is None
        assert _parse_iso("not-a-date") is None

    def test_parse_iso_trailing_z(self):
        dt = _parse_iso("2026-04-18T10:00:00Z")
        assert dt is not None and dt.tzinfo is not None

    def test_parse_feed_missing_channel(self):
        assert parse_feed(b"<rss version='2.0'></rss>") == []

    def test_fetch_feed_http_error(self, monkeypatch):
        import urllib.error

        def boom(*a, **kw):
            raise urllib.error.HTTPError("u", 500, "boom", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert fetch_feed("http://x") == []


# --------------------------------------------------------------------------- #
# hn.py
# --------------------------------------------------------------------------- #


class TestHnGaps:
    def test_strip_html_removes_tags(self):
        assert _strip_html("<p>hi <b>there</b></p>") == "hi there"

    def test_truncate_short_string(self):
        assert _truncate("abc", 100) == "abc"

    def test_truncate_long_string(self):
        out = _truncate("a " * 200, 20)
        assert out.endswith("…")
        assert len(out) <= 21

    def test_fetch_algolia_http_error(self, monkeypatch):
        import urllib.error

        def boom(*a, **kw):
            raise urllib.error.HTTPError("u", 500, "oops", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert _fetch_algolia("https://x") == []


# --------------------------------------------------------------------------- #
# arxiv.py
# --------------------------------------------------------------------------- #


class TestArxivGaps:
    def test_first_paragraph_strips_html(self):
        text = "<p>one two three <b>four</b></p>"
        assert "<" not in _first_paragraph(text, limit=100)

    def test_first_paragraph_empty(self):
        assert _first_paragraph("") == ""

    def test_first_paragraph_truncates(self):
        text = "word " * 300
        out = _first_paragraph(text, limit=100)
        assert len(out) <= 101
        assert out.endswith("…")

    def test_default_sources_default_categories(self):
        srcs = arxiv_defaults()
        ids = {s.source_id for s in srcs}
        assert "arxiv:cs.LG" in ids
        assert "arxiv:cs.CL" in ids


# --------------------------------------------------------------------------- #
# podcasts.py
# --------------------------------------------------------------------------- #


class TestPodcastsGaps:
    def test_flatten_json_scalar_unexpected_shape(self):
        # top-level int → not list/dict; should fall back to plaintext
        assert _flatten_json_transcript("42").strip() == "42"

    def test_flatten_json_nested_without_body(self):
        payload = '{"segments": [{"no_body": "x"}]}'
        assert _flatten_json_transcript(payload) == ""

    def test_fetch_transcript_http_error(self, monkeypatch):
        import urllib.error

        def boom(*a, **kw):
            raise urllib.error.HTTPError("u", 403, "no", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert _fetch_transcript("https://x/t.srt", "application/srt") == ""

    def test_fetch_transcript_too_large(self, monkeypatch):
        huge = b"a" * 3_000_000  # > TRANSCRIPT_MAX_BYTES

        resp = MagicMock()
        resp.read.return_value = huge
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: resp)
        assert _fetch_transcript("https://x/t.srt", "application/srt") == ""

    def test_fetch_transcript_json(self, monkeypatch):
        payload = json.dumps({"segments": [{"body": "hi"}]}).encode()

        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: resp)
        assert _fetch_transcript("https://x/t.json", "application/json") == "hi"

    def test_fetch_transcript_plaintext_default(self, monkeypatch):
        resp = MagicMock()
        resp.read.return_value = b"  plain body  "
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: resp)
        assert _fetch_transcript("https://x/t.txt", "text/plain") == "plain body"

    def test_source_with_no_transcript_extras(self):
        # transcript extras missing entirely — metadata should still populate.
        src = PodcastSource(show_id="x", feed_url="https://x", fetch_transcripts=True)
        from animus.lugh.sources.rss import FeedEntry

        entry = FeedEntry(
            id="e1",
            title="Ep",
            link="https://x/e1",
            published=None,
            summary="notes",
            content="",
            extra={},  # no transcript url
        )
        item = src._to_item(entry)
        assert item.metadata["has_transcript"] is False


# --------------------------------------------------------------------------- #
# base.py
# --------------------------------------------------------------------------- #


class TestBaseGaps:
    def test_recent_min_score_only(self, tmp_path: Path):
        cache = SourceCache(path=tmp_path / "t.sqlite")

        class S:
            def score(self, _):
                return 0.8

        cache.put_many([_item(item_id="a"), _item(item_id="b")], scorer=S())
        # All items score 0.8; min_score=0.9 filters them out
        assert cache.recent(min_score=0.9) == []
        assert len(cache.recent(min_score=0.1)) == 2

    def test_close_idempotent_enough(self, tmp_path):
        cache = SourceCache(path=tmp_path / "t.sqlite")
        cache.close()


# --------------------------------------------------------------------------- #
# cli.py
# --------------------------------------------------------------------------- #


class TestSecondOrderGaps:
    """Leftover branch-coverage tickles."""

    def test_put_many_with_scorer_and_duplicate(self, tmp_path):
        cache = SourceCache(path=tmp_path / "t.sqlite")

        class S:
            def score(self, _):
                return 0.3

        item = _item()
        r1 = cache.put_many([item], scorer=S())
        r2 = cache.put_many([item], scorer=S())
        assert r1 == {"inserted": 1, "skipped": 0}
        assert r2 == {"inserted": 0, "skipped": 1}

    def test_parse_algolia_date_variants(self):
        from animus.lugh.sources.hn import _parse_algolia_date

        assert _parse_algolia_date(None) is None
        assert _parse_algolia_date("") is None
        assert _parse_algolia_date("not-a-date") is None
        # no timezone -> treated as UTC
        dt = _parse_algolia_date("2026-04-18T10:00:00")
        assert dt is not None and dt.tzinfo is not None

    def test_parse_atom_entry_without_link(self):
        atom = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>urn:x:1</id>
    <title>A</title>
    <updated>2026-04-18T10:00:00Z</updated>
  </entry>
</feed>"""
        [entry] = parse_feed(atom)
        assert entry.id == "urn:x:1"
        # no link was provided — fall back to id
        assert entry.link == ""

    def test_rss_enclosure_missing_ok(self):
        rss = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><item>
  <title>No enclosure here</title>
  <link>https://example.com/x</link>
</item></channel></rss>"""
        [entry] = parse_feed(rss)
        assert entry.enclosure_url is None
        assert entry.enclosure_type is None

    def test_fetch_transcript_srt_by_extension(self, monkeypatch):
        body = b"1\n00:00:00,000 --> 00:00:01,000\nHi\n"

        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: resp)
        # mime_type empty — route by URL suffix
        out = _fetch_transcript("https://example.com/e.srt", "")
        assert "Hi" in out


class TestCliGaps:
    def test_fetch_json_output(self, tmp_path, capsys):
        class Stub:
            source_id = "arxiv:cs.LG"

            def fetch(self, limit=None):
                return [_item()]

        with patch("animus.lugh.sources.cli.instantiate", return_value=[Stub()]):
            rc = main(
                [
                    "--registry",
                    str(tmp_path / "r.json"),
                    "--cache",
                    str(tmp_path / "c.sqlite"),
                    "fetch",
                    "--json",
                    "--quiet",
                ]
            )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "arxiv:cs.LG" in data

    def test_fetch_noquiet_shows_totals(self, tmp_path, capsys):
        class Stub:
            source_id = "arxiv:cs.LG"

            def fetch(self, limit=None):
                return [_item()]

        with patch("animus.lugh.sources.cli.instantiate", return_value=[Stub()]):
            rc = main(
                [
                    "--registry",
                    str(tmp_path / "r.json"),
                    "--cache",
                    str(tmp_path / "c.sqlite"),
                    "fetch",
                ]
            )
        assert rc == 0
        out = capsys.readouterr().out
        assert "total new" in out

    def test_recent_json(self, tmp_path, capsys):
        cache_path = tmp_path / "c.sqlite"
        SourceCache(path=cache_path).put(_item(title="JSON Me"))
        rc = main(
            [
                "--registry",
                str(tmp_path / "r.json"),
                "--cache",
                str(cache_path),
                "recent",
                "--json",
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert any(d["title"] == "JSON Me" for d in data)

    def test_list_empty_registry(self, tmp_path, capsys):
        empty_reg = tmp_path / "r.json"
        empty_reg.write_text("[]")
        rc = main(["--registry", str(empty_reg), "--cache", str(tmp_path / "c.sqlite"), "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No sources configured" in out
