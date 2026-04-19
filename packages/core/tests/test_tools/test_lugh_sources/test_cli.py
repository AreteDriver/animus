"""Tests for animus.lugh.sources.cli — argparse dispatch and commands."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

from animus.lugh.sources.base import SourceItem
from animus.lugh.sources.cli import main


def _item(source_id: str = "arxiv:cs.LG", item_id: str = "2604.00001") -> SourceItem:
    return SourceItem(
        source_id=source_id,
        item_id=item_id,
        title="Test Paper About Memory",
        url="https://arxiv.org/abs/" + item_id,
        published=datetime(2026, 4, 18, tzinfo=timezone.utc),
        summary="An abstract about learned memory compression.",
        raw_text="Full abstract: memory compression matters.",
    )


class _StubSource:
    """Lightweight stand-in returned by the registry during CLI tests."""

    def __init__(self, source_id: str, items: list[SourceItem]):
        self.source_id = source_id
        self._items = items

    def fetch(self, limit=None):
        return self._items if limit is None else self._items[:limit]


class TestListCommand:
    def test_list_text(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        rc = main(["--registry", str(registry), "--cache", str(cache), "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "arxiv:cs.LG" in out
        assert "hn:front_page" in out

    def test_list_json(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        rc = main(["--registry", str(registry), "--cache", str(cache), "list", "--json"])
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert isinstance(parsed, list)
        assert any(x["source_id"].startswith("arxiv:") for x in parsed)


class TestFetchCommand:
    def test_fetch_inserts_and_dedups(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        stub = _StubSource("arxiv:cs.LG", [_item()])

        with patch("animus.lugh.sources.cli.instantiate", return_value=[stub]):
            rc = main(["--registry", str(registry), "--cache", str(cache), "fetch", "--quiet"])
            assert rc == 0
            rc2 = main(["--registry", str(registry), "--cache", str(cache), "fetch", "--quiet"])
            assert rc2 == 0

        # Second pass must be pure dedup.
        out = capsys.readouterr().out
        # We used --quiet so the only output is... nothing. That's the point.
        assert out == ""

    def test_fetch_filters_by_source(self, tmp_path):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        stub_a = _StubSource("arxiv:cs.LG", [_item("arxiv:cs.LG", "A")])
        stub_b = _StubSource("hn:front_page", [_item("hn:front_page", "B")])

        with patch("animus.lugh.sources.cli.instantiate", return_value=[stub_a, stub_b]):
            rc = main(
                [
                    "--registry",
                    str(registry),
                    "--cache",
                    str(cache),
                    "fetch",
                    "--source",
                    "hn:front_page",
                    "--quiet",
                ]
            )
        assert rc == 0

        # Only hn items should be in the cache now.
        from animus.lugh.sources.base import SourceCache

        c = SourceCache(path=cache)
        assert c.count(source_id="hn:front_page") == 1
        assert c.count(source_id="arxiv:cs.LG") == 0

    def test_fetch_unknown_source_errors(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        with patch("animus.lugh.sources.cli.instantiate", return_value=[]):
            rc = main(
                [
                    "--registry",
                    str(registry),
                    "--cache",
                    str(cache),
                    "fetch",
                    "--source",
                    "nope:none",
                ]
            )
        assert rc == 1


class TestRecentCommand:
    def test_recent_shows_items(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"

        from animus.lugh.sources.base import SourceCache

        SourceCache(path=cache).put(_item())

        rc = main(["--registry", str(registry), "--cache", str(cache), "recent", "--limit", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Test Paper About Memory" in out


class TestProbeCommand:
    def test_probe_uses_podcast_probe(self, capsys):
        with patch(
            "animus.lugh.sources.cli.probe_feed",
            return_value={"ok": True, "entry_count": 5, "sample_titles": [], "error": ""},
        ):
            rc = main(["probe", "https://example.com/feed"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True


class TestAddRemoveCommand:
    def test_add_podcast_and_remove(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        rc = main(
            [
                "--registry",
                str(registry),
                "--cache",
                str(cache),
                "add-podcast",
                "all-in",
                "https://example.com/all-in.rss",
                "--name",
                "All-In",
            ]
        )
        assert rc == 0
        rc2 = main(
            [
                "--registry",
                str(registry),
                "--cache",
                str(cache),
                "remove",
                "podcast:all-in",
            ]
        )
        assert rc2 == 0

    def test_add_podcast_duplicate_rejected(self, tmp_path):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        main(
            [
                "--registry",
                str(registry),
                "--cache",
                str(cache),
                "add-podcast",
                "all-in",
                "https://x",
            ]
        )
        rc = main(
            [
                "--registry",
                str(registry),
                "--cache",
                str(cache),
                "add-podcast",
                "all-in",
                "https://y",
            ]
        )
        assert rc == 1

    def test_remove_missing_returns_one(self, tmp_path):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        rc = main(
            [
                "--registry",
                str(registry),
                "--cache",
                str(cache),
                "remove",
                "podcast:does-not-exist",
            ]
        )
        assert rc == 1


class TestExplainCommand:
    def test_explain_emits_breakdown(self, tmp_path, capsys):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        stub = _StubSource("arxiv:cs.LG", [_item()])

        with patch("animus.lugh.sources.cli.instantiate", return_value=[stub]):
            rc = main(
                [
                    "--registry",
                    str(registry),
                    "--cache",
                    str(cache),
                    "explain",
                    "arxiv:cs.LG",
                    "2604.00001",
                ]
            )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "raw_score" in data
        assert "hits" in data
        assert data["item"]["title"].startswith("Test Paper")

    def test_explain_unknown_source(self, tmp_path):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        with patch("animus.lugh.sources.cli.instantiate", return_value=[]):
            rc = main(
                [
                    "--registry",
                    str(registry),
                    "--cache",
                    str(cache),
                    "explain",
                    "nope:none",
                    "id",
                ]
            )
        assert rc == 1

    def test_explain_unknown_item(self, tmp_path):
        registry = tmp_path / "r.json"
        cache = tmp_path / "c.sqlite"
        stub = _StubSource("arxiv:cs.LG", [_item(item_id="OTHER")])
        with patch("animus.lugh.sources.cli.instantiate", return_value=[stub]):
            rc = main(
                [
                    "--registry",
                    str(registry),
                    "--cache",
                    str(cache),
                    "explain",
                    "arxiv:cs.LG",
                    "MISSING",
                ]
            )
        assert rc == 1
