"""Tests for animus.lugh.sources.registry."""

from __future__ import annotations

import json

from animus.lugh.sources.registry import (
    _entry_source_id,
    add_podcast,
    default_registry,
    instantiate,
    load_registry,
    remove_source,
    save_registry,
)


class TestDefaultRegistry:
    def test_seeds_arxiv_and_hn(self):
        entries = default_registry()
        kinds = {e["kind"] for e in entries}
        assert "arxiv" in kinds
        assert "hn" in kinds
        # Podcasts intentionally NOT seeded (we don't guess URLs).
        assert "podcast" not in kinds

    def test_contains_expected_categories(self):
        entries = default_registry()
        arxiv_cats = {e["category"] for e in entries if e["kind"] == "arxiv"}
        assert "cs.LG" in arxiv_cats
        assert "cs.CL" in arxiv_cats
        assert "cs.AI" in arxiv_cats


class TestLoadSave:
    def test_load_seeds_file_on_first_call(self, tmp_path):
        path = tmp_path / "reg.json"
        assert not path.exists()
        entries = load_registry(path=path)
        assert path.exists()
        assert entries == default_registry()

    def test_load_existing_file(self, tmp_path):
        path = tmp_path / "reg.json"
        path.write_text(json.dumps([{"kind": "arxiv", "category": "stat.ML"}]))
        entries = load_registry(path=path)
        assert entries == [{"kind": "arxiv", "category": "stat.ML"}]

    def test_load_malformed_returns_empty(self, tmp_path):
        path = tmp_path / "reg.json"
        path.write_text("not json")
        assert load_registry(path=path) == []

    def test_load_non_list_returns_empty(self, tmp_path):
        path = tmp_path / "reg.json"
        path.write_text('{"oops": "dict, not list"}')
        assert load_registry(path=path) == []

    def test_save_round_trip(self, tmp_path):
        path = tmp_path / "reg.json"
        entries = [{"kind": "arxiv", "category": "cs.LG"}]
        saved_path = save_registry(entries, path=path)
        assert saved_path == path
        assert json.loads(path.read_text()) == entries


class TestInstantiate:
    def test_instantiate_arxiv(self):
        sources = instantiate([{"kind": "arxiv", "category": "cs.LG"}])
        assert len(sources) == 1
        assert sources[0].source_id == "arxiv:cs.LG"

    def test_instantiate_hn(self):
        sources = instantiate([{"kind": "hn", "query_id": "front_page", "tags": "front_page"}])
        assert sources[0].source_id == "hn:front_page"

    def test_instantiate_podcast(self):
        sources = instantiate(
            [
                {
                    "kind": "podcast",
                    "show_id": "all-in",
                    "feed_url": "https://x",
                    "show_name": "All-In",
                }
            ]
        )
        assert sources[0].source_id == "podcast:all-in"

    def test_instantiate_bad_entries_skipped(self, caplog):
        sources = instantiate(
            [
                "not a dict",  # invalid — ignored entirely
                {"kind": "unknown"},  # logs warning
                {"kind": "arxiv"},  # missing category — logs warning
                {"kind": "arxiv", "category": "cs.LG"},
            ]
        )
        # Only the valid arxiv entry instantiates
        assert [s.source_id for s in sources] == ["arxiv:cs.LG"]


class TestAddRemovePodcast:
    def test_add_podcast_new(self, tmp_path):
        path = tmp_path / "reg.json"
        assert add_podcast("all-in", "https://x", "All-In", path=path) is True
        entries = load_registry(path=path)
        podcasts = [e for e in entries if e["kind"] == "podcast"]
        assert len(podcasts) == 1
        assert podcasts[0]["show_id"] == "all-in"
        assert podcasts[0]["feed_url"] == "https://x"

    def test_add_podcast_dup(self, tmp_path):
        path = tmp_path / "reg.json"
        assert add_podcast("all-in", "https://x", path=path) is True
        assert add_podcast("all-in", "https://y", path=path) is False

    def test_remove_existing(self, tmp_path):
        path = tmp_path / "reg.json"
        add_podcast("all-in", "https://x", path=path)
        assert remove_source("podcast:all-in", path=path) is True
        podcasts = [e for e in load_registry(path=path) if e["kind"] == "podcast"]
        assert podcasts == []

    def test_remove_missing(self, tmp_path):
        path = tmp_path / "reg.json"
        load_registry(path=path)  # seed
        assert remove_source("podcast:does-not-exist", path=path) is False

    def test_remove_arxiv(self, tmp_path):
        path = tmp_path / "reg.json"
        load_registry(path=path)
        assert remove_source("arxiv:cs.LG", path=path) is True
        assert all(e.get("category") != "cs.LG" for e in load_registry(path=path))


class TestEntrySourceId:
    def test_arxiv(self):
        assert _entry_source_id({"kind": "arxiv", "category": "cs.LG"}) == "arxiv:cs.LG"

    def test_hn(self):
        assert _entry_source_id({"kind": "hn", "query_id": "fp"}) == "hn:fp"

    def test_podcast(self):
        assert _entry_source_id({"kind": "podcast", "show_id": "all-in"}) == "podcast:all-in"

    def test_unknown(self):
        assert _entry_source_id({"kind": "other"}) == ""
