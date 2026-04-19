"""Tests for animus.lugh.sources.base — SourceItem + SourceCache."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from animus.lugh.sources.base import SourceCache, SourceItem


def _item(source_id: str = "arxiv:cs.LG", item_id: str = "2604.00001", **kw) -> SourceItem:
    defaults = dict(
        source_id=source_id,
        item_id=item_id,
        title="A Paper About Memory",
        url="http://arxiv.org/abs/2604.00001",
        published=datetime(2026, 4, 18, tzinfo=timezone.utc),
        summary="An abstract about learned memory.",
        raw_text="Full abstract: we propose learned memory modules...",
    )
    defaults.update(kw)
    return SourceItem(**defaults)


class TestSourceItem:
    def test_fingerprint_stable_and_unique(self):
        a = _item(source_id="arxiv:cs.LG", item_id="2604.00001")
        b = _item(source_id="arxiv:cs.LG", item_id="2604.00001", title="other")
        c = _item(source_id="arxiv:cs.LG", item_id="2604.00002")
        d = _item(source_id="hn:front_page", item_id="2604.00001")
        assert a.fingerprint == b.fingerprint  # id fields identical
        assert a.fingerprint != c.fingerprint  # different item_id
        assert a.fingerprint != d.fingerprint  # different source_id

    def test_fingerprint_is_sha256_hex(self):
        fp = _item().fingerprint
        assert len(fp) == 64
        assert all(ch in "0123456789abcdef" for ch in fp)


class TestSourceCache:
    def _cache(self, tmp_path: Path) -> SourceCache:
        return SourceCache(path=tmp_path / "t.sqlite")

    def test_put_and_seen(self, tmp_path):
        cache = self._cache(tmp_path)
        item = _item()
        assert not cache.seen(item.fingerprint)
        assert cache.put(item) is True
        assert cache.seen(item.fingerprint)

    def test_put_dedup(self, tmp_path):
        cache = self._cache(tmp_path)
        item = _item()
        assert cache.put(item) is True
        assert cache.put(item) is False  # second insert is a dedup skip
        assert cache.count() == 1

    def test_put_many_counts(self, tmp_path):
        cache = self._cache(tmp_path)
        items = [_item(item_id=f"2604.{i:05d}") for i in range(5)]
        r1 = cache.put_many(items)
        r2 = cache.put_many(items)
        assert r1 == {"inserted": 5, "skipped": 0}
        assert r2 == {"inserted": 0, "skipped": 5}
        assert cache.count() == 5

    def test_put_many_with_scorer(self, tmp_path):
        cache = self._cache(tmp_path)

        class HalfScorer:
            def score(self, item):
                return 0.5

        r = cache.put_many([_item()], scorer=HalfScorer())
        assert r == {"inserted": 1, "skipped": 0}
        # recent() filter verifies score was stored
        assert cache.recent(min_score=0.4) != []
        assert cache.recent(min_score=0.9) == []

    def test_count_by_source(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put(_item(source_id="arxiv:cs.LG", item_id="1"))
        cache.put(_item(source_id="arxiv:cs.LG", item_id="2"))
        cache.put(_item(source_id="hn:front_page", item_id="10"))
        assert cache.count() == 3
        assert cache.count(source_id="arxiv:cs.LG") == 2
        assert cache.count(source_id="hn:front_page") == 1

    def test_recent_filters_by_source(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put(_item(source_id="arxiv:cs.LG", item_id="1"))
        cache.put(_item(source_id="hn:front_page", item_id="2"))
        assert len(cache.recent()) == 2
        assert len(cache.recent(source_id="hn:front_page")) == 1

    def test_recent_orders_by_published_desc(self, tmp_path):
        cache = self._cache(tmp_path)
        old = _item(item_id="A", published=datetime(2026, 1, 1, tzinfo=timezone.utc))
        new = _item(item_id="B", published=datetime(2026, 4, 18, tzinfo=timezone.utc))
        cache.put(old)
        cache.put(new)
        result = cache.recent()
        assert [i.item_id for i in result] == ["B", "A"]

    def test_round_trip_preserves_fields(self, tmp_path):
        cache = self._cache(tmp_path)
        original = _item(
            tags=["cs.LG", "cs.AI"],
            metadata={"arxiv_id": "2604.00001", "points": 42},
        )
        cache.put(original)
        [recovered] = cache.recent()
        assert recovered.title == original.title
        assert recovered.tags == original.tags
        assert recovered.metadata == original.metadata
        assert recovered.published == original.published

    def test_close_does_not_raise(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.close()  # idempotent-enough for our uses
