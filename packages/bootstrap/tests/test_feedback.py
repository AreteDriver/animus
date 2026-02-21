"""Tests for FeedbackStore."""

from __future__ import annotations

import pytest

from animus_bootstrap.intelligence.feedback import FeedbackStore


@pytest.fixture()
def store(tmp_path):
    """Return a FeedbackStore backed by tmp_path."""
    s = FeedbackStore(tmp_path / "feedback.db")
    yield s
    s.close()


class TestRecord:
    def test_record_returns_id(self, store):
        fid = store.record("hello", "hi there", 1)
        assert isinstance(fid, str)
        assert len(fid) > 0

    def test_record_positive(self, store):
        store.record("q1", "a1", 1)
        recent = store.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0]["rating"] == 1

    def test_record_negative(self, store):
        store.record("q1", "a1", -1, comment="too verbose")
        recent = store.get_recent(limit=1)
        assert recent[0]["rating"] == -1
        assert recent[0]["comment"] == "too verbose"

    def test_record_with_channel(self, store):
        store.record("q", "a", 1, channel="discord")
        recent = store.get_recent(limit=1)
        assert recent[0]["channel"] == "discord"


class TestGetRecent:
    def test_empty_store(self, store):
        assert store.get_recent() == []

    def test_returns_most_recent_first(self, store):
        store.record("first", "a1", 1)
        store.record("second", "a2", -1)
        recent = store.get_recent(limit=10)
        assert len(recent) == 2
        assert recent[0]["message_text"] == "second"

    def test_respects_limit(self, store):
        for i in range(10):
            store.record(f"q{i}", f"a{i}", 1)
        recent = store.get_recent(limit=3)
        assert len(recent) == 3


class TestPatterns:
    def test_positive_patterns(self, store):
        store.record("q1", "a1", 1)
        store.record("q2", "a2", -1)
        store.record("q3", "a3", 1)
        positives = store.get_positive_patterns()
        assert len(positives) == 2
        assert all(p["rating"] > 0 for p in positives)

    def test_negative_patterns(self, store):
        store.record("q1", "a1", 1)
        store.record("q2", "a2", -1)
        negatives = store.get_negative_patterns()
        assert len(negatives) == 1
        assert negatives[0]["rating"] < 0


class TestGetStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["positive_pct"] == 0
        assert stats["negative_pct"] == 0

    def test_stats_with_data(self, store):
        store.record("q1", "a1", 1)
        store.record("q2", "a2", 1)
        store.record("q3", "a3", -1)
        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["positive"] == 2
        assert stats["negative"] == 1
        assert stats["positive_pct"] == pytest.approx(66.7, abs=0.1)
        assert stats["negative_pct"] == pytest.approx(33.3, abs=0.1)


class TestClose:
    def test_close_and_reopen(self, tmp_path):
        db_path = tmp_path / "feedback.db"
        s = FeedbackStore(db_path)
        s.record("q", "a", 1)
        s.close()

        # Reopen should see the data
        s2 = FeedbackStore(db_path)
        assert len(s2.get_recent()) == 1
        s2.close()
