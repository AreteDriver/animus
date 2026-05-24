"""Tests for RecallLog — recall events + memory id linkage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from animus_bootstrap.intelligence.recall_log import RecallLog


@pytest.fixture
def log(tmp_path) -> RecallLog:
    log = RecallLog(tmp_path / "recalls.db")
    yield log
    log.close()


class TestRecallLog:
    def test_log_recall_returns_id(self, log: RecallLog) -> None:
        rid = log.log_recall(
            query="what tasks are open",
            memory_entries=[("m1", "episodic"), ("m2", "semantic")],
        )
        assert rid

    def test_get_recall_returns_query_and_memory_ids_in_order(self, log: RecallLog) -> None:
        rid = log.log_recall(
            query="hello",
            memory_entries=[("m1", "episodic"), ("m2", "semantic"), ("m3", "procedural")],
        )
        record = log.get_recall(rid)
        assert record is not None
        assert record.query == "hello"
        assert record.memory_ids == ("m1", "m2", "m3")

    def test_get_recall_unknown_returns_none(self, log: RecallLog) -> None:
        assert log.get_recall("nope") is None

    def test_log_recall_skips_empty_and_duplicate_memory_ids(self, log: RecallLog) -> None:
        rid = log.log_recall(
            query="q",
            memory_entries=[("m1", "episodic"), ("", "semantic"), ("m1", "procedural")],
        )
        assert log.memory_ids_for(rid) == ["m1"]

    def test_log_recall_with_zero_memories(self, log: RecallLog) -> None:
        rid = log.log_recall(query="silent", memory_entries=[])
        assert log.memory_ids_for(rid) == []
        record = log.get_recall(rid)
        assert record is not None
        assert record.memory_ids == ()

    def test_recall_count(self, log: RecallLog) -> None:
        for i in range(5):
            log.log_recall(query=f"q{i}", memory_entries=[("m", "episodic")])
        assert log.recall_count() == 5

    def test_supplied_timestamp_is_persisted(self, log: RecallLog) -> None:
        ts = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
        rid = log.log_recall(query="q", memory_entries=[("m1", "episodic")], recalled_at=ts)
        record = log.get_recall(rid)
        assert record is not None
        assert record.recalled_at == ts

    def test_context_hint_persisted(self, log: RecallLog) -> None:
        rid = log.log_recall(
            query="q",
            memory_entries=[("m1", "episodic")],
            context_hint="dashboard:/memory",
        )
        record = log.get_recall(rid)
        assert record is not None
        assert record.context_hint == "dashboard:/memory"
