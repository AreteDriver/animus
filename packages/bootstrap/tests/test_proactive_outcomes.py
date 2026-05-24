"""Tests for OutcomeStore — nudge outcomes + check fire log."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_bootstrap.intelligence.proactive.outcomes import (
    FireRecord,
    OutcomeRecord,
    OutcomeStore,
)


@pytest.fixture
def store(tmp_path) -> OutcomeStore:
    s = OutcomeStore(tmp_path / "outcomes.db")
    yield s
    s.close()


class TestOutcomeStore:
    def test_record_and_fetch_outcome(self, store: OutcomeStore) -> None:
        oid = store.record_outcome(
            nudge_id="n1",
            signal_type="explicit",
            signal_value=1.0,
            source="dashboard",
        )
        assert oid

        results = store.outcomes_for("n1")
        assert len(results) == 1
        assert isinstance(results[0], OutcomeRecord)
        assert results[0].signal_type == "explicit"
        assert results[0].signal_value == 1.0
        assert results[0].source == "dashboard"

    def test_multiple_signals_per_nudge(self, store: OutcomeStore) -> None:
        store.record_outcome("n1", "implicit_action", 1.0, source="implicit_action")
        store.record_outcome("n1", "explicit", -1.0, source="dashboard")
        results = store.outcomes_for("n1")
        assert len(results) == 2

    def test_outcomes_for_unknown_nudge_returns_empty(self, store: OutcomeStore) -> None:
        assert store.outcomes_for("nope") == []

    def test_record_outcome_uses_supplied_timestamp(self, store: OutcomeStore) -> None:
        ts = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
        store.record_outcome("n1", "explicit", 1.0, captured_at=ts)
        results = store.outcomes_for("n1")
        assert results[0].captured_at == ts

    def test_record_fire_and_query(self, store: OutcomeStore) -> None:
        store.record_fire("morning_brief", "produced")
        store.record_fire("morning_brief", "silent")
        store.record_fire("morning_brief", "error")
        records = store.fires_for("morning_brief")
        assert len(records) == 3
        results = {r.result for r in records}
        assert results == {"produced", "silent", "error"}
        assert all(isinstance(r, FireRecord) for r in records)

    def test_fires_for_filters_by_since(self, store: OutcomeStore) -> None:
        old = datetime.now(UTC) - timedelta(days=10)
        new = datetime.now(UTC) - timedelta(hours=1)
        store.record_fire("c", "produced", fired_at=old)
        store.record_fire("c", "produced", fired_at=new)

        cutoff = datetime.now(UTC) - timedelta(days=2)
        results = store.fires_for("c", since=cutoff)
        assert len(results) == 1

    def test_fire_counts_aggregates(self, store: OutcomeStore) -> None:
        for _ in range(3):
            store.record_fire("c", "produced")
        for _ in range(5):
            store.record_fire("c", "silent")
        store.record_fire("c", "error")

        counts = store.fire_counts("c", window_days=1)
        assert counts == {"produced": 3, "silent": 5, "error": 1, "total": 9}

    def test_fire_counts_empty(self, store: OutcomeStore) -> None:
        counts = store.fire_counts("absent", window_days=7)
        assert counts == {"produced": 0, "silent": 0, "error": 0, "total": 0}

    def test_fire_counts_excludes_outside_window(self, store: OutcomeStore) -> None:
        store.record_fire("c", "produced", fired_at=datetime.now(UTC) - timedelta(days=30))
        store.record_fire("c", "produced")
        counts = store.fire_counts("c", window_days=7)
        assert counts["total"] == 1

    def test_close_is_idempotent_on_reopen(self, tmp_path) -> None:
        path = tmp_path / "x.db"
        s1 = OutcomeStore(path)
        s1.record_outcome("n", "explicit", 1.0)
        s1.close()
        s2 = OutcomeStore(path)
        try:
            assert len(s2.outcomes_for("n")) == 1
        finally:
            s2.close()
