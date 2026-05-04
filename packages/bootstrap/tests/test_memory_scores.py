"""Tests for MemoryScoreStore — cumulative outcomes + recency decay."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from animus_bootstrap.intelligence.memory_scores import (
    DEFAULT_HALF_LIFE_DAYS,
    MemoryScoreStore,
)


@pytest.fixture
def store(tmp_path) -> MemoryScoreStore:
    s = MemoryScoreStore(tmp_path / "scores.db")
    yield s
    s.close()


class TestRecordOutcome:
    def test_first_outcome_creates_row(self, store: MemoryScoreStore) -> None:
        store.record_outcome("m1", 1.0)
        score = store.get("m1")
        assert score is not None
        assert score.n_recalls == 1
        assert score.sum_outcome == 1.0
        assert score.raw_mean == 1.0

    def test_subsequent_outcomes_increment(self, store: MemoryScoreStore) -> None:
        for v in (1.0, 1.0, 0.0, -1.0):
            store.record_outcome("m1", v)
        score = store.get("m1")
        assert score is not None
        assert score.n_recalls == 4
        assert score.sum_outcome == 1.0
        assert score.raw_mean == pytest.approx(0.25)

    def test_get_unknown_returns_none(self, store: MemoryScoreStore) -> None:
        assert store.get("absent") is None


class TestEffectiveScore:
    def test_unknown_memory_returns_zero(self, store: MemoryScoreStore) -> None:
        assert store.effective_score("absent") == 0.0

    def test_fresh_score_equals_raw_mean(self, store: MemoryScoreStore) -> None:
        store.record_outcome("m1", 1.0)
        # Just-recorded → ~no decay applied
        assert store.effective_score("m1") == pytest.approx(1.0, abs=1e-3)

    def test_decay_halves_after_half_life(self, tmp_path) -> None:
        store = MemoryScoreStore(tmp_path / "s.db", half_life_days=10.0)
        try:
            store.record_outcome("m1", 1.0)
            future = datetime.now(UTC) + timedelta(days=10)
            assert store.effective_score("m1", now=future) == pytest.approx(0.5, abs=1e-3)
        finally:
            store.close()

    def test_decay_quarter_after_two_half_lives(self, tmp_path) -> None:
        store = MemoryScoreStore(tmp_path / "s.db", half_life_days=10.0)
        try:
            store.record_outcome("m1", 1.0)
            future = datetime.now(UTC) + timedelta(days=20)
            assert store.effective_score("m1", now=future) == pytest.approx(0.25, abs=1e-3)
        finally:
            store.close()

    def test_decay_zero_disables_decay(self, tmp_path) -> None:
        store = MemoryScoreStore(tmp_path / "s.db", half_life_days=0.0)
        try:
            store.record_outcome("m1", 1.0)
            future = datetime.now(UTC) + timedelta(days=365)
            assert store.effective_score("m1", now=future) == pytest.approx(1.0)
        finally:
            store.close()

    def test_negative_score_decays_toward_zero_not_flipped(self, tmp_path) -> None:
        store = MemoryScoreStore(tmp_path / "s.db", half_life_days=10.0)
        try:
            store.record_outcome("m1", -1.0)
            future = datetime.now(UTC) + timedelta(days=10)
            value = store.effective_score("m1", now=future)
            assert value == pytest.approx(-0.5, abs=1e-3)
            assert value < 0  # never flipped sign
        finally:
            store.close()

    def test_default_half_life_constant(self) -> None:
        assert DEFAULT_HALF_LIFE_DAYS == 30.0
        # exp(-30/30 * ln 2) = 0.5
        assert math.pow(0.5, 30.0 / DEFAULT_HALF_LIFE_DAYS) == pytest.approx(0.5)


class TestTopBottom:
    def test_top_orders_by_mean_desc(self, store: MemoryScoreStore) -> None:
        store.record_outcome("good", 1.0)
        store.record_outcome("good", 1.0)
        store.record_outcome("mid", 1.0)
        store.record_outcome("mid", 0.0)
        store.record_outcome("bad", -1.0)

        top = store.top_memories(limit=3)
        ids = [s.memory_id for s in top]
        assert ids == ["good", "mid", "bad"]

    def test_bottom_orders_by_mean_asc(self, store: MemoryScoreStore) -> None:
        store.record_outcome("good", 1.0)
        store.record_outcome("bad", -1.0)
        store.record_outcome("bad", -1.0)
        bottom = store.bottom_memories(limit=2)
        ids = [s.memory_id for s in bottom]
        assert ids == ["bad", "good"]

    def test_stats_returns_aggregates(self, store: MemoryScoreStore) -> None:
        store.record_outcome("a", 1.0)
        store.record_outcome("a", 0.0)
        store.record_outcome("b", -1.0)
        stats = store.stats()
        assert stats["scored_memories"] == 2
        assert stats["total_recalls_attributed"] == 3
        # mean of (a's mean=0.5, b's mean=-1.0) = -0.25
        assert stats["mean_score"] == pytest.approx(-0.25)

    def test_stats_empty(self, store: MemoryScoreStore) -> None:
        stats = store.stats()
        assert stats == {
            "scored_memories": 0,
            "total_recalls_attributed": 0,
            "mean_score": 0.0,
        }
