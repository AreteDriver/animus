"""Tests for proactive metrics — hit_rate calibration."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from animus_bootstrap.intelligence.proactive.engine import ProactiveEngine
from animus_bootstrap.intelligence.proactive.metrics import hit_rate, hit_rate_all
from animus_bootstrap.intelligence.proactive.outcomes import OutcomeStore


@pytest.fixture
def stores(tmp_path):
    proactive_path = tmp_path / "proactive.db"
    outcomes_path = tmp_path / "outcomes.db"
    engine = ProactiveEngine(db_path=proactive_path)
    outcomes = OutcomeStore(outcomes_path)
    yield engine, outcomes, proactive_path, outcomes_path
    engine.close()
    outcomes.close()


def _insert_nudge(
    db: str,
    nudge_id: str,
    check_name: str,
    predicted_value: float = 0.5,
    confidence: float = 0.5,
    ts: datetime | None = None,
) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO proactive_log
            (id, check_name, text, channels, priority, timestamp, delivered,
             confidence, predicted_value, reasoning)
        VALUES (?, ?, ?, '[]', 'normal', ?, 1, ?, ?, '')
        """,
        (
            nudge_id,
            check_name,
            "text",
            (ts or datetime.now(UTC)).isoformat(),
            confidence,
            predicted_value,
        ),
    )
    conn.commit()
    conn.close()


class TestHitRate:
    def test_zero_nudges_returns_zero_metrics(self, stores) -> None:
        _, outcomes, p, o = stores
        report = hit_rate(p, o, "absent", window_days=7)
        assert report["n_nudges"] == 0
        assert report["predicted_mean"] == 0.0
        assert report["actual_mean"] == 0.0
        assert report["calibration_error"] == 0.0
        assert report["fire_counts"]["total"] == 0

    def test_perfect_calibration(self, stores) -> None:
        _, outcomes, p, o = stores
        for i in range(4):
            nid = f"n{i}"
            _insert_nudge(str(p), nid, "c", predicted_value=1.0)
            outcomes.record_outcome(nid, "implicit_action", 1.0)

        report = hit_rate(p, o, "c", window_days=7)
        assert report["n_nudges"] == 4
        assert report["n_with_outcome"] == 4
        assert report["predicted_mean"] == 1.0
        # actual_mean=1.0; rescaled to [0, 1] also 1.0; error=0
        assert report["calibration_error"] == 0.0

    def test_overconfident_checker_has_positive_error(self, stores) -> None:
        _, outcomes, p, o = stores
        for i in range(3):
            nid = f"n{i}"
            _insert_nudge(str(p), nid, "c", predicted_value=0.9)
            outcomes.record_outcome(nid, "implicit_action", 0.0)  # all silent

        report = hit_rate(p, o, "c", window_days=7)
        # actual=0.0 → rescaled 0.5; predicted=0.9 → error=0.4
        assert report["predicted_mean"] == pytest.approx(0.9)
        assert report["calibration_error"] == pytest.approx(0.4)

    def test_underconfident_checker_has_negative_error(self, stores) -> None:
        _, outcomes, p, o = stores
        for i in range(3):
            nid = f"n{i}"
            _insert_nudge(str(p), nid, "c", predicted_value=0.1)
            outcomes.record_outcome(nid, "implicit_action", 1.0)

        report = hit_rate(p, o, "c", window_days=7)
        # predicted=0.1, actual rescaled=1.0 → error=-0.9
        assert report["calibration_error"] == pytest.approx(-0.9)

    def test_silent_nudges_count_as_neutral(self, stores) -> None:
        _, outcomes, p, o = stores
        _insert_nudge(str(p), "n1", "c", predicted_value=0.5)
        # No outcome recorded → counts as 0.0 actual
        report = hit_rate(p, o, "c", window_days=7)
        assert report["n_nudges"] == 1
        assert report["n_with_outcome"] == 0
        assert report["actual_mean"] == 0.0

    def test_explicit_thumbs_aggregate_with_implicit(self, stores) -> None:
        _, outcomes, p, o = stores
        _insert_nudge(str(p), "n1", "c", predicted_value=0.5)
        outcomes.record_outcome("n1", "explicit", -1.0)
        outcomes.record_outcome("n1", "implicit_action", 1.0)
        report = hit_rate(p, o, "c", window_days=7)
        # mean of -1 + 1 = 0.0
        assert report["actual_mean"] == 0.0

    def test_window_excludes_old_nudges(self, stores) -> None:
        _, outcomes, p, o = stores
        old = datetime.now(UTC) - timedelta(days=30)
        _insert_nudge(str(p), "old", "c", predicted_value=1.0, ts=old)
        _insert_nudge(str(p), "new", "c", predicted_value=0.0)
        report = hit_rate(p, o, "c", window_days=7)
        assert report["n_nudges"] == 1

    def test_fire_counts_included(self, stores) -> None:
        _, outcomes, p, o = stores
        outcomes.record_fire("c", "produced")
        outcomes.record_fire("c", "silent")
        outcomes.record_fire("c", "silent")
        report = hit_rate(p, o, "c", window_days=7)
        assert report["fire_counts"]["produced"] == 1
        assert report["fire_counts"]["silent"] == 2


class TestHitRateAll:
    def test_returns_one_per_distinct_check(self, stores) -> None:
        _, outcomes, p, o = stores
        outcomes.record_fire("a", "produced")
        outcomes.record_fire("b", "produced")
        outcomes.record_fire("b", "silent")
        reports = hit_rate_all(p, o, window_days=7)
        names = {r["check_name"] for r in reports}
        assert names == {"a", "b"}

    def test_empty_when_no_fires(self, stores) -> None:
        _, _, p, o = stores
        assert hit_rate_all(p, o, window_days=7) == []
