"""Metrics — calibration and hit rate aggregation across the nudge → outcome pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict


class HitRateReport(TypedDict):
    """Calibration report for one check over one window."""

    check_name: str
    window_days: int
    n_nudges: int
    n_with_outcome: int
    predicted_mean: float
    actual_mean: float
    calibration_error: float
    fire_counts: dict[str, int]


def hit_rate(
    proactive_log_db: Path | str,
    outcomes_db: Path | str,
    check_name: str,
    window_days: int = 7,
    now: datetime | None = None,
) -> HitRateReport:
    """Compute hit rate + calibration error for a single check.

    predicted_mean: mean of checker self-predicted_value across delivered nudges.
    actual_mean:    mean of observed signal_value across nudges with at least one outcome.
                    Nudges with no outcome contribute 0.0 (silent = neutral, not negative).
    calibration_error: predicted_mean - actual_mean (positive = overconfident).
    """
    anchor = now or datetime.now(UTC)
    since = (anchor - timedelta(days=window_days)).isoformat()

    proactive = _connect(proactive_log_db)
    outcomes = _connect(outcomes_db)

    try:
        cursor = proactive.execute(
            "SELECT id, predicted_value FROM proactive_log WHERE check_name = ? AND timestamp >= ?",
            (check_name, since),
        )
        nudges = [(row["id"], _coerce_float(row["predicted_value"], 0.5)) for row in cursor]

        n_nudges = len(nudges)
        if n_nudges == 0:
            predicted_mean = 0.0
            actual_mean = 0.0
            n_with_outcome = 0
        else:
            predicted_mean = sum(p for _, p in nudges) / n_nudges
            actuals: list[float] = []
            n_with_outcome = 0
            for nudge_id, _ in nudges:
                signals = outcomes.execute(
                    "SELECT signal_value FROM nudge_outcomes WHERE nudge_id = ?",
                    (nudge_id,),
                ).fetchall()
                if signals:
                    n_with_outcome += 1
                    avg = sum(float(r["signal_value"]) for r in signals) / len(signals)
                    actuals.append(_clamp(avg, -1.0, 1.0))
                else:
                    actuals.append(0.0)
            actual_mean = sum(actuals) / n_nudges

        if n_nudges == 0:
            calibration_error = 0.0
        else:
            actual_norm = (actual_mean + 1.0) / 2.0  # rescale [-1, 1] → [0, 1]
            calibration_error = predicted_mean - actual_norm

        fire_counts = _fire_counts(outcomes, check_name, since)
    finally:
        proactive.close()
        outcomes.close()

    return HitRateReport(
        check_name=check_name,
        window_days=window_days,
        n_nudges=n_nudges,
        n_with_outcome=n_with_outcome,
        predicted_mean=round(predicted_mean, 4),
        actual_mean=round(actual_mean, 4),
        calibration_error=round(calibration_error, 4),
        fire_counts=fire_counts,
    )


def hit_rate_all(
    proactive_log_db: Path | str,
    outcomes_db: Path | str,
    window_days: int = 7,
    now: datetime | None = None,
) -> list[HitRateReport]:
    """Hit rate for every check that has fired in the window."""
    anchor = now or datetime.now(UTC)
    since = (anchor - timedelta(days=window_days)).isoformat()

    outcomes = _connect(outcomes_db)
    try:
        cursor = outcomes.execute(
            "SELECT DISTINCT check_name FROM check_fires WHERE fired_at >= ? ORDER BY check_name",
            (since,),
        )
        names = [row["check_name"] for row in cursor.fetchall()]
    finally:
        outcomes.close()

    return [hit_rate(proactive_log_db, outcomes_db, name, window_days, now) for name in names]


def _fire_counts(conn: sqlite3.Connection, check_name: str, since_iso: str) -> dict[str, int]:
    cursor = conn.execute(
        "SELECT result, COUNT(*) as n FROM check_fires "
        "WHERE check_name = ? AND fired_at >= ? GROUP BY result",
        (check_name, since_iso),
    )
    counts = {"produced": 0, "silent": 0, "error": 0}
    for row in cursor.fetchall():
        counts[row["result"]] = row["n"]
    counts["total"] = sum(counts.values())
    return counts


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
