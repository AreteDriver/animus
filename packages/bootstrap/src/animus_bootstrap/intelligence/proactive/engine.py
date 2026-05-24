"""Proactive engine — scheduled checks that produce nudges."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from animus_bootstrap.intelligence.proactive.outcomes import OutcomeStore
from animus_bootstrap.intelligence.proactive.schedule import ScheduleParser

logger = logging.getLogger(__name__)


DEFAULT_OUTCOME_WINDOW_SECONDS = 1800  # 30 minutes


@dataclass(frozen=True)
class NudgePrediction:
    """A checker's self-prediction alongside the nudge text.

    confidence: 0.0–1.0 — checker's certainty that this signal is real.
    predicted_value: 0.0–1.0 — checker's estimate of how useful this nudge will be.
    reasoning: free-text trace, kept short. Used when later debugging calibration.
    """

    text: str
    confidence: float = 0.5
    predicted_value: float = 0.5
    reasoning: str = ""


CheckerReturn = NudgePrediction | str | None
"""Acceptable return types from a checker. ``str`` is shimmed to NudgePrediction."""

ActionObserver = Callable[[datetime, datetime], Awaitable[int] | int]
"""Callable that returns the count of relevant user actions in [since, until)."""


@dataclass
class ProactiveCheck:
    """A scheduled check that may produce a nudge."""

    name: str
    schedule: str  # cron expression or interval ("every 30m")
    checker: Callable[[], Coroutine[Any, Any, CheckerReturn]]
    channels: list[str] = field(default_factory=list)
    priority: str = "normal"  # "low" | "normal" | "high"
    enabled: bool = True
    outcome_window_seconds: int = DEFAULT_OUTCOME_WINDOW_SECONDS


@dataclass
class NudgeRecord:
    """A logged proactive nudge."""

    id: str
    check_name: str
    text: str
    channels: list[str]
    priority: str
    timestamp: datetime
    delivered: bool = False
    confidence: float = 0.5
    predicted_value: float = 0.5
    reasoning: str = ""


class ProactiveEngine:
    """Runs scheduled checks, produces nudges, and observes outcomes for calibration."""

    def __init__(
        self,
        db_path: Path | str,
        quiet_hours: tuple[str, str] = ("22:00", "07:00"),
        send_callback: (Callable[[str, list[str]], Coroutine[Any, Any, None]] | None) = None,
        outcome_store: OutcomeStore | None = None,
        action_observers: list[ActionObserver] | None = None,
    ) -> None:
        self._checks: dict[str, ProactiveCheck] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._quiet_start = quiet_hours[0]
        self._quiet_end = quiet_hours[1]
        self._send_callback = send_callback
        self._next_fires: dict[str, datetime] = {}
        self._outcome_store = outcome_store
        self._action_observers: list[ActionObserver] = list(action_observers or [])
        self._observer_tasks: set[asyncio.Task[None]] = set()
        # SQLite for nudge log
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create proactive_log table and add prediction columns if missing."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS proactive_log (
                id TEXT PRIMARY KEY,
                check_name TEXT NOT NULL,
                text TEXT NOT NULL,
                channels TEXT NOT NULL,
                priority TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.5,
                predicted_value REAL NOT NULL DEFAULT 0.5,
                reasoning TEXT NOT NULL DEFAULT ''
            )
        """)
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(proactive_log)")}
        migrations = (
            (
                "confidence",
                "ALTER TABLE proactive_log ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5",
            ),
            (
                "predicted_value",
                "ALTER TABLE proactive_log ADD COLUMN predicted_value REAL NOT NULL DEFAULT 0.5",
            ),
            (
                "reasoning",
                "ALTER TABLE proactive_log ADD COLUMN reasoning TEXT NOT NULL DEFAULT ''",
            ),
        )
        for col, ddl in migrations:
            if col not in existing:
                self._conn.execute(ddl)
        self._conn.commit()

    def register_check(self, check: ProactiveCheck) -> None:
        """Register a proactive check. Overwrites if same name exists."""
        self._checks[check.name] = check
        logger.info("Registered proactive check: %s", check.name)

    def unregister_check(self, name: str) -> None:
        """Unregister a proactive check. No-op if not found."""
        removed = self._checks.pop(name, None)
        self._next_fires.pop(name, None)
        if removed:
            logger.info("Unregistered proactive check: %s", name)

    def list_checks(self) -> list[ProactiveCheck]:
        """Return all registered checks."""
        return list(self._checks.values())

    def add_action_observer(self, observer: ActionObserver) -> None:
        """Register an action observer used for implicit outcome capture."""
        self._action_observers.append(observer)

    @property
    def outcome_store(self) -> OutcomeStore | None:
        return self._outcome_store

    def is_quiet_hours(self, now: datetime | None = None) -> bool:
        """Check if current time is within quiet hours.

        Handles overnight spans (e.g. 22:00-07:00).
        """
        if now is None:
            now = datetime.now(UTC)

        current_time = now.strftime("%H:%M")
        start = self._quiet_start
        end = self._quiet_end

        if start <= end:
            # Same-day span (e.g. 09:00-17:00)
            return start <= current_time < end
        else:
            # Overnight span (e.g. 22:00-07:00)
            return current_time >= start or current_time < end

    async def run_check(self, name: str) -> NudgeRecord | None:
        """Run a single check immediately.

        Returns nudge if produced, None otherwise.

        Raises:
            KeyError: If the check name is not registered.
        """
        check = self._checks.get(name)
        if check is None:
            msg = f"Check not found: {name!r}"
            raise KeyError(msg)

        try:
            raw = await check.checker()
        except Exception:
            logger.exception("Proactive check %r failed", name)
            self._record_fire(check.name, "error")
            return None

        prediction = self._normalize_return(raw)
        if prediction is None:
            self._record_fire(check.name, "silent")
            return None

        self._record_fire(check.name, "produced")

        nudge = NudgeRecord(
            id=str(uuid.uuid4()),
            check_name=check.name,
            text=prediction.text,
            channels=list(check.channels),
            priority=check.priority,
            timestamp=datetime.now(UTC),
            confidence=_clamp01(prediction.confidence),
            predicted_value=_clamp01(prediction.predicted_value),
            reasoning=prediction.reasoning,
        )

        # Log to DB
        self._conn.execute(
            """
            INSERT INTO proactive_log
                (id, check_name, text, channels, priority, timestamp, delivered,
                 confidence, predicted_value, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nudge.id,
                nudge.check_name,
                nudge.text,
                json.dumps(nudge.channels),
                nudge.priority,
                nudge.timestamp.isoformat(),
                0,
                nudge.confidence,
                nudge.predicted_value,
                nudge.reasoning,
            ),
        )
        self._conn.commit()

        # Deliver via callback if set and not quiet hours
        if self._send_callback and not self.is_quiet_hours():
            try:
                await self._send_callback(nudge.text, nudge.channels)
                nudge.delivered = True
                self._conn.execute(
                    "UPDATE proactive_log SET delivered = 1 WHERE id = ?",
                    (nudge.id,),
                )
                self._conn.commit()
                self._spawn_observer(nudge, check.outcome_window_seconds)
            except Exception:
                logger.exception("Failed to deliver nudge %s", nudge.id)

        return nudge

    async def start(self) -> None:
        """Start the background scheduler loop."""
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Proactive engine started")

    async def stop(self) -> None:
        """Stop the scheduler loop and cancel any in-flight observer tasks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Proactive engine scheduler task cancelled")
            self._task = None

        for task in list(self._observer_tasks):
            task.cancel()
        if self._observer_tasks:
            await asyncio.gather(*self._observer_tasks, return_exceptions=True)
            self._observer_tasks.clear()

        logger.info("Proactive engine stopped")

    async def _scheduler_loop(self) -> None:
        """Main loop: check each registered check, sleep, repeat."""
        while self._running:
            now = datetime.now(UTC)

            for name, check in list(self._checks.items()):
                if not check.enabled:
                    continue

                # Initialize next_fire if not set
                if name not in self._next_fires:
                    self._next_fires[name] = self._compute_next_fire(check, now)

                if now >= self._next_fires[name]:
                    try:
                        await self.run_check(name)
                    except Exception:
                        logger.exception("Scheduler error running check %r", name)
                    # Schedule next fire
                    self._next_fires[name] = self._compute_next_fire(check, datetime.now(UTC))

            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break

    def _compute_next_fire(self, check: ProactiveCheck, after: datetime) -> datetime:
        """Compute the next fire time for a check."""
        if ScheduleParser.is_interval(check.schedule):
            seconds = ScheduleParser.parse_interval(check.schedule)
            return after + timedelta(seconds=seconds)
        else:
            return ScheduleParser.next_cron_fire(check.schedule, after=after)

    def get_nudge_history(self, limit: int = 50) -> list[NudgeRecord]:
        """Return recent nudge history from DB."""
        cursor = self._conn.execute(
            "SELECT * FROM proactive_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        return [self._row_to_nudge(row) for row in rows]

    def clear_history(self, before: datetime | None = None) -> int:
        """Clear nudge history. Returns number of rows deleted."""
        if before is not None:
            cursor = self._conn.execute(
                "DELETE FROM proactive_log WHERE timestamp < ?",
                (before.isoformat(),),
            )
        else:
            cursor = self._conn.execute("DELETE FROM proactive_log")
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the DB connection."""
        self._conn.close()

    @property
    def running(self) -> bool:
        """Whether the scheduler loop is running."""
        return self._running

    def _normalize_return(self, raw: CheckerReturn) -> NudgePrediction | None:
        """Shim: accept legacy str/None returns alongside NudgePrediction."""
        if raw is None:
            return None
        if isinstance(raw, NudgePrediction):
            return raw
        if isinstance(raw, str):
            return NudgePrediction(text=raw)
        logger.warning("Checker returned unexpected type %r; coercing to str", type(raw))
        return NudgePrediction(text=str(raw))

    def _record_fire(self, check_name: str, result: str) -> None:
        """Log a scheduled tick to the outcome store, if configured."""
        if self._outcome_store is None:
            return
        try:
            self._outcome_store.record_fire(check_name, result)
        except Exception:
            logger.exception("Failed to record fire for %r", check_name)

    def _spawn_observer(self, nudge: NudgeRecord, window_seconds: int) -> None:
        """Spawn a background task that records the implicit outcome at window close."""
        if self._outcome_store is None or not self._action_observers:
            return
        if window_seconds <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        task = loop.create_task(self._observe_window(nudge, window_seconds))
        self._observer_tasks.add(task)
        task.add_done_callback(self._observer_tasks.discard)

    async def _observe_window(self, nudge: NudgeRecord, window_seconds: int) -> None:
        """Sleep the window, then poll observers and record an implicit outcome."""
        delivered_at = nudge.timestamp
        try:
            await asyncio.sleep(window_seconds)
        except asyncio.CancelledError:
            return

        until = datetime.now(UTC)
        total = 0
        for observer in list(self._action_observers):
            try:
                result = observer(delivered_at, until)
                if asyncio.iscoroutine(result):
                    result = await result
                total += int(result or 0)
            except Exception:
                logger.exception("Action observer raised; skipping for nudge %s", nudge.id)

        signal_value = 1.0 if total > 0 else 0.0
        source = "implicit_action" if total > 0 else "implicit_silent"
        if self._outcome_store is None:
            return
        try:
            self._outcome_store.record_outcome(
                nudge_id=nudge.id,
                signal_type="implicit_action",
                signal_value=signal_value,
                source=source,
                captured_at=until,
            )
        except Exception:
            logger.exception("Failed to record implicit outcome for nudge %s", nudge.id)

    @staticmethod
    def _row_to_nudge(row: sqlite3.Row) -> NudgeRecord:
        """Convert a DB row to a NudgeRecord."""
        keys = row.keys()
        return NudgeRecord(
            id=row["id"],
            check_name=row["check_name"],
            text=row["text"],
            channels=json.loads(row["channels"]),
            priority=row["priority"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            delivered=bool(row["delivered"]),
            confidence=float(row["confidence"]) if "confidence" in keys else 0.5,
            predicted_value=float(row["predicted_value"]) if "predicted_value" in keys else 0.5,
            reasoning=row["reasoning"] if "reasoning" in keys else "",
        )


def _clamp01(value: float) -> float:
    if value != value:  # NaN guard
        return 0.5
    return max(0.0, min(1.0, float(value)))
