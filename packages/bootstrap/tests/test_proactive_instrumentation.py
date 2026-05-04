"""Tests for the instrumentation pipeline — predictions, fires, observers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from animus_bootstrap.intelligence.proactive.engine import (
    NudgePrediction,
    ProactiveCheck,
    ProactiveEngine,
)
from animus_bootstrap.intelligence.proactive.outcomes import OutcomeStore


@pytest.fixture
def engine_pair(tmp_path):
    outcomes = OutcomeStore(tmp_path / "outcomes.db")
    engine = ProactiveEngine(
        db_path=tmp_path / "proactive.db",
        quiet_hours=("00:00", "00:00"),  # never quiet
        outcome_store=outcomes,
    )
    yield engine, outcomes
    engine.close()
    outcomes.close()


class TestPredictionShim:
    @pytest.mark.asyncio
    async def test_str_return_is_shimmed_with_default_prediction(self, engine_pair) -> None:
        engine, _ = engine_pair

        async def checker():
            return "hello"

        engine.register_check(ProactiveCheck(name="legacy", schedule="every 1h", checker=checker))
        nudge = await engine.run_check("legacy")
        assert nudge is not None
        assert nudge.text == "hello"
        assert nudge.confidence == 0.5
        assert nudge.predicted_value == 0.5
        assert nudge.reasoning == ""

    @pytest.mark.asyncio
    async def test_nudge_prediction_propagates(self, engine_pair) -> None:
        engine, _ = engine_pair

        async def checker():
            return NudgePrediction(
                text="urgent",
                confidence=0.8,
                predicted_value=0.9,
                reasoning="overdue task",
            )

        engine.register_check(ProactiveCheck(name="modern", schedule="every 1h", checker=checker))
        nudge = await engine.run_check("modern")
        assert nudge is not None
        assert nudge.confidence == 0.8
        assert nudge.predicted_value == 0.9
        assert nudge.reasoning == "overdue task"

    @pytest.mark.asyncio
    async def test_prediction_values_are_clamped(self, engine_pair) -> None:
        engine, _ = engine_pair

        async def checker():
            return NudgePrediction(text="x", confidence=2.0, predicted_value=-0.5)

        engine.register_check(ProactiveCheck(name="oob", schedule="every 1h", checker=checker))
        nudge = await engine.run_check("oob")
        assert nudge is not None
        assert nudge.confidence == 1.0
        assert nudge.predicted_value == 0.0

    @pytest.mark.asyncio
    async def test_none_return_records_silent_fire(self, engine_pair) -> None:
        engine, outcomes = engine_pair

        async def checker():
            return None

        engine.register_check(ProactiveCheck(name="quiet", schedule="every 1h", checker=checker))
        result = await engine.run_check("quiet")
        assert result is None
        fires = outcomes.fires_for("quiet")
        assert len(fires) == 1
        assert fires[0].result == "silent"

    @pytest.mark.asyncio
    async def test_exception_records_error_fire(self, engine_pair) -> None:
        engine, outcomes = engine_pair

        async def checker():
            raise RuntimeError("boom")

        engine.register_check(ProactiveCheck(name="bad", schedule="every 1h", checker=checker))
        result = await engine.run_check("bad")
        assert result is None
        fires = outcomes.fires_for("bad")
        assert len(fires) == 1
        assert fires[0].result == "error"

    @pytest.mark.asyncio
    async def test_produced_fire_recorded(self, engine_pair) -> None:
        engine, outcomes = engine_pair

        async def checker():
            return "x"

        engine.register_check(ProactiveCheck(name="ok", schedule="every 1h", checker=checker))
        await engine.run_check("ok")
        fires = outcomes.fires_for("ok")
        assert len(fires) == 1
        assert fires[0].result == "produced"


class TestSchemaMigration:
    def test_existing_db_without_columns_gets_migrated(self, tmp_path) -> None:
        import sqlite3

        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            """
            CREATE TABLE proactive_log (
                id TEXT PRIMARY KEY,
                check_name TEXT NOT NULL,
                text TEXT NOT NULL,
                channels TEXT NOT NULL,
                priority TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO proactive_log VALUES "
            "('id1', 'c', 'hi', '[]', 'normal', '2026-01-01T00:00:00+00:00', 0)"
        )
        conn.commit()
        conn.close()

        engine = ProactiveEngine(db_path=path)
        try:
            history = engine.get_nudge_history()
            assert len(history) == 1
            assert history[0].confidence == 0.5
            assert history[0].predicted_value == 0.5
            assert history[0].reasoning == ""
        finally:
            engine.close()


class TestObserverAndDelivery:
    @pytest.mark.asyncio
    async def test_observer_records_implicit_positive_when_action_seen(self, tmp_path) -> None:
        outcomes = OutcomeStore(tmp_path / "outcomes.db")
        send = AsyncMock()
        engine = ProactiveEngine(
            db_path=tmp_path / "proactive.db",
            quiet_hours=("00:00", "00:00"),
            send_callback=send,
            outcome_store=outcomes,
            action_observers=[lambda since, until: 3],
        )
        try:

            async def checker():
                return "hi"

            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=0,  # short-circuit; we'll await manually
                )
            )
            # outcome_window_seconds=0 means no observer spawned; use a real small window
            engine.unregister_check("c")
            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=1,
                )
            )
            nudge = await engine.run_check("c")
            assert nudge is not None
            send.assert_awaited_once()

            await asyncio.sleep(1.2)

            results = outcomes.outcomes_for(nudge.id)
            assert len(results) == 1
            assert results[0].signal_type == "implicit_action"
            assert results[0].signal_value == 1.0
            assert results[0].source == "implicit_action"
        finally:
            await engine.stop()
            engine.close()
            outcomes.close()

    @pytest.mark.asyncio
    async def test_observer_records_silent_when_no_action(self, tmp_path) -> None:
        outcomes = OutcomeStore(tmp_path / "outcomes.db")
        send = AsyncMock()
        engine = ProactiveEngine(
            db_path=tmp_path / "proactive.db",
            quiet_hours=("00:00", "00:00"),
            send_callback=send,
            outcome_store=outcomes,
            action_observers=[lambda since, until: 0],
        )
        try:

            async def checker():
                return "hi"

            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=1,
                )
            )
            nudge = await engine.run_check("c")
            assert nudge is not None
            await asyncio.sleep(1.2)

            results = outcomes.outcomes_for(nudge.id)
            assert len(results) == 1
            assert results[0].signal_value == 0.0
            assert results[0].source == "implicit_silent"
        finally:
            await engine.stop()
            engine.close()
            outcomes.close()

    @pytest.mark.asyncio
    async def test_observer_skipped_when_no_outcome_store(self, tmp_path) -> None:
        send = AsyncMock()
        engine = ProactiveEngine(
            db_path=tmp_path / "proactive.db",
            quiet_hours=("00:00", "00:00"),
            send_callback=send,
            outcome_store=None,
            action_observers=[lambda since, until: 5],
        )
        try:

            async def checker():
                return "hi"

            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=1,
                )
            )
            nudge = await engine.run_check("c")
            assert nudge is not None
            await asyncio.sleep(1.2)
            # No outcome store → no records, but no crash either
        finally:
            await engine.stop()
            engine.close()

    @pytest.mark.asyncio
    async def test_observer_handles_async_callable(self, tmp_path) -> None:
        outcomes = OutcomeStore(tmp_path / "outcomes.db")
        send = AsyncMock()

        async def async_observer(since, until):
            return 2

        engine = ProactiveEngine(
            db_path=tmp_path / "proactive.db",
            quiet_hours=("00:00", "00:00"),
            send_callback=send,
            outcome_store=outcomes,
            action_observers=[async_observer],
        )
        try:

            async def checker():
                return "hi"

            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=1,
                )
            )
            nudge = await engine.run_check("c")
            assert nudge is not None
            await asyncio.sleep(1.2)
            results = outcomes.outcomes_for(nudge.id)
            assert len(results) == 1
            assert results[0].signal_value == 1.0
        finally:
            await engine.stop()
            engine.close()
            outcomes.close()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_observer_tasks(self, tmp_path) -> None:
        outcomes = OutcomeStore(tmp_path / "outcomes.db")
        send = AsyncMock()
        engine = ProactiveEngine(
            db_path=tmp_path / "proactive.db",
            quiet_hours=("00:00", "00:00"),
            send_callback=send,
            outcome_store=outcomes,
            action_observers=[lambda since, until: 1],
        )
        try:

            async def checker():
                return "hi"

            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=300,
                )
            )
            nudge = await engine.run_check("c")
            assert nudge is not None
            assert engine._observer_tasks  # task spawned
            await engine.stop()
            assert not engine._observer_tasks
        finally:
            engine.close()
            outcomes.close()

    @pytest.mark.asyncio
    async def test_observer_failure_does_not_block_others(self, tmp_path) -> None:
        outcomes = OutcomeStore(tmp_path / "outcomes.db")
        send = AsyncMock()

        def bad(since, until):
            raise RuntimeError("nope")

        def good(since, until):
            return 4

        engine = ProactiveEngine(
            db_path=tmp_path / "proactive.db",
            quiet_hours=("00:00", "00:00"),
            send_callback=send,
            outcome_store=outcomes,
            action_observers=[bad, good],
        )
        try:

            async def checker():
                return "hi"

            engine.register_check(
                ProactiveCheck(
                    name="c",
                    schedule="every 1h",
                    checker=checker,
                    outcome_window_seconds=1,
                )
            )
            nudge = await engine.run_check("c")
            await asyncio.sleep(1.2)
            results = outcomes.outcomes_for(nudge.id)
            assert len(results) == 1
            assert results[0].signal_value == 1.0  # good observer counted 4
        finally:
            await engine.stop()
            engine.close()
            outcomes.close()
